package googleadk_test

import (
	"context"
	"errors"
	"iter"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/nexus-rpc/sdk-go/nexus"
	"github.com/stretchr/testify/require"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/converter"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	remoteagent "google.golang.org/adk/v2/agent/remoteagent/v2"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"

	"go.temporal.io/googleadk"
)

// toolSpec describes an in-workflow function tool the agent advertises to the
// model. Its handler runs in-workflow (on the deterministic Temporal dispatcher)
// and returns a constant result; a test can assert its effect via the returned
// events or a package-level flag.
type toolSpec struct {
	Name        string
	Description string
}

// subAgentSpec describes a child agent in a SubAgents tree.
type subAgentSpec struct {
	Name        string
	Description string
	ModelName   string
	// RemoteCardURL, when set, makes the child ADK's remote A2A agent, with its
	// agent card at this URL, instead of an LLM agent. It uses the plugin's
	// card and client providers.
	RemoteCardURL string
	// WorkflowSideA2A makes that remote agent use ADK's own providers instead,
	// which fetch the card and send messages from Workflow code.
	WorkflowSideA2A bool
}

// remoteSubAgent is a remote A2A sub-agent named "remote" with its card at url.
func remoteSubAgent(url string) subAgentSpec {
	return subAgentSpec{Name: "remote", Description: "remote A2A agent", RemoteCardURL: url}
}

// transferToRemoteModel is a root model that transfers to the "remote" agent.
func transferToRemoteModel() googleadk.ModelFactory {
	return scriptedModelFactory(
		googleadk.FunctionCallResponse("c1", "transfer_to_agent", map[string]any{"agent_name": "remote"}),
	)
}

// a2aServer configures startA2AServer.
type a2aServer struct {
	// Reply answers each message by yielding events or an error.
	Reply func(ec *a2asrv.ExecutorContext, yield func(a2a.Event, error) bool)
	// Token, when set, makes the server answer A2A calls that lack
	// "Authorization: Bearer <Token>" with HTTP 401. The card stays public.
	Token string
	// StreamErrorAfter, when positive, ends every stream with an error after
	// that many events.
	StreamErrorAfter int
}

// streamErrorHandler ends every stream with an error after its first n events.
type streamErrorHandler struct {
	a2asrv.RequestHandler
	n int
}

func (h streamErrorHandler) SendStreamingMessage(ctx context.Context, req *a2a.SendMessageRequest) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		sent := 0
		for ev, err := range h.RequestHandler.SendStreamingMessage(ctx, req) {
			if !yield(ev, err) || err != nil {
				return
			}
			if sent++; sent == h.n {
				yield(nil, errors.New("stream broke"))
				return
			}
		}
	}
}

// a2aReply answers every message with ev and err.
func a2aReply(ev a2a.Event, err error) func(*a2asrv.ExecutorContext, func(a2a.Event, error) bool) {
	return func(_ *a2asrv.ExecutorContext, yield func(a2a.Event, error) bool) { yield(ev, err) }
}

// startA2AServer starts an in-process A2A server and returns its URL and a
// count of the requests it received.
func startA2AServer(t *testing.T, cfg a2aServer) (string, *atomic.Int32) {
	t.Helper()
	var requests atomic.Int32
	mux := http.NewServeMux()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		if cfg.Token != "" && r.URL.Path != a2asrv.WellKnownAgentCardPath && r.Header.Get("Authorization") != "Bearer "+cfg.Token {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		mux.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	card := &a2a.AgentCard{
		Name: "remote", Description: "Answers questions.", Version: "1.0.0",
		SupportedInterfaces: []*a2a.AgentInterface{{
			URL: srv.URL + "/", ProtocolBinding: a2a.TransportProtocolJSONRPC, ProtocolVersion: a2a.Version,
		}},
		DefaultInputModes: []string{"text/plain"}, DefaultOutputModes: []string{"text/plain"},
		Capabilities: a2a.AgentCapabilities{Streaming: true},
	}
	executor := a2asrv.AgentExecutorFunc(func(_ context.Context, ec *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
		return func(yield func(a2a.Event, error) bool) { cfg.Reply(ec, yield) }
	})
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(card))
	handler := a2asrv.NewHandler(executor)
	if cfg.StreamErrorAfter > 0 {
		handler = streamErrorHandler{RequestHandler: handler, n: cfg.StreamErrorAfter}
	}
	mux.Handle("/", a2asrv.NewJSONRPCHandler(handler))
	return srv.URL, &requests
}

// runInput is the serializable input to agentRunWorkflow.
type runInput struct {
	ModelName        string
	UserMessage      string
	Tools            []toolSpec
	SubAgents        []subAgentSpec
	MCPToolset       string
	Sequential       bool
	StreamingTopic   string
	PerModelTimeouts map[string]time.Duration
	UseCustomSummary bool
	// SSE runs ADK in streaming mode without a model streaming topic: model
	// calls still return one response, and a remote A2A agent streams.
	SSE bool
	// SequentialRoot makes the root a sequential agent of the sub-agents
	// instead of an LLM agent.
	SequentialRoot bool
	// RunnerPlugin adds googleadk.NewRunnerPlugin to the runner.
	RunnerPlugin bool
}

// runResult is the serializable output of agentRunWorkflow.
type runResult struct {
	Texts                []string
	FunctionCalls        []string
	ToolResponses        []string
	ToolResponsePayloads map[string]map[string]any
	Authors              []string
	// Errors holds the error messages of the run's events.
	Errors []string
	// StateFlag carries a session-state value a test workflow read back after the
	// run (used by the in-workflow state-mutation regression test). Empty when
	// unused.
	StateFlag string
}

// summaryFnCalled is flipped by the custom summary function installed when
// runInput.UseCustomSummary is set (via googleadk.WithModelSummary), so a test
// can prove the function ran on the dispatch path rather than merely being
// stored on the model.
var summaryFnCalled atomic.Bool

// inWorkflowToolRan is flipped by a pure-compute in-workflow function tool,
// proving its handler ran inside the workflow coroutine rather than in an
// Activity.
var inWorkflowToolRan atomic.Bool

// agentBuild is the fully-built input to runAgent: a real ADK agent tree plus the
// model options and context options used to drive it.
type agentBuild struct {
	ctxOpts        []googleadk.ContextOption
	modelName      string
	modelOpts      []googleadk.ModelOption
	streamingTopic string
	sse            bool
	sequentialRoot bool
	runnerPlugin   bool
	userMessage    string
	tools          []tool.Tool
	toolsets       []tool.Toolset
	subAgents      []agent.Agent
	// sessionService overrides the default in-memory session service. Tests that
	// need to inspect session state/events after the run (state mutation,
	// continue-as-new) supply their own and read it back via ExportSession.
	sessionService session.Service
	// appName/userID/sessionID override the default run identity so a caller can
	// look the session up afterward. Defaults are used when empty.
	appName   string
	userID    string
	sessionID string
}

// runAgent builds a real ADK llmagent + runner using googleadk.NewModel as the
// agent's Model and the bridged context, then drives runner.Run — the native ADK
// entry point — entirely inside the workflow. Model calls are dispatched to the
// InvokeModel Activity; ordinary function tools run in-workflow; ActivityAsTool
// and MCP tools dispatch their own Activities.
func runAgent(ctx workflow.Context, b agentBuild) (runResult, error) {
	var root agent.Agent
	var err error
	if b.sequentialRoot {
		root, err = sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{Name: "assistant", Description: "root sequence", SubAgents: b.subAgents},
		})
	} else {
		root, err = llmagent.New(llmagent.Config{
			Name:        "assistant",
			Description: "root assistant",
			Model:       googleadk.NewModel(b.modelName, b.modelOpts...),
			Instruction: "be helpful",
			Tools:       b.tools,
			Toolsets:    b.toolsets,
			SubAgents:   b.subAgents,
		})
	}
	if err != nil {
		return runResult{}, err
	}

	svc := b.sessionService
	if svc == nil {
		svc = session.InMemoryService()
	}
	appName := b.appName
	if appName == "" {
		appName = "test-app"
	}
	userID := b.userID
	if userID == "" {
		userID = "user-1"
	}
	sessionID := b.sessionID
	if sessionID == "" {
		sessionID = "session-1"
	}
	var plugins []*plugin.Plugin
	if b.runnerPlugin {
		p, perr := googleadk.NewRunnerPlugin(root)
		if perr != nil {
			return runResult{}, perr
		}
		plugins = append(plugins, p)
	}
	r, err := runner.New(runner.Config{
		AppName:           appName,
		Agent:             root,
		SessionService:    svc,
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		return runResult{}, err
	}

	runCfg := agent.RunConfig{}
	if b.sse {
		runCfg.StreamingMode = agent.StreamingModeSSE
	}
	if b.streamingTopic != "" {
		runCfg.StreamingMode = agent.StreamingModeSSE
		// Streaming to external consumers requires the workflow-side stream server
		// so the InvokeModel Activity's published chunks have somewhere to land.
		if err := googleadk.StreamServer(ctx); err != nil {
			return runResult{}, err
		}
	}

	adkCtx := googleadk.NewContext(ctx, b.ctxOpts...)
	msg := genai.NewContentFromText(b.userMessage, genai.RoleUser)

	var res runResult
	for ev, rerr := range r.Run(adkCtx, userID, sessionID, msg, runCfg) {
		if rerr != nil {
			return res, rerr
		}
		collectEvent(&res, ev)
	}
	return res, nil
}

// collectEvent folds one ADK event into the serializable runResult.
func collectEvent(res *runResult, ev *session.Event) {
	if ev == nil {
		return
	}
	res.Authors = append(res.Authors, ev.Author)
	if ev.ErrorMessage != "" {
		res.Errors = append(res.Errors, ev.ErrorMessage)
	}
	if ev.Content == nil {
		return
	}
	for _, p := range ev.Content.Parts {
		if p == nil {
			continue
		}
		if p.Text != "" {
			res.Texts = append(res.Texts, p.Text)
		}
		if p.FunctionCall != nil {
			res.FunctionCalls = append(res.FunctionCalls, p.FunctionCall.Name)
		}
		if p.FunctionResponse != nil {
			res.ToolResponses = append(res.ToolResponses, p.FunctionResponse.Name)
			if res.ToolResponsePayloads == nil {
				res.ToolResponsePayloads = make(map[string]map[string]any)
			}
			res.ToolResponsePayloads[p.FunctionResponse.Name] = p.FunctionResponse.Response
		}
	}
}

// agentRunWorkflow assembles an agentBuild from a serializable runInput and runs
// it. Tools are REAL in-workflow function tools whose no-op handlers run on the
// deterministic dispatcher; sub-agents each get their own NewModel; the MCP
// toolset (if any) is wired through NewMCPToolset.
func agentRunWorkflow(ctx workflow.Context, in runInput) (runResult, error) {
	var modelOpts []googleadk.ModelOption
	if in.StreamingTopic != "" {
		modelOpts = append(modelOpts, googleadk.WithStreaming(in.StreamingTopic, 0))
	}
	if in.UseCustomSummary {
		modelOpts = append(modelOpts, googleadk.WithModelSummary(func(*model.LLMRequest) string {
			summaryFnCalled.Store(true)
			return "custom-summary"
		}))
	}
	if d, ok := in.PerModelTimeouts[in.ModelName]; ok {
		modelOpts = append(modelOpts, googleadk.WithModelActivityOptions(workflow.ActivityOptions{
			StartToCloseTimeout: d,
		}))
	}

	var tools []tool.Tool
	for _, ts := range in.Tools {
		t, terr := noopTool(ts.Name, ts.Description)
		if terr != nil {
			return runResult{}, terr
		}
		tools = append(tools, t)
	}

	var toolsets []tool.Toolset
	if in.MCPToolset != "" {
		toolsets = append(toolsets, googleadk.NewMCPToolset(googleadk.MCPToolsetOptions{Name: in.MCPToolset}))
	}

	var subAgents []agent.Agent
	for _, sa := range in.SubAgents {
		if sa.RemoteCardURL != "" {
			cfg := remoteagent.A2AConfig{
				Name:              sa.Name,
				Description:       sa.Description,
				AgentCardProvider: googleadk.NewA2AAgentCardProvider(sa.RemoteCardURL, workflow.ActivityOptions{}),
				ClientProvider:    googleadk.NewA2AClientProvider(workflow.ActivityOptions{}),
			}
			if sa.WorkflowSideA2A {
				cfg.AgentCardProvider = remoteagent.NewAgentCardProvider(sa.RemoteCardURL)
				cfg.ClientProvider = nil
			}
			remote, rerr := remoteagent.NewA2A(cfg)
			if rerr != nil {
				return runResult{}, rerr
			}
			subAgents = append(subAgents, remote)
			continue
		}
		child, cerr := llmagent.New(llmagent.Config{
			Name:        sa.Name,
			Description: sa.Description,
			Model:       googleadk.NewModel(sa.ModelName),
			Instruction: "child agent",
		})
		if cerr != nil {
			return runResult{}, cerr
		}
		subAgents = append(subAgents, child)
	}

	var ctxOpts []googleadk.ContextOption
	if in.Sequential {
		ctxOpts = append(ctxOpts, googleadk.WithSequentialToolFanout())
	}

	return runAgent(ctx, agentBuild{
		ctxOpts:        ctxOpts,
		modelName:      in.ModelName,
		modelOpts:      modelOpts,
		streamingTopic: in.StreamingTopic,
		sse:            in.SSE,
		sequentialRoot: in.SequentialRoot,
		runnerPlugin:   in.RunnerPlugin,
		userMessage:    in.UserMessage,
		tools:          tools,
		toolsets:       toolsets,
		subAgents:      subAgents,
	})
}

// ----------------------------------------------------------------------------
// Test helpers.
// ----------------------------------------------------------------------------

// noopTool builds a real in-workflow function tool whose handler is a no-op. The
// agent advertises it to the model; its Run executes in-workflow.
func noopTool(name, description string) (tool.Tool, error) {
	return functiontool.New[map[string]any, map[string]any](
		functiontool.Config{Name: name, Description: description},
		func(agent.Context, map[string]any) (map[string]any, error) { return map[string]any{}, nil },
	)
}

// funcTool builds a real in-workflow function tool with the given handler.
func funcTool(name string, handler func(agent.Context, map[string]any) (map[string]any, error)) (tool.Tool, error) {
	return functiontool.New[map[string]any, map[string]any](
		functiontool.Config{Name: name, Description: name + " tool"},
		handler,
	)
}

// testRegistry adapts a TestWorkflowEnvironment to worker.Registry so the real
// Activities.Register path (the exported worker-wiring helper) is exercised by
// the tests. The Nexus method is unused here.
type testRegistry struct {
	*testsuite.TestWorkflowEnvironment
}

func (testRegistry) RegisterNexusService(*nexus.Service) {}

// activityCounter records how many times each Activity type is scheduled, so
// tests can assert exact per-Activity counts (the side-effects criterion).
type activityCounter struct {
	mu     sync.Mutex
	counts map[string]int
}

func newActivityCounter() *activityCounter {
	return &activityCounter{counts: map[string]int{}}
}

func (c *activityCounter) record(name string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.counts[name]++
}

func (c *activityCounter) get(name string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.counts[name]
}

// wireActivities registers the plugin Activities (via the real Register path)
// onto env and installs an activity-start counter.
func wireActivities(t *testing.T, env *testsuite.TestWorkflowEnvironment, cfg googleadk.Config) *activityCounter {
	t.Helper()
	acts, err := googleadk.NewActivities(cfg)
	require.NoError(t, err)
	acts.Register(testRegistry{env})

	counter := newActivityCounter()
	env.SetOnActivityStartedListener(func(info *activity.Info, _ context.Context, _ converter.EncodedValues) {
		counter.record(info.ActivityType.Name)
	})
	return counter
}

// newEnv builds a test workflow environment with agentRunWorkflow registered and
// the plugin Activities wired from cfg.
func newEnv(t *testing.T, s *testsuite.WorkflowTestSuite, cfg googleadk.Config) (*testsuite.TestWorkflowEnvironment, *activityCounter) {
	t.Helper()
	env := s.NewTestWorkflowEnvironment()
	env.RegisterWorkflow(agentRunWorkflow)
	counter := wireActivities(t, env, cfg)
	return env, counter
}

// scriptedModelFactory returns a ModelFactory that yields a single shared
// FakeModel replaying the given scripted responses. Sharing one instance across
// Activity invocations lets the scripted sequence advance turn by turn (turn 1
// returns the first response, turn 2 the second, ...).
func scriptedModelFactory(responses ...*model.LLMResponse) googleadk.ModelFactory {
	fm := googleadk.NewFakeModel(responses...)
	return func(context.Context, string) (model.LLM, error) {
		return fm, nil
	}
}

// recordingTool builds a real in-workflow function tool whose handler records
// the args it was called with and returns the supplied result.
func recordingTool(t *testing.T, name string, result map[string]any, onCall func(map[string]any)) tool.Tool {
	t.Helper()
	ft, err := funcTool(name, func(_ agent.Context, args map[string]any) (map[string]any, error) {
		if onCall != nil {
			onCall(args)
		}
		return result, nil
	})
	require.NoError(t, err)
	return ft
}
