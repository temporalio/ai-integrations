package googleadk_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/converter"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"

	"go.temporal.io/googleadk"
)

// recordingModel wraps a FakeModel and records the tool declarations carried in
// each LLMRequest it receives. Because LLMRequest.Tools is json:"-" but
// LLMRequest.Config.Tools (the genai declarations) crosses the wire, this proves
// the MCP tool's full parameter schema reached the model via ListMcpTools.
type recordingModel struct {
	inner *googleadk.FakeModel

	mu    sync.Mutex
	decls []*genai.FunctionDeclaration
}

func newRecordingModel(name string, responses ...*model.LLMResponse) *recordingModel {
	return &recordingModel{inner: googleadk.NewFakeModel(responses...).WithName(name)}
}

func (m *recordingModel) Name() string { return m.inner.Name() }

func (m *recordingModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	if req.Config != nil {
		for _, gt := range req.Config.Tools {
			if gt != nil {
				m.decls = append(m.decls, gt.FunctionDeclarations...)
			}
		}
	}
	m.mu.Unlock()
	return m.inner.GenerateContent(ctx, req, stream)
}

// declFor returns the recorded declaration with the given name, if any.
func (m *recordingModel) declFor(name string) *genai.FunctionDeclaration {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, d := range m.decls {
		if d != nil && d.Name == name {
			return d
		}
	}
	return nil
}

// searchSchema is the parameter schema the fake MCP server advertises for its
// "search" tool. ListMcpTools must carry it across to the model.
func searchSchema() *genai.Schema {
	return &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"query": {Type: genai.TypeString, Description: "the search query"},
		},
		Required: []string{"query"},
	}
}

// TestMcpListToolsCarriesParameters proves NewMCPToolset advertises the full
// declaration (name + description + PARAMETERS) of each remote MCP tool, not just
// {name, description}: the parameter schema travels ListMcpTools -> packTool ->
// the InvokeModel request and is observed by the model.
func TestMcpListToolsCarriesParameters(t *testing.T) {
	rm := newRecordingModel("fake-model",
		googleadk.FunctionCallResponse("c1", "search", map[string]any{"query": "temporal"}),
		googleadk.TextResponse("found it"),
	)
	server := googleadk.NewFakeMCPServer("search-server").AddTool(
		"search", "search the web", searchSchema(),
		func(args map[string]any) (map[string]any, error) {
			return map[string]any{"hits": 1}, nil
		},
	)

	var s testsuite.WorkflowTestSuite
	env, counter := newEnv(t, &s, googleadk.Config{
		Models: map[string]googleadk.ModelFactory{
			"fake-model": func(context.Context, string) (model.LLM, error) { return rm, nil },
		},
		MCPToolsets: map[string]googleadk.MCPFactory{"search-server": server.Factory()},
	})

	env.ExecuteWorkflow(agentRunWorkflow, runInput{
		ModelName:   "fake-model",
		UserMessage: "search the web",
		MCPToolset:  "search-server",
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	// The model saw the MCP tool with its full parameter schema.
	decl := rm.declFor("search")
	require.NotNil(t, decl, "model never saw the MCP tool declaration")
	require.NotNil(t, decl.Parameters, "MCP tool declaration carried no parameters")
	assert.Contains(t, decl.Parameters.Properties, "query")

	// The tools were listed once per model step, on every ADK release, and
	// called once. ADK v2.6.0 asks the toolset for its tools twice per step;
	// a second listing would add an Activity, and Histories recorded with
	// another ADK release would fail replay.
	assert.Equal(t, 2, counter.get(googleadk.InvokeModelActivityName))
	assert.Equal(t, 2, counter.get(googleadk.ListMcpToolsActivityName), "one listing per model step")
	assert.Equal(t, 1, counter.get(googleadk.CallMcpToolActivityName))
}

// TestMcpToolCallRoundTrip proves a model-issued MCP tool call is short-circuited
// into the CallMcpTool Activity, executed worker-side against the live (here,
// fake) toolset, and its result fed back into the agent loop.
func TestMcpToolCallRoundTrip(t *testing.T) {
	var (
		mu      sync.Mutex
		gotArgs map[string]any
	)
	server := googleadk.NewFakeMCPServer("search-server").AddTool(
		"search", "search the web", searchSchema(),
		func(args map[string]any) (map[string]any, error) {
			mu.Lock()
			gotArgs = args
			mu.Unlock()
			return map[string]any{"answer": "42"}, nil
		},
	)

	var s testsuite.WorkflowTestSuite
	env, counter := newEnv(t, &s, googleadk.Config{
		Models: map[string]googleadk.ModelFactory{
			"fake-model": scriptedModelFactory(
				googleadk.FunctionCallResponse("c1", "search", map[string]any{"query": "meaning of life"}),
				googleadk.TextResponse("the answer is 42"),
			),
		},
		MCPToolsets: map[string]googleadk.MCPFactory{"search-server": server.Factory()},
	})

	env.ExecuteWorkflow(agentRunWorkflow, runInput{
		ModelName:   "fake-model",
		UserMessage: "what is the meaning of life",
		MCPToolset:  "search-server",
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var res runResult
	require.NoError(t, env.GetWorkflowResult(&res))
	assert.Contains(t, res.ToolResponses, "search")
	assert.Contains(t, res.Texts, "the answer is 42")
	assert.Equal(t, 1, counter.get(googleadk.CallMcpToolActivityName))

	mu.Lock()
	assert.Equal(t, "meaning of life", gotArgs["query"])
	mu.Unlock()
}

// ruleModel answers each request with the response its rule picks from the
// request, so concurrent runs that share it each get their own script.
type ruleModel struct {
	name string
	rule func(req *model.LLMRequest) *model.LLMResponse

	mu    sync.Mutex
	decls [][]string
}

func (m *ruleModel) Name() string { return m.name }

func (m *ruleModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	names := declaredNames(req)
	m.mu.Lock()
	m.decls = append(m.decls, names)
	m.mu.Unlock()
	resp := m.rule(req)
	return func(yield func(*model.LLMResponse, error) bool) { yield(resp, nil) }
}

func (m *ruleModel) factory() googleadk.ModelFactory {
	return func(context.Context, string) (model.LLM, error) { return m, nil }
}

// declaredNames returns the names of the tools a model request declares.
func declaredNames(req *model.LLMRequest) []string {
	var names []string
	if req.Config != nil {
		for _, gt := range req.Config.Tools {
			if gt != nil {
				for _, d := range gt.FunctionDeclarations {
					names = append(names, d.Name)
				}
			}
		}
	}
	return names
}

// declared returns the tool names each model request declared, in order.
func (m *ruleModel) declared() [][]string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.decls
}

// functionResponses counts the function responses in a request's contents.
func functionResponses(req *model.LLMRequest) int {
	n := 0
	for _, c := range req.Contents {
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			if p.FunctionResponse != nil {
				n++
			}
		}
	}
	return n
}

// lastIsFunctionResponse reports whether the request's last content carries a
// function response.
func lastIsFunctionResponse(req *model.LLMRequest) bool {
	if len(req.Contents) == 0 {
		return false
	}
	for _, p := range req.Contents[len(req.Contents)-1].Parts {
		if p.FunctionResponse != nil {
			return true
		}
	}
	return false
}

// lastUserText returns the text of the request's last user content.
func lastUserText(req *model.LLMRequest) string {
	for i := len(req.Contents) - 1; i >= 0; i-- {
		c := req.Contents[i]
		if c == nil || c.Role != genai.RoleUser {
			continue
		}
		for _, p := range c.Parts {
			if p.Text != "" {
				return p.Text
			}
		}
	}
	return ""
}

// commandLog records the Activities and timers a test workflow schedules, in
// order.
type commandLog struct {
	mu  sync.Mutex
	log []string
}

func (l *commandLog) add(s string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.log = append(l.log, s)
}

func (l *commandLog) get() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.log...)
}

func (l *commandLog) count(s string) int {
	n := 0
	for _, e := range l.get() {
		if e == s {
			n++
		}
	}
	return n
}

// watchCommands records env's Activity starts and timers in order, each timer
// as "Timer <duration>". It replaces the activityCounter listener newEnv
// installs.
func watchCommands(env *testsuite.TestWorkflowEnvironment) *commandLog {
	l := &commandLog{}
	env.SetOnActivityStartedListener(func(info *activity.Info, _ context.Context, _ converter.EncodedValues) {
		l.add(info.ActivityType.Name)
	})
	env.SetOnTimerScheduledListener(func(_ string, d time.Duration) { l.add("Timer " + d.String()) })
	return l
}

// sleepingInstructionWorkflow runs two turns of an MCP agent whose
// InstructionProvider sleeps on a 1s workflow timer before each model call, the
// way an instruction that reads workflow state through a command would. With
// filterSleeps, the toolset's ToolFilter also sleeps, on a 2s timer.
func sleepingInstructionWorkflow(ctx workflow.Context, filterSleeps bool) error {
	sleep := func(rctx agent.ReadonlyContext, d time.Duration) error {
		wfCtx, ok := googleadk.WorkflowContext(rctx)
		if !ok {
			return errors.New("no workflow context")
		}
		return workflow.Sleep(wfCtx, d)
	}
	opts := googleadk.MCPToolsetOptions{Name: "search-server"}
	if filterSleeps {
		opts.ToolFilter = func(rctx agent.ReadonlyContext, _ tool.Tool) bool {
			return sleep(rctx, 2*time.Second) == nil
		}
	}
	root, err := llmagent.New(llmagent.Config{
		Name:  "assistant",
		Model: googleadk.NewModel("fake-model"),
		InstructionProvider: func(rctx agent.ReadonlyContext) (string, error) {
			if err := sleep(rctx, time.Second); err != nil {
				return "", err
			}
			return "be helpful", nil
		},
		Toolsets: []tool.Toolset{googleadk.NewMCPToolset(opts)},
	})
	if err != nil {
		return err
	}
	r, err := runner.New(runner.Config{AppName: "test-app", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		return err
	}
	adkCtx := googleadk.NewContext(ctx)
	for _, text := range []string{"search", "thanks"} {
		for _, rerr := range r.Run(adkCtx, "user-1", "session-1", genai.NewContentFromText(text, genai.RoleUser), agent.RunConfig{}) {
			if rerr != nil {
				return rerr
			}
		}
	}
	return nil
}

// TestMcpListsAtTheSamePointInEveryStep proves the MCP toolset lists at a point
// that does not depend on the ADK release: in the first model step of each
// turn before the request processors (ADK calls Tools there on every release),
// and in every later step after them, in ProcessRequest. ADK v2.6.0 also calls
// Tools before the request processors of every step; a listing there would put
// ListMcpTools before the instruction's timer, where ADK v2.5.0 and earlier put
// it after, and Histories would not replay across releases. ToolFilter runs at
// the same points too: in the first Tools call of each turn and in each
// ProcessRequest, never in the later Tools calls only ADK v2.6.0 makes.
func TestMcpListsAtTheSamePointInEveryStep(t *testing.T) {
	list, invoke, call := googleadk.ListMcpToolsActivityName, googleadk.InvokeModelActivityName, googleadk.CallMcpToolActivityName
	instr, filter := "Timer 1s", "Timer 2s"
	for _, tc := range []struct {
		filterSleeps bool
		want         []string
	}{
		{false, []string{
			// Turn 1, step 1: listed through Tools, before the instruction.
			list, instr, invoke, call,
			// Turn 1, step 2: listed in ProcessRequest, after the instruction.
			instr, list, invoke,
			// Turn 2: listed through Tools, before the instruction.
			list, instr, invoke,
		}},
		{true, []string{
			// Turn 1, step 1: Tools lists and filters, ProcessRequest filters.
			list, filter, instr, filter, invoke, call,
			// Turn 1, step 2: ProcessRequest lists and filters.
			instr, list, filter, invoke,
			// Turn 2: Tools lists and filters, ProcessRequest filters.
			list, filter, instr, filter, invoke,
		}},
	} {
		t.Run(fmt.Sprintf("filterSleeps=%v", tc.filterSleeps), func(t *testing.T) {
			m := &ruleModel{name: "fake-model", rule: func(req *model.LLMRequest) *model.LLMResponse {
				if lastUserText(req) == "search" && !lastIsFunctionResponse(req) {
					return googleadk.FunctionCallResponse("c1", "search", map[string]any{"query": "temporal"})
				}
				return googleadk.TextResponse("done")
			}}
			server := googleadk.NewFakeMCPServer("search-server").AddTool(
				"search", "search the web", searchSchema(),
				func(map[string]any) (map[string]any, error) { return map[string]any{"hits": 1}, nil },
			)

			var s testsuite.WorkflowTestSuite
			env, _ := newEnv(t, &s, googleadk.Config{
				Models:      map[string]googleadk.ModelFactory{"fake-model": m.factory()},
				MCPToolsets: map[string]googleadk.MCPFactory{"search-server": server.Factory()},
			})
			commands := watchCommands(env)
			env.RegisterWorkflow(sleepingInstructionWorkflow)
			env.ExecuteWorkflow(sleepingInstructionWorkflow, tc.filterSleeps)

			require.True(t, env.IsWorkflowCompleted())
			require.NoError(t, env.GetWorkflowError())

			assert.Equal(t, tc.want, commands.get())
			for i, names := range m.declared() {
				assert.Equal(t, []string{"search"}, names, "model request %d", i)
			}
		})
	}
}

// hidingToolset hides the "search" tool of the wrapped MCP server once hide is
// set, the way a server drops a tool from its list.
type hidingToolset struct {
	tool.Toolset
	hide *atomic.Bool
}

func (h hidingToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	tools, err := h.Toolset.Tools(ctx)
	if err != nil || !h.hide.Load() {
		return tools, err
	}
	var kept []tool.Tool
	for _, t := range tools {
		if t.Name() != "search" {
			kept = append(kept, t)
		}
	}
	return kept, nil
}

// searchDropped tells catalogChangeWorkflow's ToolFilter to drop "search".
var searchDropped atomic.Bool

// catalogChangeWorkflow runs one turn of an MCP agent and returns how many
// UUIDs ADK drew from the context. With filterDrops, the toolset's ToolFilter
// drops "search" once searchDropped is set.
func catalogChangeWorkflow(ctx workflow.Context, filterDrops bool) (int, error) {
	opts := googleadk.MCPToolsetOptions{Name: "search-server"}
	if filterDrops {
		opts.ToolFilter = func(_ agent.ReadonlyContext, t tool.Tool) bool {
			return t.Name() != "search" || !searchDropped.Load()
		}
	}
	root, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Model:       googleadk.NewModel("fake-model"),
		Instruction: "be helpful",
		Toolsets:    []tool.Toolset{googleadk.NewMCPToolset(opts)},
	})
	if err != nil {
		return 0, err
	}
	r, err := runner.New(runner.Config{AppName: "test-app", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		return 0, err
	}
	draws := 0
	adkCtx := platform.WithUUIDProvider(googleadk.NewContext(ctx), func() string {
		draws++
		return fmt.Sprintf("00000000-0000-0000-0000-%012d", draws)
	})
	for _, rerr := range r.Run(adkCtx, "user-1", "session-1", genai.NewContentFromText("search, then fetch", genai.RoleUser), agent.RunConfig{}) {
		if rerr != nil {
			return 0, rerr
		}
	}
	return draws, nil
}

// TestMcpDroppedToolIsNeitherDeclaredNorCalled proves each model request
// declares and dispatches the toolset's current list. ADK v2.5.0 and earlier
// keep the tools from the first Tools call of an agent run; if their proxies
// still packed themselves, a tool the server dropped would stay declared and
// a call to it would run CallMcpTool there, but not with ADK v2.6.0.
//
// It also proves a change in the list does not change the IDs ADK generates.
// ADK draws a UUID for each tool it kept from Tools before every model step:
// v2.5.0 and earlier keep the first list of the agent run, and v2.6.0 keeps
// what each step's Tools call returns. If those calls returned the current
// list, a dropped tool would shift the IDs drawn after it, tool confirmation
// IDs included, on v2.6.0 only, and Histories would not replay across releases.
func TestMcpDroppedToolIsNeitherDeclaredNorCalled(t *testing.T) {
	t.Run("not called", func(t *testing.T) {
		var hide atomic.Bool
		server := googleadk.NewFakeMCPServer("search-server").
			AddTool("search", "search the web", searchSchema(),
				func(map[string]any) (map[string]any, error) {
					hide.Store(true)
					return map[string]any{"hits": 1}, nil
				}).
			AddTool("fetch", "fetch a page", searchSchema(),
				func(map[string]any) (map[string]any, error) { return map[string]any{}, nil })
		m := &ruleModel{name: "fake-model", rule: func(req *model.LLMRequest) *model.LLMResponse {
			if n := functionResponses(req); n < 2 {
				return googleadk.FunctionCallResponse(fmt.Sprintf("c%d", n+1), "search", map[string]any{"query": "temporal"})
			}
			return googleadk.TextResponse("done")
		}}

		var s testsuite.WorkflowTestSuite
		env, counter := newEnv(t, &s, googleadk.Config{
			Models: map[string]googleadk.ModelFactory{"fake-model": m.factory()},
			MCPToolsets: map[string]googleadk.MCPFactory{"search-server": func(context.Context) (tool.Toolset, error) {
				return hidingToolset{Toolset: server, hide: &hide}, nil
			}},
		})
		env.ExecuteWorkflow(agentRunWorkflow, runInput{ModelName: "fake-model", UserMessage: "search twice", MCPToolset: "search-server"})

		require.True(t, env.IsWorkflowCompleted())
		require.NoError(t, env.GetWorkflowError())

		assert.Equal(t, [][]string{{"search", "fetch"}, {"fetch"}, {"fetch"}}, m.declared())
		assert.Equal(t, 3, counter.get(googleadk.ListMcpToolsActivityName))
		assert.Equal(t, 1, counter.get(googleadk.CallMcpToolActivityName), "the dropped tool ran again")
	})

	t.Run("same generated IDs", func(t *testing.T) {
		draws := map[string]int{}
		for _, tc := range []struct {
			name             string
			drop, filterDrop bool
			want             [][]string
		}{
			{"unchanged", false, false, [][]string{{"search", "fetch"}, {"search", "fetch"}, {"search", "fetch"}}},
			{"server drops search", true, false, [][]string{{"search", "fetch"}, {"fetch"}, {"fetch"}}},
			{"filter drops search", true, true, [][]string{{"search", "fetch"}, {"fetch"}, {"fetch"}}},
		} {
			searchDropped.Store(false)
			server := googleadk.NewFakeMCPServer("search-server").
				AddTool("search", "search the web", searchSchema(),
					func(map[string]any) (map[string]any, error) {
						searchDropped.Store(tc.drop)
						return map[string]any{"hits": 1}, nil
					}).
				AddTool("fetch", "fetch a page", searchSchema(),
					func(map[string]any) (map[string]any, error) { return map[string]any{}, nil })
			var ts tool.Toolset = server
			if !tc.filterDrop {
				ts = hidingToolset{Toolset: server, hide: &searchDropped}
			}
			m := &ruleModel{name: "fake-model", rule: func(req *model.LLMRequest) *model.LLMResponse {
				switch functionResponses(req) {
				case 0:
					return googleadk.FunctionCallResponse("c1", "search", map[string]any{"query": "temporal"})
				case 1:
					return googleadk.FunctionCallResponse("c2", "fetch", map[string]any{"query": "temporal"})
				}
				return googleadk.TextResponse("done")
			}}

			var s testsuite.WorkflowTestSuite
			env, counter := newEnv(t, &s, googleadk.Config{
				Models: map[string]googleadk.ModelFactory{"fake-model": m.factory()},
				MCPToolsets: map[string]googleadk.MCPFactory{"search-server": func(context.Context) (tool.Toolset, error) {
					return ts, nil
				}},
			})
			env.RegisterWorkflow(catalogChangeWorkflow)
			env.ExecuteWorkflow(catalogChangeWorkflow, tc.filterDrop)

			require.True(t, env.IsWorkflowCompleted(), tc.name)
			require.NoError(t, env.GetWorkflowError(), tc.name)
			var n int
			require.NoError(t, env.GetWorkflowResult(&n), tc.name)
			draws[tc.name] = n

			assert.Equal(t, tc.want, m.declared(), tc.name)
			assert.Equal(t, 2, counter.get(googleadk.CallMcpToolActivityName), tc.name)
		}
		searchDropped.Store(false)
		assert.Equal(t, draws["unchanged"], draws["server drops search"], "UUIDs drawn: %v", draws)
		assert.Equal(t, draws["unchanged"], draws["filter drops search"], "UUIDs drawn: %v", draws)
	})
}

// wrappedToolsLog records, for TestMcpWrappedToolsetListsOnEveryToolsCall, the
// tools each Tools call of the wrapped MCP toolset returns and the tools each
// model request declares, in order.
var wrappedToolsLog *commandLog

// loggingToolset records in wrappedToolsLog the tools each Tools call returns.
// Like the ADK wrappers, it has no ProcessRequest.
type loggingToolset struct {
	tool.Toolset
}

func (l loggingToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	tools, err := l.Toolset.Tools(ctx)
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		names = append(names, t.Name())
	}
	wrappedToolsLog.add(fmt.Sprintf("tools %v", names))
	return tools, err
}

// wrappedMCPWorkflow runs an agent whose MCP toolset is wrapped in the named
// ADK wrapper, which does not forward the toolset's ProcessRequest.
func wrappedMCPWorkflow(ctx workflow.Context, wrapper string) (runResult, error) {
	ts := googleadk.NewMCPToolset(googleadk.MCPToolsetOptions{Name: "search-server"})
	switch wrapper {
	case "FilterToolset":
		ts = tool.FilterToolset(ts, func(agent.ReadonlyContext, tool.Tool) bool { return true })
	case "WithConfirmation":
		ts = tool.WithConfirmation(ts, false, nil)
	default:
		return runResult{}, fmt.Errorf("unknown wrapper %q", wrapper)
	}
	return runAgent(ctx, agentBuild{
		modelName:   "fake-model",
		userMessage: "search twice",
		toolsets:    []tool.Toolset{loggingToolset{ts}},
	})
}

// TestMcpWrappedToolsetListsOnEveryToolsCall proves that an MCP toolset whose
// ProcessRequest ADK never calls, because a wrapper hides it, lists on every
// Tools call, as before the toolset listed once per model step: once per agent
// run with ADK v2.5.0 and earlier, and once per model step with v2.6.0. Each
// model request declares what the latest Tools call returned, so with v2.6.0 a
// tool the server dropped is neither declared nor called again.
func TestMcpWrappedToolsetListsOnEveryToolsCall(t *testing.T) {
	for _, wrapper := range []string{"FilterToolset", "WithConfirmation"} {
		t.Run(wrapper, func(t *testing.T) {
			wrappedToolsLog = &commandLog{}
			var hide atomic.Bool
			server := googleadk.NewFakeMCPServer("search-server").
				AddTool("search", "search the web", searchSchema(),
					func(map[string]any) (map[string]any, error) {
						hide.Store(true)
						return map[string]any{"hits": 1}, nil
					}).
				AddTool("fetch", "fetch a page", searchSchema(),
					func(map[string]any) (map[string]any, error) { return map[string]any{}, nil })
			m := &ruleModel{name: "fake-model", rule: func(req *model.LLMRequest) *model.LLMResponse {
				wrappedToolsLog.add(fmt.Sprintf("model %v", declaredNames(req)))
				if n := functionResponses(req); n < 2 {
					return googleadk.FunctionCallResponse(fmt.Sprintf("c%d", n+1), "search", map[string]any{"query": "temporal"})
				}
				return googleadk.TextResponse("done")
			}}

			var s testsuite.WorkflowTestSuite
			env, counter := newEnv(t, &s, googleadk.Config{
				Models: map[string]googleadk.ModelFactory{"fake-model": m.factory()},
				MCPToolsets: map[string]googleadk.MCPFactory{"search-server": func(context.Context) (tool.Toolset, error) {
					return hidingToolset{Toolset: server, hide: &hide}, nil
				}},
			})
			env.RegisterWorkflow(wrappedMCPWorkflow)
			env.ExecuteWorkflow(wrappedMCPWorkflow, wrapper)

			require.True(t, env.IsWorkflowCompleted())
			require.NoError(t, env.GetWorkflowError())

			toolsCalls, latest := 0, ""
			for _, e := range wrappedToolsLog.get() {
				if after, ok := strings.CutPrefix(e, "tools "); ok {
					toolsCalls++
					latest = after
					continue
				}
				assert.Equal(t, "model "+latest, e, "a model request declares the latest Tools list")
			}
			require.Positive(t, toolsCalls)
			assert.Equal(t, toolsCalls, counter.get(googleadk.ListMcpToolsActivityName), "one listing per Tools call")

			declared := m.declared()
			require.Len(t, declared, 3)
			wantCalls := 0
			for _, names := range declared[:2] {
				if slices.Contains(names, "search") {
					wantCalls++
				}
			}
			assert.Equal(t, wantCalls, counter.get(googleadk.CallMcpToolActivityName),
				"a search call runs CallMcpTool only when the request declared search")
			if toolsCalls == len(declared) {
				assert.Equal(t, [][]string{{"search", "fetch"}, {"fetch"}, {"fetch"}}, declared)
			}
		})
	}
}

// sharedProxyWorkflow runs one turn of an agent that uses an MCP toolset
// directly, then one turn of another agent, named secondName, that uses the
// same toolset through tool.FilterToolset. The runners share the bridged
// context.
func sharedProxyWorkflow(ctx workflow.Context, secondName string) error {
	ts := googleadk.NewMCPToolset(googleadk.MCPToolsetOptions{Name: "search-server"})
	adkCtx := googleadk.NewContext(ctx)
	for _, a := range []struct {
		name    string
		toolset tool.Toolset
	}{
		{"assistant", ts},
		{secondName, tool.FilterToolset(ts, func(agent.ReadonlyContext, tool.Tool) bool { return true })},
	} {
		ag, err := llmagent.New(llmagent.Config{
			Name:     a.name,
			Model:    googleadk.NewModel("fake-model"),
			Toolsets: []tool.Toolset{a.toolset},
		})
		if err != nil {
			return err
		}
		r, err := runner.New(runner.Config{AppName: "test-app", Agent: ag, SessionService: session.InMemoryService(), AutoCreateSession: true})
		if err != nil {
			return err
		}
		for _, rerr := range r.Run(adkCtx, "user-1", "session-1", genai.NewContentFromText("hello", genai.RoleUser), agent.RunConfig{}) {
			if rerr != nil {
				return rerr
			}
		}
	}
	return nil
}

// TestMcpSameNamedAgentsKeepTheirOwnListings proves that the toolset's
// ProcessRequest running for one agent does not stand in for another agent
// with the same name, whose wrapper hides that ProcessRequest: the second
// agent's request still declares the toolset's tools.
func TestMcpSameNamedAgentsKeepTheirOwnListings(t *testing.T) {
	for _, secondName := range []string{"assistant", "other"} {
		t.Run(secondName, func(t *testing.T) {
			m := &ruleModel{name: "fake-model", rule: func(*model.LLMRequest) *model.LLMResponse {
				return googleadk.TextResponse("done")
			}}
			server := googleadk.NewFakeMCPServer("search-server").AddTool("search", "search the web", searchSchema(),
				func(map[string]any) (map[string]any, error) { return map[string]any{"hits": 1}, nil })

			var s testsuite.WorkflowTestSuite
			env, counter := newEnv(t, &s, googleadk.Config{
				Models:      map[string]googleadk.ModelFactory{"fake-model": m.factory()},
				MCPToolsets: map[string]googleadk.MCPFactory{"search-server": server.Factory()},
			})
			env.RegisterWorkflow(sharedProxyWorkflow)
			env.ExecuteWorkflow(sharedProxyWorkflow, secondName)

			require.True(t, env.IsWorkflowCompleted())
			require.NoError(t, env.GetWorkflowError())

			assert.Equal(t, [][]string{{"search"}, {"search"}}, m.declared())
			assert.Equal(t, 2, counter.get(googleadk.ListMcpToolsActivityName), "one listing per agent run")
		})
	}
}

// searchFetchHelper returns a helper agent with two MCP toolsets.
func searchFetchHelper() (agent.Agent, error) {
	return llmagent.New(llmagent.Config{
		Name:        "helper",
		Description: "searches",
		Model:       googleadk.NewModel("helper-model"),
		Instruction: "search",
		Toolsets: []tool.Toolset{
			googleadk.NewMCPToolset(googleadk.MCPToolsetOptions{Name: "search-server"}),
			googleadk.NewMCPToolset(googleadk.MCPToolsetOptions{Name: "fetch-server"}),
		},
	})
}

// parallelHelperWorkflow runs one helper agent twice, in parallel unless
// sequential. Through an agent tool, the root agent calls it twice, and each
// call is its own invocation. With sharedInvocation, a custom root agent runs
// it on branches root.a and root.b through platform.RunTasks instead, so both
// runs share the root's invocation ID and the helper's name.
func parallelHelperWorkflow(ctx workflow.Context, sequential, sharedInvocation bool) (runResult, error) {
	helper, err := searchFetchHelper()
	if err != nil {
		return runResult{}, err
	}
	var opts []googleadk.ContextOption
	if sequential {
		opts = append(opts, googleadk.WithSequentialToolFanout())
	}
	if !sharedInvocation {
		return runAgent(ctx, agentBuild{
			ctxOpts:     opts,
			modelName:   "root-model",
			userMessage: "ask the helper twice",
			tools:       []tool.Tool{agenttool.New(helper, nil)},
		})
	}
	root, err := agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{helper},
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				stopped := false
				var tasks []func(context.Context)
				for _, branch := range []string{"root.a", "root.b"} {
					tasks = append(tasks, func(c context.Context) {
						bctx := agent.PromoteWithDelta(ic, &agent.CommonContextDelta{
							InvocationContextDelta: &agent.InvocationContextDelta{Context: &c, Branch: &branch},
						})
						for ev, err := range helper.Run(bctx) {
							if stopped || !yield(ev, err) {
								stopped = true
								return
							}
						}
					})
				}
				platform.RunTasks(ic, tasks)
			}
		},
	})
	if err != nil {
		return runResult{}, err
	}
	r, err := runner.New(runner.Config{AppName: "test-app", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		return runResult{}, err
	}
	for _, rerr := range r.Run(googleadk.NewContext(ctx, opts...), "user-1", "session-1", genai.NewContentFromText("ask the helper twice", genai.RoleUser), agent.RunConfig{}) {
		if rerr != nil {
			return runResult{}, rerr
		}
	}
	return runResult{}, nil
}

// TestMcpParallelInvocationsKeepTheirOwnListings proves concurrent runs of one
// agent each reuse the list their own Tools call took, whether they are
// separate invocations or share one invocation and agent name. Two helper runs
// of two model steps each, with two MCP toolsets, list 8 times, in parallel as
// in sequence.
func TestMcpParallelInvocationsKeepTheirOwnListings(t *testing.T) {
	for _, sharedInvocation := range []bool{false, true} {
		for _, sequential := range []bool{false, true} {
			t.Run(fmt.Sprintf("sharedInvocation=%v/sequential=%v", sharedInvocation, sequential), func(t *testing.T) {
				root := &ruleModel{name: "root-model", rule: func(req *model.LLMRequest) *model.LLMResponse {
					if functionResponses(req) == 0 {
						return reproFnCalls("helper", "helper")
					}
					return googleadk.TextResponse("root done")
				}}
				helper := &ruleModel{name: "helper-model", rule: func(req *model.LLMRequest) *model.LLMResponse {
					if lastIsFunctionResponse(req) {
						return googleadk.TextResponse("helper done")
					}
					return googleadk.FunctionCallResponse("h1", "search", map[string]any{"query": "temporal"})
				}}
				search := googleadk.NewFakeMCPServer("search-server").AddTool("search", "search the web", searchSchema(),
					func(map[string]any) (map[string]any, error) { return map[string]any{"hits": 1}, nil })
				fetch := googleadk.NewFakeMCPServer("fetch-server").AddTool("fetch", "fetch a page", searchSchema(),
					func(map[string]any) (map[string]any, error) { return map[string]any{}, nil })

				var s testsuite.WorkflowTestSuite
				env, counter := newEnv(t, &s, googleadk.Config{
					Models: map[string]googleadk.ModelFactory{"root-model": root.factory(), "helper-model": helper.factory()},
					MCPToolsets: map[string]googleadk.MCPFactory{
						"search-server": search.Factory(),
						"fetch-server":  fetch.Factory(),
					},
				})
				env.RegisterWorkflow(parallelHelperWorkflow)
				env.ExecuteWorkflow(parallelHelperWorkflow, sequential, sharedInvocation)

				require.True(t, env.IsWorkflowCompleted())
				require.NoError(t, env.GetWorkflowError())

				rootCalls := 2
				if sharedInvocation {
					rootCalls = 0
				}
				assert.Equal(t, 4+rootCalls, counter.get(googleadk.InvokeModelActivityName))
				assert.Equal(t, 8, counter.get(googleadk.ListMcpToolsActivityName), "one listing per toolset per helper model step")
				assert.Equal(t, 2, counter.get(googleadk.CallMcpToolActivityName))
				assert.Len(t, helper.declared(), 4)
				for i, names := range helper.declared() {
					assert.ElementsMatch(t, []string{"search", "fetch"}, names, "helper request %d", i)
				}
			})
		}
	}
}
