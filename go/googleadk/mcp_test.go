package googleadk_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
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
	m.mu.Lock()
	m.decls = append(m.decls, names)
	m.mu.Unlock()
	resp := m.rule(req)
	return func(yield func(*model.LLMResponse, error) bool) { yield(resp, nil) }
}

func (m *ruleModel) factory() googleadk.ModelFactory {
	return func(context.Context, string) (model.LLM, error) { return m, nil }
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
// that does not depend on the ADK release: in the first model step of the
// first turn before the request processors (ADK calls Tools there on every
// release), and in every later step and turn after them, in ProcessRequest.
// ADK v2.6.0 also calls Tools before the request processors of every step; a
// listing there would put ListMcpTools before the instruction's timer, where
// ADK v2.5.0 and earlier put it after, and Histories would not replay across
// releases. ToolFilter runs at the same points too: in the first Tools call of
// each turn and in each ProcessRequest, never in the later Tools calls only
// ADK v2.6.0 makes.
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
			// Turn 2: listed in ProcessRequest.
			instr, list, invoke,
		}},
		{true, []string{
			// Turn 1, step 1: Tools lists and filters, ProcessRequest filters.
			list, filter, instr, filter, invoke, call,
			// Turn 1, step 2: ProcessRequest lists and filters.
			instr, list, filter, invoke,
			// Turn 2: Tools filters the kept list, ProcessRequest lists and
			// filters.
			filter, instr, list, filter, invoke,
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

// TestMcpDroppedToolIsNeitherDeclaredNorCalled proves each model request
// declares and dispatches the toolset's current list. ADK v2.5.0 and earlier
// keep the tools from the first Tools call of an agent run; if their proxies
// still packed themselves, a tool the server dropped would stay declared and
// a call to it would run CallMcpTool there, but not with ADK v2.6.0.
func TestMcpDroppedToolIsNeitherDeclaredNorCalled(t *testing.T) {
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
}

// parallelHelperWorkflow has the root agent call one helper agent twice in
// parallel through an agent tool. The helper has two MCP toolsets.
func parallelHelperWorkflow(ctx workflow.Context, sequential bool) (runResult, error) {
	helper, err := llmagent.New(llmagent.Config{
		Name:        "helper",
		Description: "searches",
		Model:       googleadk.NewModel("helper-model"),
		Instruction: "search",
		Toolsets: []tool.Toolset{
			googleadk.NewMCPToolset(googleadk.MCPToolsetOptions{Name: "search-server"}),
			googleadk.NewMCPToolset(googleadk.MCPToolsetOptions{Name: "fetch-server"}),
		},
	})
	if err != nil {
		return runResult{}, err
	}
	var opts []googleadk.ContextOption
	if sequential {
		opts = append(opts, googleadk.WithSequentialToolFanout())
	}
	return runAgent(ctx, agentBuild{
		ctxOpts:     opts,
		modelName:   "root-model",
		userMessage: "ask the helper twice",
		tools:       []tool.Tool{agenttool.New(helper, nil)},
	})
}

// TestMcpParallelInvocationsKeepTheirOwnListings proves concurrent invocations
// of one agent each reuse the list their own Tools call took. Two helper runs
// of two model steps each, with two MCP toolsets, list 8 times, in parallel as
// in sequence.
func TestMcpParallelInvocationsKeepTheirOwnListings(t *testing.T) {
	for _, sequential := range []bool{false, true} {
		t.Run(fmt.Sprintf("sequential=%v", sequential), func(t *testing.T) {
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
			env.ExecuteWorkflow(parallelHelperWorkflow, sequential)

			require.True(t, env.IsWorkflowCompleted())
			require.NoError(t, env.GetWorkflowError())

			assert.Equal(t, 6, counter.get(googleadk.InvokeModelActivityName))
			assert.Equal(t, 8, counter.get(googleadk.ListMcpToolsActivityName), "one listing per toolset per helper model step")
			assert.Equal(t, 2, counter.get(googleadk.CallMcpToolActivityName))
			for i, names := range helper.declared() {
				assert.ElementsMatch(t, []string{"search", "fetch"}, names, "helper request %d", i)
			}
		})
	}
}
