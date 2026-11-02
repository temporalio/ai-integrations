package googleadk

// White-box tests for the MCP listing store: mcpListings and agentRunOf are
// unexported.

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"testing"
	"weak"

	"github.com/nexus-rpc/sdk-go/nexus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

type listingsTestRegistry struct {
	*testsuite.TestWorkflowEnvironment
}

func (listingsTestRegistry) RegisterNexusService(*nexus.Service) {}

// listingsAfterTurns is what listingsLifetimeWorkflow saw in the store after
// each turn and a garbage collection.
type listingsAfterTurns struct {
	Entries []int
	Live    []int
}

// listingsLifetimeWorkflow runs turns turns of an MCP agent on one NewContext,
// and records the store's entries, and those whose agent run is still
// reachable, after each turn and a garbage collection.
func listingsLifetimeWorkflow(ctx workflow.Context, turns int) (listingsAfterTurns, error) {
	var out listingsAfterTurns
	root, err := llmagent.New(llmagent.Config{
		Name:     "assistant",
		Model:    NewModel("fake-model"),
		Toolsets: []tool.Toolset{NewMCPToolset(MCPToolsetOptions{Name: "search-server"})},
	})
	if err != nil {
		return out, err
	}
	r, err := runner.New(runner.Config{AppName: "test-app", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		return out, err
	}
	adkCtx := NewContext(ctx)
	l := listingsFrom(adkCtx)
	for i := range turns {
		msg := genai.NewContentFromText(fmt.Sprintf("turn %d", i), genai.RoleUser)
		for _, rerr := range r.Run(adkCtx, "user-1", "session-1", msg, agent.RunConfig{}) {
			if rerr != nil {
				return out, rerr
			}
		}
		runtime.GC()
		l.mu.Lock()
		live := 0
		for k := range l.runs {
			if k.run.Value() != nil {
				live++
			}
		}
		out.Entries = append(out.Entries, len(l.runs))
		out.Live = append(out.Live, live)
		l.mu.Unlock()
	}
	return out, nil
}

// TestMcpListingsReleaseFinishedRuns proves the listing store does not keep
// finished turns: each turn on one NewContext is a new invocation, and its
// state, with the InvocationContext and the session events it holds, must go
// once the turn ends. Only the last turn's entry may remain, unreachable, until
// the next agent run starts and drops it.
func TestMcpListingsReleaseFinishedRuns(t *testing.T) {
	const turns = 20
	server := NewFakeMCPServer("search-server").AddTool("search", "search the web", nil,
		func(map[string]any) (map[string]any, error) { return map[string]any{"hits": 1}, nil })
	fm := NewFakeModel(
		FunctionCallResponse("c1", "search", map[string]any{"query": "temporal"}),
		TextResponse("done"),
	)
	acts, err := NewActivities(Config{
		Models:      map[string]ModelFactory{"fake-model": func(context.Context, string) (model.LLM, error) { return fm, nil }},
		MCPToolsets: map[string]MCPFactory{"search-server": server.Factory()},
	})
	require.NoError(t, err)

	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()
	acts.Register(listingsTestRegistry{env})
	env.RegisterWorkflow(listingsLifetimeWorkflow)
	env.ExecuteWorkflow(listingsLifetimeWorkflow, turns)
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var got listingsAfterTurns
	require.NoError(t, env.GetWorkflowResult(&got))
	require.Len(t, got.Entries, turns)
	for i := range turns {
		assert.LessOrEqual(t, got.Entries[i], 1, "entries after turn %d", i)
		assert.Zero(t, got.Live[i], "reachable agent runs after turn %d", i)
	}
}

// runProbe is a toolset that records what agentRunOf tells for the contexts
// ADK passes to its Tools and ProcessRequest.
type runProbe struct {
	mu    sync.Mutex
	calls []probeCall
}

type probeCall struct {
	method string
	run    weak.Pointer[byte]
	ok     bool
}

func (*runProbe) Name() string { return "probe" }

func (p *runProbe) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	p.record("Tools", ctx)
	return nil, nil
}

func (p *runProbe) ProcessRequest(ctx agent.Context, _ *model.LLMRequest) error {
	p.record("ProcessRequest", ctx)
	return nil
}

func (p *runProbe) record(method string, ctx context.Context) {
	run, ok := agentRunOf(ctx)
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls = append(p.calls, probeCall{method, run, ok})
}

// decoratorContext is a caller's context that keeps its parent in a field
// named context, like ADK's tool context does.
type decoratorContext struct {
	context.Context
	context context.Context
}

type contextHolder struct {
	context context.Context
}

// embeddingContext embeds a pointer to a struct with a context field.
type embeddingContext struct {
	context.Context
	*contextHolder
}

type probeKey struct{}

// TestMcpAgentRunOfReadsOnlyADKContexts proves agentRunOf tells the agent run
// from the contexts ADK passes to a toolset's Tools and ProcessRequest, and
// that it does not read other contexts. If an ADK release renamed a field it
// reads, the first case would fail here rather than as a listing count.
func TestMcpAgentRunOfReadsOnlyADKContexts(t *testing.T) {
	t.Run("ADK contexts", func(t *testing.T) {
		p := &runProbe{}
		root, err := llmagent.New(llmagent.Config{
			Name:     "assistant",
			Model:    NewFakeModel(TextResponse("done")),
			Toolsets: []tool.Toolset{p},
		})
		require.NoError(t, err)
		r, err := runner.New(runner.Config{AppName: "test-app", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true})
		require.NoError(t, err)
		for _, rerr := range r.Run(context.Background(), "user-1", "session-1", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
			require.NoError(t, rerr)
		}

		methods := map[string]bool{}
		for _, c := range p.calls {
			methods[c.method] = true
			assert.True(t, c.ok, "%s: agentRunOf did not tell the agent run", c.method)
			assert.True(t, c.run == p.calls[0].run, "%s: another agent run", c.method)
		}
		assert.Equal(t, map[string]bool{"Tools": true, "ProcessRequest": true}, methods)
	})

	t.Run("other contexts", func(t *testing.T) {
		parent := context.WithValue(context.Background(), probeKey{}, 1)
		for name, ctx := range map[string]context.Context{
			"decorator":           &decoratorContext{Context: parent, context: parent},
			"nil embedded struct": &embeddingContext{Context: parent},
		} {
			_, ok := agentRunOf(ctx)
			assert.False(t, ok, name)
		}
	})
}
