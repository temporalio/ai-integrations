package googleadk_test

import (
	"context"
	"errors"
	"regexp"
	"sync/atomic"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/converter"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/mcptoolset"

	"go.temporal.io/googleadk"
)

// mcpLookupWorkflow runs an agent with the "orders" MCP toolset, two attempts
// per MCP Activity, and returns the "lookup" tool response the model saw. Its
// tool-error callback hands the model the error it got, with "callback": true.
func mcpLookupWorkflow(ctx workflow.Context) (map[string]any, error) {
	root, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Model:       googleadk.NewModel("fake-model"),
		Instruction: "look up orders",
		OnToolErrorCallbacks: []llmagent.OnToolErrorCallback{
			func(_ agent.Context, _ tool.Tool, _ map[string]any, err error) (map[string]any, error) {
				return map[string]any{"error": err.Error(), "callback": true}, nil
			},
		},
		Toolsets: []tool.Toolset{googleadk.NewMCPToolset(googleadk.MCPToolsetOptions{
			Name: "orders",
			ActivityOptions: workflow.ActivityOptions{
				RetryPolicy: &temporal.RetryPolicy{InitialInterval: time.Millisecond, MaximumAttempts: 2},
			},
		})},
	})
	if err != nil {
		return nil, err
	}
	r, err := runner.New(runner.Config{AppName: "test-app", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		return nil, err
	}
	var res runResult
	msg := genai.NewContentFromText("look up order A-1", genai.RoleUser)
	for ev, err := range r.Run(googleadk.NewContext(ctx), "user-1", "session-1", msg, agent.RunConfig{}) {
		if err != nil {
			return nil, err
		}
		collectEvent(&res, ev)
	}
	return res.ToolResponsePayloads["lookup"], nil
}

type lookupArgs struct {
	Order string `json:"order"`
}

// adkLookup returns a factory for ADK's mcptoolset over an in-memory go-sdk MCP
// server whose "lookup" tool always returns an isError result. With
// failFirst, the server answers the first tools/call request with a JSON-RPC
// error, like a failed request.
func adkLookup(failFirst bool) func(runs *atomic.Int32) googleadk.MCPFactory {
	return func(runs *atomic.Int32) googleadk.MCPFactory {
		return func(context.Context) (tool.Toolset, error) {
			srv := mcp.NewServer(&mcp.Implementation{Name: "orders", Version: "1"}, nil)
			mcp.AddTool(srv, &mcp.Tool{Name: "lookup", Description: "Look up an order."},
				func(_ context.Context, _ *mcp.CallToolRequest, in lookupArgs) (*mcp.CallToolResult, any, error) {
					runs.Add(1)
					return &mcp.CallToolResult{IsError: true, Content: []mcp.Content{&mcp.TextContent{Text: "order " + in.Order + " not found"}}}, nil, nil
				})
			var failed atomic.Bool
			srv.AddReceivingMiddleware(func(next mcp.MethodHandler) mcp.MethodHandler {
				return func(ctx context.Context, method string, req mcp.Request) (mcp.Result, error) {
					if failFirst && method == "tools/call" && failed.CompareAndSwap(false, true) {
						return nil, errors.New("server unavailable")
					}
					return next(ctx, method, req)
				}
			})
			clientTransport, serverTransport := mcp.NewInMemoryTransports()
			if _, err := srv.Connect(context.Background(), serverTransport, nil); err != nil {
				return nil, err
			}
			return mcptoolset.New(mcptoolset.Config{Transport: clientTransport})
		}
	}
}

// fakeLookup returns a factory for a FakeMCPServer whose "lookup" handler
// always returns err.
func fakeLookup(err error) func(runs *atomic.Int32) googleadk.MCPFactory {
	return func(runs *atomic.Int32) googleadk.MCPFactory {
		return googleadk.NewFakeMCPServer("orders").AddTool("lookup", "Look up an order.", nil,
			func(map[string]any) (map[string]any, error) {
				runs.Add(1)
				return nil, err
			}).Factory()
	}
}

// TestMcpToolErrorReachesModel proves an MCP tool's own error, such as an
// isError result, reaches the tool-error callback and the model once with the
// text ADK gives it without Temporal, while a failed MCP request, or an error
// that wraps several errors, is still retried and an ApplicationError still
// fails the Activity. It also decodes CallMcpTool's
// output as a workflow worker from before the fix does, reading only Result,
// and checks that it holds the same error.
func TestMcpToolErrorReachesModel(t *testing.T) {
	notFound := "^" + regexp.QuoteMeta("Tool execution failed. Details: order A-1 not found") + "$"
	for _, tc := range []struct {
		name      string
		factory   func(runs *atomic.Int32) googleadk.MCPFactory
		wantCalls int    // CallMcpTool attempts
		wantRuns  int32  // tool runs
		want      string // pattern for the "error" the callback gets and the model sees
		answered  bool   // CallMcpTool succeeds with the tool's error
	}{
		{"isError result", adkLookup(false), 1, 1, notFound, true},
		{"failed request, then isError result", adkLookup(true), 2, 1, notFound, true},
		{"FakeMCPServer handler error", fakeLookup(errors.New("order A-1 not found")), 1, 1, "^order A-1 not found$", true},
		{"FakeMCPServer handler joined errors", fakeLookup(errors.Join(errors.New("order A-1"), errors.New("not found"))), 2, 2,
			`(?s)^activity error \(type: googleadk\.CallMcpTool, .*\(type: googleadk\.McpError, retryable: true\)`, false},
		{"ApplicationError", fakeLookup(temporal.NewNonRetryableApplicationError("order A-1 not found", "NotFound", nil)), 1, 1,
			`^activity error \(type: googleadk\.CallMcpTool, .*\(type: NotFound, retryable: false\)$`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var runs atomic.Int32
			var s testsuite.WorkflowTestSuite
			env := s.NewTestWorkflowEnvironment()
			env.RegisterWorkflow(mcpLookupWorkflow)
			counter := wireActivities(t, env, googleadk.Config{
				Models: map[string]googleadk.ModelFactory{"fake-model": scriptedModelFactory(
					googleadk.FunctionCallResponse("c1", "lookup", map[string]any{"order": "A-1"}),
					googleadk.TextResponse("done"),
				)},
				MCPToolsets: map[string]googleadk.MCPFactory{"orders": tc.factory(&runs)},
			})
			// The fields a workflow worker from before the fix decodes.
			var oldOut *struct{ Result map[string]any }
			env.SetOnActivityCompletedListener(func(info *activity.Info, result converter.EncodedValue, err error) {
				if info.ActivityType.Name == googleadk.CallMcpToolActivityName && err == nil {
					oldOut = new(struct{ Result map[string]any })
					require.NoError(t, result.Get(oldOut))
				}
			})

			env.ExecuteWorkflow(mcpLookupWorkflow)

			require.True(t, env.IsWorkflowCompleted())
			require.NoError(t, env.GetWorkflowError())
			var got map[string]any
			require.NoError(t, env.GetWorkflowResult(&got))
			assert.Regexp(t, tc.want, got["error"])
			assert.Equal(t, true, got["callback"], "tool-error callback")
			assert.Equal(t, tc.wantRuns, runs.Load(), "tool runs")
			assert.Equal(t, tc.wantCalls, counter.get(googleadk.CallMcpToolActivityName), "CallMcpTool attempts")
			if tc.answered {
				require.NotNil(t, oldOut, "CallMcpTool output")
				assert.Equal(t, map[string]any{"error": got["error"]}, oldOut.Result, "Result read by an older workflow worker")
			} else {
				assert.Nil(t, oldOut, "CallMcpTool output")
			}
		})
	}
}

// TestMcpToolErrorAfterWorkerStops proves an error a tool returns after the
// Activity's context is cancelled, as when the Worker stops during the call,
// fails CallMcpTool with a retryable McpError, so the call runs again, instead
// of reaching the model as the tool's answer. The workflow test environment
// cannot show it: it has no Worker stop.
func TestMcpToolErrorAfterWorkerStops(t *testing.T) {
	bg, stop := context.WithCancel(context.Background())
	defer stop()
	lookup, err := funcTool("lookup", func(ctx agent.Context, _ map[string]any) (map[string]any, error) {
		stop() // the Worker stops during the call
		<-ctx.Done()
		return nil, ctx.Err()
	})
	require.NoError(t, err)
	acts, err := googleadk.NewActivities(googleadk.Config{MCPToolsets: map[string]googleadk.MCPFactory{
		"orders": func(context.Context) (tool.Toolset, error) {
			return staticToolset{name: "orders", tools: []tool.Tool{lookup}}, nil
		},
	}})
	require.NoError(t, err)
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()
	env.SetWorkerOptions(worker.Options{BackgroundActivityContext: bg})
	env.RegisterActivityWithOptions(acts.CallMcpTool, activity.RegisterOptions{Name: googleadk.CallMcpToolActivityName})

	_, err = env.ExecuteActivity(googleadk.CallMcpToolActivityName,
		map[string]any{"Toolset": "orders", "Tool": "lookup", "Args": map[string]any{}})

	var appErr *temporal.ApplicationError
	require.ErrorAs(t, err, &appErr, "CallMcpTool must fail, not return the tool's answer")
	assert.Equal(t, googleadk.ErrorTypeMCP, appErr.Type())
	assert.False(t, appErr.NonRetryable())
}
