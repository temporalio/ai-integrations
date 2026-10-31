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

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"
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
// per MCP Activity, and returns the "lookup" tool response the model saw.
func mcpLookupWorkflow(ctx workflow.Context) (map[string]any, error) {
	root, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Model:       googleadk.NewModel("fake-model"),
		Instruction: "look up orders",
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
// isError result, reaches the model once with the text ADK gives it without
// Temporal, while a failed MCP request is still retried and an
// ApplicationError still fails the Activity.
func TestMcpToolErrorReachesModel(t *testing.T) {
	notFound := "^" + regexp.QuoteMeta("Tool execution failed. Details: order A-1 not found") + "$"
	for _, tc := range []struct {
		name      string
		factory   func(runs *atomic.Int32) googleadk.MCPFactory
		wantCalls int    // CallMcpTool attempts
		want      string // pattern for the "error" the model sees
	}{
		{"isError result", adkLookup(false), 1, notFound},
		{"failed request, then isError result", adkLookup(true), 2, notFound},
		{"FakeMCPServer handler error", fakeLookup(errors.New("order A-1 not found")), 1, "^order A-1 not found$"},
		{"ApplicationError", fakeLookup(temporal.NewNonRetryableApplicationError("order A-1 not found", "NotFound", nil)), 1,
			`^activity error \(type: googleadk\.CallMcpTool, .*\(type: NotFound, retryable: false\)$`},
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

			env.ExecuteWorkflow(mcpLookupWorkflow)

			require.True(t, env.IsWorkflowCompleted())
			require.NoError(t, env.GetWorkflowError())
			var got map[string]any
			require.NoError(t, env.GetWorkflowResult(&got))
			assert.Regexp(t, tc.want, got["error"])
			assert.EqualValues(t, 1, runs.Load(), "tool runs")
			assert.Equal(t, tc.wantCalls, counter.get(googleadk.CallMcpToolActivityName), "CallMcpTool attempts")
		})
	}
}
