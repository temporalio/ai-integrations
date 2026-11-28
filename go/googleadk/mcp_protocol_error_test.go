package googleadk_test

import (
	"context"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"

	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/mcptoolset"

	"go.temporal.io/googleadk"
)

// failingMCP returns a factory for ADK's mcptoolset over an in-memory go-sdk MCP
// server that answers every request for method with err.
func failingMCP(method string, err error) googleadk.MCPFactory {
	return func(context.Context) (tool.Toolset, error) {
		srv := mcp.NewServer(&mcp.Implementation{Name: "orders", Version: "1"}, nil)
		mcp.AddTool(srv, &mcp.Tool{Name: "lookup", Description: "Look up an order."},
			func(context.Context, *mcp.CallToolRequest, lookupArgs) (*mcp.CallToolResult, any, error) {
				return &mcp.CallToolResult{}, nil, nil
			})
		srv.AddReceivingMiddleware(func(next mcp.MethodHandler) mcp.MethodHandler {
			return func(ctx context.Context, m string, req mcp.Request) (mcp.Result, error) {
				if m == method {
					return nil, err
				}
				return next(ctx, m, req)
			}
		})
		clientTransport, serverTransport := mcp.NewInMemoryTransports()
		if _, err := srv.Connect(context.Background(), serverTransport, nil); err != nil {
			return nil, err
		}
		return mcptoolset.New(mcptoolset.Config{Transport: clientTransport})
	}
}

// TestMcpProtocolErrorRetryability proves a JSON-RPC error that a retry of the
// same request cannot fix fails the MCP Activity as non-retryable, while other
// codes and a lost session stay retryable.
func TestMcpProtocolErrorRetryability(t *testing.T) {
	for _, tc := range []struct {
		name         string
		err          *jsonrpc.Error
		nonRetryable bool
	}{
		{"parse error", &jsonrpc.Error{Code: jsonrpc.CodeParseError, Message: "bad"}, true},
		{"invalid request", &jsonrpc.Error{Code: jsonrpc.CodeInvalidRequest, Message: "bad"}, true},
		{"method not found", &jsonrpc.Error{Code: jsonrpc.CodeMethodNotFound, Message: "bad"}, true},
		{"invalid params", &jsonrpc.Error{Code: jsonrpc.CodeInvalidParams, Message: "bad"}, true},
		{"header mismatch", &jsonrpc.Error{Code: mcp.CodeHeaderMismatch, Message: "bad"}, true},
		{"missing client capability", &jsonrpc.Error{Code: mcp.CodeMissingRequiredClientCapabilities, Message: "bad"}, true},
		{"unsupported protocol version", &jsonrpc.Error{Code: mcp.CodeUnsupportedProtocolVersion, Message: "bad"}, true},
		{"URL elicitation required", &jsonrpc.Error{Code: mcp.CodeURLElicitationRequired, Message: "bad"}, true},
		{"internal error", &jsonrpc.Error{Code: jsonrpc.CodeInternalError, Message: "bad"}, false},
		{"unknown code", &jsonrpc.Error{Code: -32099, Message: "bad"}, false},
		{"session not found", &jsonrpc.Error{Code: jsonrpc.CodeInvalidRequest, Message: "Session not found"}, false},
		{"session terminated", &jsonrpc.Error{Code: jsonrpc.CodeInvalidRequest, Message: "Session terminated"}, false},
	} {
		for _, req := range []struct{ activity, method string }{
			{googleadk.ListMcpToolsActivityName, "tools/list"},
			{googleadk.CallMcpToolActivityName, "tools/list"},
			{googleadk.CallMcpToolActivityName, "tools/call"},
		} {
			t.Run(tc.name+"/"+req.activity+"/"+req.method, func(t *testing.T) {
				acts, err := googleadk.NewActivities(googleadk.Config{
					MCPToolsets: map[string]googleadk.MCPFactory{"orders": failingMCP(req.method, tc.err)},
				})
				require.NoError(t, err)
				t.Cleanup(func() { _ = acts.Close() })
				var s testsuite.WorkflowTestSuite
				env := s.NewTestActivityEnvironment()
				env.RegisterActivityWithOptions(acts.ListMcpTools, activity.RegisterOptions{Name: googleadk.ListMcpToolsActivityName})
				env.RegisterActivityWithOptions(acts.CallMcpTool, activity.RegisterOptions{Name: googleadk.CallMcpToolActivityName})

				_, err = env.ExecuteActivity(req.activity, map[string]any{"Toolset": "orders", "Tool": "lookup"})

				var appErr *temporal.ApplicationError
				require.ErrorAs(t, err, &appErr)
				assert.Equal(t, googleadk.ErrorTypeMCP, appErr.Type())
				assert.Equal(t, tc.nonRetryable, appErr.NonRetryable(), "non-retryable")
			})
		}
	}
}
