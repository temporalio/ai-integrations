package googleadk

// White-box tests for the terminal behavior of Activities.Close: mcpToolset and
// mcpClosed are unexported. The externally-visible lifecycle behavior (caching,
// Close-closes-toolsets, idempotency) is covered in mcp_lifecycle_test.go.

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.temporal.io/sdk/temporal"
)

// TestMcpToolsetClosedIsTerminal proves Close is terminal: a late or
// Temporal-retried MCP Activity resolving a toolset after Close fails instead
// of silently repopulating the cache with a toolset nothing will ever close.
// The failure is a retryable McpError, so another worker can run the call.
func TestMcpToolsetClosedIsTerminal(t *testing.T) {
	server := NewFakeMCPServer("late-server").AddTool("noop", "noop tool", nil,
		func(map[string]any) (map[string]any, error) { return map[string]any{}, nil })
	acts, err := NewActivities(Config{
		MCPToolsets: map[string]MCPFactory{"late-server": server.Factory()},
	})
	require.NoError(t, err)

	// Sanity: resolvable (and cached) before Close.
	_, err = acts.mcpToolset(context.Background(), "late-server")
	require.NoError(t, err)

	require.NoError(t, acts.Close())

	_, err = acts.mcpToolset(context.Background(), "late-server")
	var appErr *temporal.ApplicationError
	require.ErrorAs(t, err, &appErr)
	assert.Equal(t, ErrorTypeMCP, appErr.Type())
	assert.False(t, appErr.NonRetryable(), "post-Close resolution must be retried: another worker can run the call")
	assert.Contains(t, err.Error(), "closed")
	assert.Empty(t, acts.mcpCache, "post-Close resolution must not repopulate the cache")
}
