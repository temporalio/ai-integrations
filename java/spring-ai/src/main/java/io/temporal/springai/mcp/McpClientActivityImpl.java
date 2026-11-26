package io.temporal.springai.mcp;

import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.spec.McpError;
import io.modelcontextprotocol.spec.McpSchema;
import io.temporal.failure.ApplicationFailure;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Implementation of {@link McpClientActivity} that delegates to Spring AI MCP clients.
 *
 * <p>This activity provides durable access to MCP servers. It is automatically registered by the
 * plugin when MCP clients are available in the Spring context.
 */
public class McpClientActivityImpl implements McpClientActivity {

  private final Map<String, McpSyncClient> mcpClients;

  /**
   * Creates an activity implementation with the given MCP clients.
   *
   * @param mcpClients list of MCP sync clients from Spring context
   */
  public McpClientActivityImpl(List<McpSyncClient> mcpClients) {
    this.mcpClients =
        mcpClients.stream()
            .collect(
                Collectors.toMap(
                    c -> c.getClientInfo().name(),
                    c -> c,
                    (existing, duplicate) -> {
                      throw new IllegalArgumentException(
                          "Duplicate MCP client name: '"
                              + existing.getClientInfo().name()
                              + "'. Each MCP client must have a unique name.");
                    }));
  }

  @Override
  public Map<String, McpSchema.ServerCapabilities> getServerCapabilities() {
    return mcpClients.entrySet().stream()
        .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue().getServerCapabilities()));
  }

  @Override
  public Map<String, McpSchema.Implementation> getClientInfo() {
    return mcpClients.entrySet().stream()
        .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue().getClientInfo()));
  }

  @Override
  public McpSchema.CallToolResult callTool(String clientName, McpSchema.CallToolRequest request) {
    McpSyncClient client = mcpClients.get(clientName);
    if (client == null) {
      throw ApplicationFailure.newBuilder()
          .setType("ClientNotFound")
          .setMessage(
              "MCP client '"
                  + clientName
                  + "' not found. Available clients: "
                  + mcpClients.keySet())
          .setNonRetryable(true)
          .build();
    }
    try {
      return client.callTool(request);
    } catch (RuntimeException e) {
      throw classify(clientName, e);
    }
  }

  @Override
  public Map<String, McpSchema.ListToolsResult> listTools() {
    return mcpClients.entrySet().stream()
        .collect(
            Collectors.toMap(
                Map.Entry::getKey,
                e -> {
                  try {
                    return e.getValue().listTools();
                  } catch (RuntimeException ex) {
                    throw classify(e.getKey(), ex);
                  }
                }));
  }

  /**
   * Fails the activity non-retryably when the MCP server rejected the request with a JSON-RPC
   * protocol error that retrying the same request cannot fix (parse error, invalid request, method
   * not found, invalid params). Any other error, including transport failures, is returned as is so
   * the activity retry policy still applies.
   */
  private static RuntimeException classify(String clientName, RuntimeException error) {
    Throwable cause = error;
    for (int depth = 0; cause != null && depth < 16; depth++) {
      if (cause instanceof McpError mcpError && isNonRetryable(mcpError)) {
        return ApplicationFailure.newNonRetryableFailureWithCause(
            "MCP client '" + clientName + "' rejected the request: " + error.getMessage(),
            "McpError",
            error);
      }
      cause = cause.getCause();
    }
    return error;
  }

  private static boolean isNonRetryable(McpError error) {
    if (error.getJsonRpcError() == null || error.getJsonRpcError().code() == null) {
      return false;
    }
    int code = error.getJsonRpcError().code();
    return code == McpSchema.ErrorCodes.PARSE_ERROR
        || code == McpSchema.ErrorCodes.INVALID_REQUEST
        || code == McpSchema.ErrorCodes.METHOD_NOT_FOUND
        || code == McpSchema.ErrorCodes.INVALID_PARAMS;
  }
}
