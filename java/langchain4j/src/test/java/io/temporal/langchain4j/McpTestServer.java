package io.temporal.langchain4j;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.modelcontextprotocol.json.jackson2.JacksonMcpJsonMapper;
import io.modelcontextprotocol.server.McpServer;
import io.modelcontextprotocol.server.transport.StdioServerTransportProvider;
import io.modelcontextprotocol.spec.McpError;
import io.modelcontextprotocol.spec.McpSchema;
import java.util.List;
import java.util.Map;

/**
 * A local MCP server, built on the official MCP Java SDK, that {@link McpWorkflowTest} launches
 * over stdio; requires only Java.
 */
public final class McpTestServer {

  /** Text in server errors that must never reach the workflow history or the model. */
  static final String SECRET = "secret-token-1234";

  private McpTestServer() {}

  public static void main(String[] args) {
    // Reserve stdout for the MCP protocol; send any library logging to stderr.
    var protocolOutput = System.out;
    System.setOut(System.err);
    var transport =
        new StdioServerTransportProvider(
            new JacksonMcpJsonMapper(new ObjectMapper()), System.in, protocolOutput);
    McpServer.sync(transport)
        .serverInfo("langchain4j-test", "1.0")
        .capabilities(McpSchema.ServerCapabilities.builder().tools(false).build())
        .toolCall(
            tool("echo", "Echoes the text"),
            (exchange, request) ->
                McpSchema.CallToolResult.builder()
                    .addTextContent("echo: " + request.arguments().get("text"))
                    .isError(false)
                    .build())
        .toolCall(
            tool("lookup", "Looks up a record"),
            (exchange, request) ->
                McpSchema.CallToolResult.builder()
                    .addTextContent("No record matches " + request.arguments().get("text") + ".")
                    .isError(true)
                    .build())
        .toolCall(
            tool("strict", "Rejects every argument"),
            (exchange, request) -> {
              throw McpError.builder(McpSchema.ErrorCodes.INVALID_PARAMS)
                  .message("invalid text " + SECRET)
                  .build();
            })
        .toolCall(
            tool("broken", "Always fails"),
            (exchange, request) -> {
              throw McpError.builder(McpSchema.ErrorCodes.INTERNAL_ERROR)
                  .message("database password " + SECRET)
                  .build();
            })
        .build();
  }

  private static McpSchema.Tool tool(String name, String description) {
    return McpSchema.Tool.builder(
            name,
            Map.of(
                "type", "object",
                "properties", Map.of("text", Map.of("type", "string")),
                "required", List.of("text")))
        .description(description)
        .build();
  }
}
