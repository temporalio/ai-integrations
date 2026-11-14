package io.temporal.springai;

import io.modelcontextprotocol.json.jackson3.JacksonMcpJsonMapper;
import io.modelcontextprotocol.server.McpServer;
import io.modelcontextprotocol.server.transport.StdioServerTransportProvider;
import io.modelcontextprotocol.spec.McpSchema;
import java.util.List;
import java.util.Map;
import tools.jackson.databind.json.JsonMapper;

/** Local MCP server launched by {@link SpringBootAutoConfigurationTest}; requires only Java. */
public class StdioMcpTestServer {

  public static void main(String[] args) {
    // Reserve stdout for the MCP protocol; send any library logging to stderr.
    var protocolOutput = System.out;
    System.setOut(System.err);
    var transport =
        new StdioServerTransportProvider(
            new JacksonMcpJsonMapper(JsonMapper.builder().build()), System.in, protocolOutput);
    McpSchema.Tool tool =
        McpSchema.Tool.builder(
                "read_file",
                Map.of(
                    "type", "object",
                    "properties", Map.of("path", Map.of("type", "string")),
                    "required", List.of("path")))
            .description("Read a file")
            .build();
    McpServer.sync(transport)
        .serverInfo("test-filesystem", "1.0")
        .capabilities(McpSchema.ServerCapabilities.builder().tools(false).build())
        .toolCall(
            tool,
            (exchange, request) ->
                McpSchema.CallToolResult.builder()
                    .addTextContent("hello from " + request.arguments().get("path"))
                    .isError(false)
                    .build())
        .build();
  }
}
