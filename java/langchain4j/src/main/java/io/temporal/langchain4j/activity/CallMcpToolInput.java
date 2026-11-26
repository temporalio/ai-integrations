package io.temporal.langchain4j.activity;

import javax.annotation.Nullable;

/**
 * Input of the {@code langchain4j.CallMcpTool} activity: a LangChain4j {@code ToolExecutionRequest}
 * for the MCP client registered under {@code client}.
 *
 * @param client the name the MCP client is registered under on the worker
 * @param id the tool execution request id, or {@code null} when the model gave none
 * @param tool the name of the tool on the MCP server
 * @param arguments the tool arguments as a JSON object
 */
public record CallMcpToolInput(
    String client, @Nullable String id, String tool, @Nullable String arguments) {}
