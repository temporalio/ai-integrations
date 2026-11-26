package io.temporal.langchain4j.activity;

/**
 * Input of the {@code langchain4j.ListMcpTools} activity.
 *
 * @param client the name the MCP client is registered under on the worker
 */
public record ListMcpToolsInput(String client) {}
