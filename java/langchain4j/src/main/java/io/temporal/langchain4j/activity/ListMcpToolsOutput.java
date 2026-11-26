package io.temporal.langchain4j.activity;

import java.util.List;

/**
 * Result of the {@code langchain4j.ListMcpTools} activity.
 *
 * @param tools the server's tool specifications, each as written by {@code
 *     ToolSpecification.toJson()}
 */
public record ListMcpToolsOutput(List<String> tools) {}
