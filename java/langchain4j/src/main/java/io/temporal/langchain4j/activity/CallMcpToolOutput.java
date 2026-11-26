package io.temporal.langchain4j.activity;

import javax.annotation.Nullable;

/**
 * Result of the {@code langchain4j.CallMcpTool} activity: a LangChain4j {@code
 * ToolExecutionResult}, or the error text an MCP server wrote for the model.
 *
 * @param contents the result contents, as the contents of a {@code ToolExecutionResultMessage}
 *     written by {@code ChatMessageSerializer}, or {@code null} for an error for the model
 * @param isError whether the server reported the result as an error that the client returned
 *     instead of throwing
 * @param result the structured result as JSON, or {@code null} when the tool returned none
 * @param attributes the result attributes as a JSON object, or {@code null} when there are none
 * @param errorForLlm the text of a failure that LangChain4j shows the model ({@code
 *     ToolErrorVisibleToLlm}, such as an MCP application error), or {@code null}
 */
public record CallMcpToolOutput(
    @Nullable String contents,
    boolean isError,
    @Nullable String result,
    @Nullable String attributes,
    @Nullable String errorForLlm) {}
