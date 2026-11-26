package io.temporal.langchain4j.activity;

import javax.annotation.Nullable;

/**
 * One partial output of a streaming chat model, in the order the model produced it.
 *
 * @param type what the part carries
 * @param text the partial response or thinking text, the partial arguments of a partial tool call,
 *     or the complete arguments of a complete tool call
 * @param index the tool call index, or {@code null} for text parts
 * @param id the tool call id, or {@code null} for text parts and when the model gave none
 * @param name the tool name, or {@code null} for text parts
 */
public record ChatStreamPart(
    Type type,
    @Nullable String text,
    @Nullable Integer index,
    @Nullable String id,
    @Nullable String name) {

  /** The kind of partial output. */
  public enum Type {
    /** A {@code PartialResponse}. */
    PARTIAL_RESPONSE,
    /** A {@code PartialThinking}. */
    PARTIAL_THINKING,
    /** A {@code PartialToolCall}. */
    PARTIAL_TOOL_CALL,
    /** A {@code CompleteToolCall}. */
    COMPLETE_TOOL_CALL
  }
}
