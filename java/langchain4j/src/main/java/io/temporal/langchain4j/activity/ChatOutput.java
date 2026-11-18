package io.temporal.langchain4j.activity;

import javax.annotation.Nullable;

/**
 * Result of the {@code langchain4j.Chat} activity: the model's {@code AiMessage} and the
 * provider-independent {@code ChatResponseMetadata}.
 *
 * @param aiMessage the response message, as written by {@code ChatMessageSerializer}
 * @param id the response id
 * @param modelName the model that produced the response
 * @param inputTokenCount the input token count
 * @param outputTokenCount the output token count
 * @param totalTokenCount the total token count
 * @param finishReason the {@code FinishReason} name
 */
public record ChatOutput(
    String aiMessage,
    @Nullable String id,
    @Nullable String modelName,
    @Nullable Integer inputTokenCount,
    @Nullable Integer outputTokenCount,
    @Nullable Integer totalTokenCount,
    @Nullable String finishReason) {}
