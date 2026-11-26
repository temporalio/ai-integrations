package io.temporal.langchain4j.activity;

import javax.annotation.Nullable;

/**
 * Input of the {@code langchain4j.StreamingChat} activity.
 *
 * @param request the registered model name and the chat request
 * @param streamingTopic the Workflow Streams topic that partial output is published to while the
 *     activity runs, or {@code null} to only return it as the result
 */
public record StreamingChatInput(ChatInput request, @Nullable String streamingTopic) {}
