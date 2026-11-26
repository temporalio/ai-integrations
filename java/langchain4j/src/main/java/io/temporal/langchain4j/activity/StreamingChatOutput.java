package io.temporal.langchain4j.activity;

import java.util.List;

/**
 * Result of the {@code langchain4j.StreamingChat} activity: the complete response and the partial
 * output that preceded it, which the workflow passes to the response handler in order.
 *
 * @param response the complete response
 * @param parts the partial output, in the order the model produced it
 */
public record StreamingChatOutput(ChatOutput response, List<ChatStreamPart> parts) {}
