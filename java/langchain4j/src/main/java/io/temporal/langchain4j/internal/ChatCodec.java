package io.temporal.langchain4j.internal;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ChatMessageDeserializer;
import dev.langchain4j.data.message.ChatMessageSerializer;
import dev.langchain4j.internal.JsonSchemaElementJsonUtils;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.ChatResponseMetadata;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import io.temporal.langchain4j.activity.ChatInput;
import io.temporal.langchain4j.activity.ChatOutput;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Converts LangChain4j chat requests and responses to and from the activity's input and output.
 * Messages and tool specifications use LangChain4j's own JSON forms, so every content type they
 * support, such as images and tool execution requests, crosses the activity boundary unchanged.
 */
public final class ChatCodec {

  private ChatCodec() {}

  /**
   * Encodes a request for the activity.
   *
   * @throws IllegalArgumentException if the request carries provider-specific parameters, which
   *     only the provider's own model understands
   */
  public static ChatInput encode(String model, ChatRequest request) {
    return new ChatInput(
        model,
        ChatMessageSerializer.messagesToJson(request.messages()),
        encode(request.parameters()));
  }

  /** Decodes the activity input back into the request it was encoded from. */
  public static ChatRequest decode(ChatInput input) {
    List<ChatMessage> messages = ChatMessageDeserializer.messagesFromJson(input.messages());
    ChatRequest.Builder request = ChatRequest.builder().messages(messages);
    ChatInput.Parameters parameters = input.parameters();
    if (parameters != null) {
      request.parameters(decode(parameters));
    }
    return request.build();
  }

  /** Encodes a response as the activity result. */
  public static ChatOutput encode(ChatResponse response) {
    ChatResponseMetadata metadata = response.metadata();
    TokenUsage usage = metadata == null ? null : metadata.tokenUsage();
    FinishReason finishReason = metadata == null ? null : metadata.finishReason();
    return new ChatOutput(
        ChatMessageSerializer.messageToJson(response.aiMessage()),
        metadata == null ? null : metadata.id(),
        metadata == null ? null : metadata.modelName(),
        usage == null ? null : usage.inputTokenCount(),
        usage == null ? null : usage.outputTokenCount(),
        usage == null ? null : usage.totalTokenCount(),
        finishReason == null ? null : finishReason.name());
  }

  /** Decodes the activity result back into a response. */
  public static ChatResponse decode(ChatOutput output) {
    ChatResponse.Builder response =
        ChatResponse.builder()
            .aiMessage((AiMessage) ChatMessageDeserializer.messageFromJson(output.aiMessage()))
            .id(output.id())
            .modelName(output.modelName());
    if (output.inputTokenCount() != null
        || output.outputTokenCount() != null
        || output.totalTokenCount() != null) {
      response.tokenUsage(
          new TokenUsage(
              output.inputTokenCount(), output.outputTokenCount(), output.totalTokenCount()));
    }
    if (output.finishReason() != null) {
      response.finishReason(FinishReason.valueOf(output.finishReason()));
    }
    return response.build();
  }

  @Nullable
  private static ChatInput.Parameters encode(@Nullable ChatRequestParameters parameters) {
    if (parameters == null) {
      return null;
    }
    if (parameters.getClass() != DefaultChatRequestParameters.class) {
      throw new IllegalArgumentException(
          "Provider-specific chat request parameters ("
              + parameters.getClass().getName()
              + ") are not supported by TemporalChatModel: only the provider's own model"
              + " understands them, and it runs on the worker. Set them on the ChatModel"
              + " registered with LangChain4jPlugin instead.");
    }
    List<ToolSpecification> tools = parameters.toolSpecifications();
    ResponseFormat format = parameters.responseFormat();
    ToolChoice toolChoice = parameters.toolChoice();
    return new ChatInput.Parameters(
        parameters.modelName(),
        parameters.temperature(),
        parameters.topP(),
        parameters.topK(),
        parameters.frequencyPenalty(),
        parameters.presencePenalty(),
        parameters.maxOutputTokens(),
        nullIfEmpty(parameters.stopSequences()),
        tools == null || tools.isEmpty()
            ? null
            : tools.stream().map(ToolSpecification::toJson).toList(),
        toolChoice == null ? null : toolChoice.name(),
        format == null ? null : encode(format));
  }

  private static ChatInput.ResponseFormat encode(ResponseFormat format) {
    JsonSchema schema = format.jsonSchema();
    return new ChatInput.ResponseFormat(
        format.type().name(),
        schema == null ? null : schema.name(),
        schema == null || schema.rootElement() == null
            ? null
            : JsonSchemaElementJsonUtils.toMap(schema.rootElement()));
  }

  private static ChatRequestParameters decode(ChatInput.Parameters parameters) {
    DefaultChatRequestParameters.Builder<?> builder =
        DefaultChatRequestParameters.builder()
            .modelName(parameters.modelName())
            .temperature(parameters.temperature())
            .topP(parameters.topP())
            .topK(parameters.topK())
            .frequencyPenalty(parameters.frequencyPenalty())
            .presencePenalty(parameters.presencePenalty())
            .maxOutputTokens(parameters.maxOutputTokens())
            .stopSequences(parameters.stopSequences());
    if (parameters.toolSpecifications() != null) {
      builder.toolSpecifications(
          parameters.toolSpecifications().stream().map(ToolSpecification::fromJson).toList());
    }
    if (parameters.toolChoice() != null) {
      builder.toolChoice(ToolChoice.valueOf(parameters.toolChoice()));
    }
    ChatInput.ResponseFormat format = parameters.responseFormat();
    if (format != null) {
      ResponseFormat.Builder responseFormat =
          ResponseFormat.builder().type(ResponseFormatType.valueOf(format.type()));
      if (format.schemaName() != null || format.schema() != null) {
        responseFormat.jsonSchema(
            JsonSchema.builder()
                .name(format.schemaName())
                .rootElement(
                    format.schema() == null
                        ? null
                        : JsonSchemaElementJsonUtils.fromMap(format.schema()))
                .build());
      }
      builder.responseFormat(responseFormat.build());
    }
    return builder.build();
  }

  @Nullable
  private static <T> List<T> nullIfEmpty(@Nullable List<T> list) {
    return list == null || list.isEmpty() ? null : list;
  }
}
