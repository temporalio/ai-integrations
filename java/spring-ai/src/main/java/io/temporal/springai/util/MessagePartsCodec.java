package io.temporal.springai.util;

import io.temporal.failure.ApplicationFailure;
import io.temporal.springai.model.ChatModelTypes;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.springframework.ai.chat.messages.AbstractMessage;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.messages.part.MediaPart;
import org.springframework.ai.chat.messages.part.MessagePart;
import org.springframework.ai.chat.messages.part.TextPart;
import org.springframework.ai.chat.messages.part.ToolResultPart;
import org.springframework.ai.util.JacksonUtils;
import tools.jackson.databind.JavaType;
import tools.jackson.databind.ObjectReader;
import tools.jackson.databind.ObjectWriter;
import tools.jackson.databind.SerializationFeature;
import tools.jackson.databind.json.JsonMapper;

/** Transports ordered content through the public Spring AI 2.1 message and Jackson APIs. */
public final class MessagePartsCodec {
  private static final JsonMapper MAPPER =
      JacksonUtils.getDefaultJsonMapper()
          .rebuild()
          .enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS)
          .build();
  private static final JavaType PARTS_TYPE =
      MAPPER.getTypeFactory().constructCollectionType(List.class, MessagePart.class);
  private static final ObjectReader PARTS_READER = MAPPER.readerFor(PARTS_TYPE);
  private static final ObjectWriter PARTS_WRITER = MAPPER.writerFor(PARTS_TYPE);

  private MessagePartsCodec() {}

  public static ChatModelTypes.Message encode(Message message) {
    MessageType role = message.getMessageType();
    List<MessagePart> parts = partsForEncoding(message, role);
    validateMedia(parts);
    Map<String, Object> metadata = new LinkedHashMap<>(message.getMetadata());
    // Spring AI reconstructs this entry from the message role when decoding.
    metadata.remove(AbstractMessage.MESSAGE_TYPE);
    if (role != MessageType.ASSISTANT
        && !isJsonMetadata(metadata, Collections.newSetFromMap(new IdentityHashMap<>()))) {
      throw ApplicationFailure.newNonRetryableFailure(
          "Spring AI user, system, and tool message metadata requires string keys and values of"
              + " type String, Boolean, Integer, or finite Double, or lists and maps of these"
              + " values. Null values are supported. Convert other values to one of these types"
              + " and remove any cyclic references before calling the chat model.",
          "UnsupportedSpringAiMessageMetadata");
    }
    return new ChatModelTypes.Message(role, PARTS_WRITER.writeValueAsString(parts), metadata);
  }

  private static List<MessagePart> partsForEncoding(Message message, MessageType role) {
    if (message instanceof AbstractMessage springMessage) {
      if (role != MessageType.SYSTEM) {
        return springMessage.getParts();
      }
      validateSystemParts(springMessage.getParts());
    }
    // SystemMessage takes one string. Use the same text the provider reads from the prompt.
    String text = message.getText();
    if (role == MessageType.SYSTEM && text == null) {
      throw ApplicationFailure.newNonRetryableFailure(
          "Spring AI system messages require text content.", "UnsupportedSpringAiSystemMessage");
    }
    return List.of(TextPart.of(text));
  }

  private static void validateSystemParts(List<MessagePart> parts) {
    if (parts.isEmpty()
        || parts.stream()
            .anyMatch(
                part ->
                    !(part instanceof TextPart text)
                        || text.payload() != null
                        || !text.attributes().isEmpty())) {
      throw ApplicationFailure.newNonRetryableFailure(
          "Spring AI system messages require plain text parts without payloads or attributes.",
          "UnsupportedSpringAiSystemMessage");
    }
  }

  private static void validateMedia(List<MessagePart> parts) {
    for (MessagePart part : parts) {
      if (part instanceof MediaPart mediaPart) {
        Object data = mediaPart.media().getData();
        if (data instanceof byte[] bytes) {
          ChatModelTypes.checkMediaSize(bytes);
        } else if (!(data instanceof String)) {
          throw ApplicationFailure.newNonRetryableFailure(
              "Spring AI media data requires a String or byte array. Received "
                  + (data == null ? "null" : data.getClass().getTypeName())
                  + ". Pass the media URI as a string or supply its bytes.",
              "UnsupportedSpringAiMedia");
        }
      } else if (part instanceof ToolResultPart toolResult) {
        validateMedia(toolResult.content());
      }
    }
  }

  private static boolean isJsonMetadata(Object value, Set<Object> ancestors) {
    if (value instanceof List<?> list) {
      if (!ancestors.add(list)) {
        return false;
      }
      boolean supported = list.stream().allMatch(item -> isJsonMetadata(item, ancestors));
      ancestors.remove(list);
      return supported;
    }
    if (value instanceof Map<?, ?> map) {
      if (!ancestors.add(map)) {
        return false;
      }
      boolean supported =
          map.entrySet().stream()
              .allMatch(
                  entry ->
                      entry.getKey() instanceof String
                          && isJsonMetadata(entry.getValue(), ancestors));
      ancestors.remove(map);
      return supported;
    }
    return value == null
        || value instanceof String
        || value instanceof Boolean
        || value instanceof Integer
        || (value instanceof Double number && Double.isFinite(number));
  }

  public static Message decode(ChatModelTypes.Message message) {
    List<MessagePart> parts = PARTS_READER.readValue(message.partsJson());
    Map<String, Object> metadata = message.metadata();
    return switch (message.role()) {
      case ASSISTANT -> AssistantMessage.builder().parts(parts).properties(metadata).build();
      case USER -> UserMessage.builder().parts(parts).metadata(metadata).build();
      case TOOL -> ToolResponseMessage.builder().parts(parts).metadata(metadata).build();
      case SYSTEM -> {
        validateSystemParts(parts);
        String text =
            parts.stream()
                .map(TextPart.class::cast)
                .map(TextPart::text)
                .collect(Collectors.joining());
        yield SystemMessage.builder().text(text).metadata(metadata).build();
      }
    };
  }
}
