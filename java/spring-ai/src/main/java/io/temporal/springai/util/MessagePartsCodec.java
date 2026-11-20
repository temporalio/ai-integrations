package io.temporal.springai.util;

import io.temporal.failure.ApplicationFailure;
import io.temporal.springai.model.ChatModelTypes;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import org.springframework.ai.chat.messages.AbstractMessage;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.messages.part.MessagePart;
import org.springframework.ai.chat.messages.part.TextPart;
import org.springframework.ai.util.JacksonUtils;
import tools.jackson.databind.JavaType;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectReader;
import tools.jackson.databind.ObjectWriter;
import tools.jackson.databind.SerializationFeature;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ObjectNode;

/** Transports ordered content through the public Spring AI 2.1 message and Jackson APIs. */
public final class MessagePartsCodec {
  private static final Pattern GENERATED_MEDIA_NAME =
      Pattern.compile("(media-.+-)[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}");
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
    List<MessagePart> parts =
        message instanceof AbstractMessage springMessage
            ? springMessage.getParts()
            : List.of(TextPart.of(message.getText()));
    JsonNode tree = MAPPER.readTree(PARTS_WRITER.writeValueAsString(parts));
    prepareMedia(tree, "");
    return new ChatModelTypes.Message(
        message.getMessageType(), MAPPER.writeValueAsString(tree), message.getMetadata());
  }

  private static void prepareMedia(JsonNode parts, String parent) {
    for (int index = 0; index < parts.size(); index++) {
      JsonNode part = parts.get(index);
      String position = parent + index;
      switch (part.path("type").asString()) {
        case "media" -> {
          ObjectNode media = (ObjectNode) part.get("media");
          String kind = media.path("dataKind").asString();
          if ("url".equals(kind)) {
            // Spring AI's media deserializer turns URL objects into strings.
            throw ApplicationFailure.newNonRetryableFailure(
                "Spring AI ordered media parts require a URI string or byte array."
                    + " Pass the URI as a string instead of a java.net.URL object.",
                "UnsupportedSpringAiMedia");
          }
          if ("bytes".equals(kind)) {
            String data = media.get("data").asString();
            int padding = data.endsWith("==") ? 2 : data.endsWith("=") ? 1 : 0;
            ChatModelTypes.checkMediaSize(data.length() / 4 * 3 - padding);
          }
          // Spring AI generates these names with a random UUID. Keep caller-provided names.
          var generatedName = GENERATED_MEDIA_NAME.matcher(media.get("name").asString());
          if (generatedName.matches()) {
            media.put("name", generatedName.group(1) + position);
          }
        }
        case "tool_result" -> prepareMedia(part.get("content"), position + "-");
        default -> {
          // Other parts have no inline media.
        }
      }
    }
  }

  public static Message decode(ChatModelTypes.Message message) {
    List<MessagePart> parts = PARTS_READER.readValue(message.partsJson());
    Map<String, Object> metadata = message.metadata();
    return switch (message.role()) {
      case ASSISTANT -> AssistantMessage.builder().parts(parts).properties(metadata).build();
      case USER -> UserMessage.builder().parts(parts).metadata(metadata).build();
      case TOOL -> ToolResponseMessage.builder().parts(parts).metadata(metadata).build();
      case SYSTEM -> {
        if (parts.size() != 1 || !(parts.get(0) instanceof TextPart text)) {
          throw new IllegalArgumentException("System messages require a single text part");
        }
        yield SystemMessage.builder().text(text.text()).metadata(metadata).build();
      }
    };
  }
}
