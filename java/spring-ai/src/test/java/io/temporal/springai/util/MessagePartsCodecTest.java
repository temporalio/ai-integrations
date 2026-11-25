package io.temporal.springai.util;

import static org.junit.jupiter.api.Assertions.*;

import io.temporal.common.converter.DefaultDataConverter;
import io.temporal.failure.ApplicationFailure;
import io.temporal.springai.model.ChatModelTypes;
import java.net.URI;
import java.nio.ByteBuffer;
import java.time.LocalDate;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AbstractMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.messages.part.MediaPart;
import org.springframework.ai.chat.messages.part.MessagePart;
import org.springframework.ai.chat.messages.part.ReasoningPart;
import org.springframework.ai.chat.messages.part.TextPart;
import org.springframework.ai.chat.messages.part.ToolResultPart;
import org.springframework.ai.content.Media;
import org.springframework.util.MimeTypeUtils;

class MessagePartsCodecTest {
  @Test
  void preservesJoinedSystemTextThroughTheTemporalConverter() {
    var message = new SystemPartsMessage(List.of(TextPart.of("First.\n"), TextPart.of("Second.")));

    var decoded = assertInstanceOf(SystemMessage.class, roundTrip(message));

    assertEquals("First.\nSecond.", decoded.getText());
    assertEquals(message.getMetadata(), decoded.getMetadata());
  }

  @Test
  void decodesSystemTextFromPayloadsWithSeparateParts() {
    List<MessagePart> parts = List.of(TextPart.of("First.\n"), TextPart.of("Second."));
    var orderedPartsJson =
        MessagePartsCodec.encode(UserMessage.builder().parts(parts).build()).partsJson();
    var payload = new ChatModelTypes.Message(MessageType.SYSTEM, orderedPartsJson, Map.of());

    assertEquals("First.\nSecond.", MessagePartsCodec.decode(payload).getText());
  }

  @Test
  void rejectsUnsupportedSystemPartsAtEncoding() {
    var unsupportedParts =
        List.of(
            new ReasoningPart("thinking", null, null, Map.of()),
            new TextPart("decorated", null, Map.of("provider", "value")));
    for (var part : unsupportedParts) {
      var message = new SystemPartsMessage(List.of(TextPart.of("instruction"), part));

      var failure = assertThrows(ApplicationFailure.class, () -> MessagePartsCodec.encode(message));
      assertTrue(failure.isNonRetryable());
      assertEquals("UnsupportedSpringAiSystemMessage", failure.getType());
    }
  }

  @Test
  void nestedToolMediaAndUriRoundTripThroughTheTemporalConverter() {
    byte[] bytes = {0, 1, -1};
    String uri = "https://example.test/image.png";
    String label = "media-png-00000000-0000-4000-8000-000000000001";
    var inlineImage = Media.builder().mimeType(MimeTypeUtils.IMAGE_PNG).data(bytes).build();
    var linkedImage =
        Media.builder().mimeType(MimeTypeUtils.IMAGE_PNG).name(label).data(uri).build();
    var tool =
        ToolResponseMessage.builder()
            .result(
                new ToolResultPart(
                    "call",
                    "image",
                    List.of(
                        MediaPart.of(inlineImage),
                        TextPart.of("image result"),
                        MediaPart.of(linkedImage)),
                    false,
                    null,
                    Map.of()))
            .build();

    var decoded = assertInstanceOf(ToolResponseMessage.class, roundTrip(tool));

    assertEquals(MessagePartsCodec.encode(tool), MessagePartsCodec.encode(decoded));
    var content = decoded.getResults().get(0).content();
    var restoredInline = assertInstanceOf(MediaPart.class, content.get(0)).media();
    assertArrayEquals(bytes, restoredInline.getDataAsByteArray());
    assertEquals(inlineImage.getName(), restoredInline.getName());
    assertEquals(TextPart.of("image result"), content.get(1));
    var restoredLink = assertInstanceOf(MediaPart.class, content.get(2)).media();
    assertEquals(label, restoredLink.getName());
    assertEquals(uri, restoredLink.getData());
  }

  @Test
  void rejectsOversizedNestedMediaBeforeSpringAiSerializesIt() {
    // The base64 form exceeds Jackson's string limit if the bytes are serialized first.
    var media =
        Media.builder()
            .mimeType(MimeTypeUtils.IMAGE_PNG)
            .name("oversized image")
            .data(new byte[80 * 1024 * 1024])
            .build();
    var message =
        ToolResponseMessage.builder()
            .result(
                new ToolResultPart(
                    "call", "image", List.of(MediaPart.of(media)), false, null, Map.of()))
            .build();

    var failure = assertThrows(ApplicationFailure.class, () -> MessagePartsCodec.encode(message));
    assertTrue(failure.isNonRetryable());
    assertEquals(ChatModelTypes.MEDIA_SIZE_EXCEEDED_FAILURE_TYPE, failure.getType());
  }

  @Test
  void preservesJsonMetadataThroughTheTemporalConverter() {
    var values = Arrays.asList("text", true, 5, 1.25, null);
    var metadata = new HashMap<String, Object>();
    metadata.put("nested", Map.of("values", values));
    metadata.put("empty", null);
    for (var message :
        List.of(
            UserMessage.builder().text("hi").metadata(metadata).build(),
            SystemMessage.builder().text("hi").metadata(metadata).build(),
            ToolResponseMessage.builder().metadata(metadata).build())) {
      assertEquals(message.getMetadata(), roundTrip(message).getMetadata());
    }
  }

  @Test
  void rejectsMetadataValuesThatWouldChangeTypeThroughTheTemporalConverter() throws Exception {
    var user = UserMessage.builder().text("hi").metadata(Map.of("bytes", new byte[] {42})).build();
    var system =
        SystemMessage.builder()
            .text("hi")
            .metadata(Map.of("date", LocalDate.of(2026, 11, 25)))
            .build();
    var tool =
        ToolResponseMessage.builder()
            .metadata(
                Map.of(
                    "nested", List.of(Map.of("url", URI.create("https://example.test").toURL()))))
            .build();
    for (var message : List.of(user, system, tool)) {
      var failure = assertThrows(ApplicationFailure.class, () -> MessagePartsCodec.encode(message));
      assertTrue(failure.isNonRetryable());
      assertEquals("UnsupportedSpringAiMessageMetadata", failure.getType());
    }
  }

  @Test
  void rejectsMediaWhoseUrlTypeSpringAiCannotRoundTrip() throws Exception {
    var media =
        Media.builder()
            .mimeType(MimeTypeUtils.IMAGE_PNG)
            .data(URI.create("https://example.test/image.png").toURL())
            .build();
    var message = UserMessage.builder().text("picture").media(List.of(media)).build();

    var failure = assertThrows(ApplicationFailure.class, () -> MessagePartsCodec.encode(message));
    assertTrue(failure.isNonRetryable());
    assertTrue(failure.getOriginalMessage().contains("URI as a string"));
  }

  @Test
  void rejectsCustomMediaDataThatSpringAiWouldStringify() {
    var media =
        new Media(MimeTypeUtils.IMAGE_PNG, URI.create("https://example.test/image.png")) {
          @Override
          public Object getData() {
            return ByteBuffer.wrap(new byte[] {42});
          }
        };
    var message = UserMessage.builder().text("picture").media(List.of(media)).build();

    var failure = assertThrows(ApplicationFailure.class, () -> MessagePartsCodec.encode(message));
    assertTrue(failure.isNonRetryable());
    assertEquals("UnsupportedSpringAiMedia", failure.getType());
    assertTrue(failure.getOriginalMessage().contains("ByteBuffer"));
  }

  private static final class SystemPartsMessage extends AbstractMessage {
    private SystemPartsMessage(List<MessagePart> parts) {
      super(MessageType.SYSTEM, parts, Map.of("source", "test"));
    }
  }

  private static Message roundTrip(Message message) {
    var converter = DefaultDataConverter.STANDARD_INSTANCE;
    var payload = converter.toPayload(MessagePartsCodec.encode(message)).orElseThrow();
    return MessagePartsCodec.decode(
        converter.fromPayload(payload, ChatModelTypes.Message.class, ChatModelTypes.Message.class));
  }
}
