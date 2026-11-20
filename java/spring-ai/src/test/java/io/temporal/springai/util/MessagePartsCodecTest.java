package io.temporal.springai.util;

import static org.junit.jupiter.api.Assertions.*;

import io.temporal.common.converter.DefaultDataConverter;
import io.temporal.failure.ApplicationFailure;
import io.temporal.springai.model.ChatModelTypes.Message;
import java.net.URI;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.messages.part.MediaPart;
import org.springframework.ai.chat.messages.part.TextPart;
import org.springframework.ai.chat.messages.part.ToolResultPart;
import org.springframework.ai.content.Media;
import org.springframework.util.MimeTypeUtils;

class MessagePartsCodecTest {
  @Test
  void nestedToolMediaAndUriRoundTripThroughTheTemporalConverter() {
    byte[] bytes = {0, 1, -1};
    String uri = "https://example.test/image.png";
    var inlineImage = Media.builder().mimeType(MimeTypeUtils.IMAGE_PNG).data(bytes).build();
    var linkedImage =
        Media.builder().mimeType(MimeTypeUtils.IMAGE_PNG).name("image reference").data(uri).build();
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

    var encoded = MessagePartsCodec.encode(tool);
    var converter = DefaultDataConverter.STANDARD_INSTANCE;
    var payload = converter.toPayload(encoded).orElseThrow();
    var decoded =
        assertInstanceOf(
            ToolResponseMessage.class,
            MessagePartsCodec.decode(converter.fromPayload(payload, Message.class, Message.class)));

    assertEquals(encoded, MessagePartsCodec.encode(decoded));
    var content = decoded.getResults().get(0).content();
    var restoredInline = assertInstanceOf(MediaPart.class, content.get(0)).media();
    assertArrayEquals(bytes, restoredInline.getDataAsByteArray());
    assertEquals("media-png-0-0", restoredInline.getName());
    assertEquals(TextPart.of("image result"), content.get(1));
    var restoredLink = assertInstanceOf(MediaPart.class, content.get(2)).media();
    assertEquals("image reference", restoredLink.getName());
    assertEquals(uri, restoredLink.getData());
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
}
