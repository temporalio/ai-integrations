package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.genai.types.Content;
import com.google.genai.types.FinishReason;
import com.google.genai.types.FunctionCall;
import com.google.genai.types.FunctionDeclaration;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponseUsageMetadata;
import com.google.genai.types.Part;
import com.google.genai.types.Schema;
import com.google.genai.types.Tool;
import com.google.genai.types.Type;
import io.temporal.api.common.v1.Payload;
import io.temporal.common.converter.DefaultDataConverter;
import io.temporal.googleadk.converter.GoogleAdkDataConverter;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** ADK's request and response types survive the plugin's data converter unchanged and compact. */
class PayloadRoundTripTest {

  private final DefaultDataConverter converter = GoogleAdkDataConverter.newDefaultInstance();

  @Test
  void llmRequestRoundTrips() {
    LlmRequest request =
        LlmRequest.builder()
            .model("gemini-2.5-flash")
            .contents(
                List.of(
                    Content.fromParts(Part.fromText("hi")),
                    Content.builder()
                        .role("model")
                        .parts(
                            List.of(
                                Part.fromFunctionCall("lookup", Map.of("city", "Paris", "n", 2))))
                        .build()))
            .config(
                GenerateContentConfig.builder()
                    .temperature(0.1f)
                    .systemInstruction(Content.fromParts(Part.fromText("be brief")))
                    .tools(
                        List.of(
                            Tool.builder()
                                .functionDeclarations(
                                    List.of(
                                        FunctionDeclaration.builder()
                                            .name("lookup")
                                            .description("looks up")
                                            .parameters(
                                                Schema.builder()
                                                    .type(Type.Known.OBJECT)
                                                    .properties(
                                                        Map.of(
                                                            "city",
                                                            Schema.builder()
                                                                .type(Type.Known.STRING)
                                                                .build()))
                                                    .build())
                                            .build()))
                                .build()))
                    .build())
            .build();

    Optional<Payload> payload = converter.toPayload(request);
    assertTrue(payload.isPresent());
    String json = payload.get().getData().toStringUtf8();
    assertFalse(json.contains("\"tools\":{"), "ignored tool instances must not be serialized");
    assertFalse(json.contains("null"), "absent optionals must be omitted");
    LlmRequest back = converter.fromPayload(payload.get(), LlmRequest.class, LlmRequest.class);
    assertEquals(request.model(), back.model());
    assertEquals(request.contents(), back.contents());
    assertEquals(request.config(), back.config());
    assertEquals(request.toJson(), back.toJson());
  }

  @Test
  void llmResponseRoundTrips() {
    LlmResponse response =
        LlmResponse.builder()
            .content(
                Content.builder()
                    .role("model")
                    .parts(
                        List.of(
                            Part.fromText("hello"),
                            Part.builder()
                                .functionCall(
                                    FunctionCall.builder()
                                        .id("call-1")
                                        .name("lookup")
                                        .args(Map.of("city", "Paris"))
                                        .build())
                                .build()))
                    .build())
            .partial(false)
            .turnComplete(true)
            .finishReason(new FinishReason(FinishReason.Known.STOP))
            .usageMetadata(
                GenerateContentResponseUsageMetadata.builder()
                    .promptTokenCount(3)
                    .candidatesTokenCount(5)
                    .totalTokenCount(8)
                    .build())
            .build();

    Optional<Payload> payload = converter.toPayload(response);
    assertTrue(payload.isPresent());
    LlmResponse back = converter.fromPayload(payload.get(), LlmResponse.class, LlmResponse.class);
    assertEquals(response, back);
    assertEquals(response.toJson(), back.toJson());
  }
}
