package io.temporal.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonEnumSchema;
import dev.langchain4j.model.chat.request.json.JsonIntegerSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import io.temporal.api.common.v1.Payload;
import io.temporal.common.converter.DataConverter;
import io.temporal.common.converter.DefaultDataConverter;
import io.temporal.langchain4j.activity.ChatInput;
import io.temporal.langchain4j.activity.ChatOutput;
import io.temporal.langchain4j.internal.ChatCodec;
import java.util.Base64;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Sends chat requests and responses through the real payload converter and back. */
class PayloadRoundTripTest {

  private static final DataConverter CONVERTER = DefaultDataConverter.newDefaultInstance();

  private static <T> T roundTrip(T value, Class<T> type) {
    Optional<Payload> payload = CONVERTER.toPayload(value);
    assertTrue(payload.isPresent());
    return CONVERTER.fromPayload(payload.get(), type, type);
  }

  private static ChatRequest roundTrip(ChatRequest request) {
    return ChatCodec.decode(roundTrip(ChatCodec.encode("model", request), ChatInput.class));
  }

  private static ChatResponse roundTrip(ChatResponse response) {
    return ChatCodec.decode(roundTrip(ChatCodec.encode(response), ChatOutput.class));
  }

  @Test
  void requestWithEveryParameterAndContentType() {
    JsonObjectSchema weatherParameters =
        JsonObjectSchema.builder()
            .addProperty("city", JsonStringSchema.builder().description("The city").build())
            .addProperty(
                "unit", JsonEnumSchema.builder().enumValues("celsius", "fahrenheit").build())
            .addProperty(
                "days",
                JsonArraySchema.builder().items(JsonIntegerSchema.builder().build()).build())
            .required("city")
            .build();
    ToolSpecification weather =
        ToolSpecification.builder()
            .name("weather")
            .description("Returns the weather")
            .parameters(weatherParameters)
            .build();
    ResponseFormat format =
        ResponseFormat.builder()
            .type(ResponseFormatType.JSON)
            .jsonSchema(
                JsonSchema.builder()
                    .name("Forecast")
                    .rootElement(
                        JsonObjectSchema.builder()
                            .addStringProperty("summary")
                            .addIntegerProperty("high")
                            .required("summary", "high")
                            .build())
                    .build())
            .build();
    byte[] png = new byte[] {(byte) 0x89, 'P', 'N', 'G', 0, 1, 2, (byte) 0xff};
    ChatRequest request =
        ChatRequest.builder()
            .messages(
                SystemMessage.from("You are a forecaster."),
                UserMessage.from(
                    TextContent.from("What do these show?"),
                    ImageContent.from(Base64.getEncoder().encodeToString(png), "image/png"),
                    ImageContent.from("https://example.com/map.png")),
                AiMessage.from(
                    ToolExecutionRequest.builder()
                        .id("call-1")
                        .name("weather")
                        .arguments("{\"city\":\"Oslo\",\"days\":[1,2]}")
                        .build()),
                ToolExecutionResultMessage.from("call-1", "weather", "Snow, -3°C"))
            .parameters(
                DefaultChatRequestParameters.builder()
                    .modelName("gpt-4o-mini")
                    .temperature(0.2)
                    .topP(0.9)
                    .topK(40)
                    .frequencyPenalty(0.1)
                    .presencePenalty(0.3)
                    .maxOutputTokens(256)
                    .stopSequences("END")
                    .toolSpecifications(weather)
                    .toolChoice(ToolChoice.REQUIRED)
                    .responseFormat(format)
                    .build())
            .build();

    assertEquals(request, roundTrip(request));
  }

  @Test
  void requestWithoutParameters() {
    ChatRequest request = ChatRequest.builder().messages(UserMessage.from("Hello")).build();

    assertEquals(request, roundTrip(request));
  }

  @Test
  void responseWithToolCallsAndMetadata() {
    ChatResponse response =
        ChatResponse.builder()
            .aiMessage(
                AiMessage.builder()
                    .text("Checking the weather.")
                    .thinking("The user wants Oslo.")
                    .toolExecutionRequests(
                        java.util.List.of(
                            ToolExecutionRequest.builder()
                                .id("call-2")
                                .name("weather")
                                .arguments("{\"city\":\"Oslo\"}")
                                .build()))
                    .build())
            .id("resp-1")
            .modelName("gpt-4o-mini")
            .tokenUsage(new TokenUsage(12, 7, 19))
            .finishReason(FinishReason.TOOL_EXECUTION)
            .build();

    assertEquals(response, roundTrip(response));
  }

  @Test
  void responseWithTextOnly() {
    ChatResponse response = ChatResponse.builder().aiMessage(AiMessage.from("Hi")).build();

    assertEquals(response, roundTrip(response));
  }

  @Test
  void providerSpecificParametersAreRefused() {
    ChatRequest request =
        ChatRequest.builder()
            .messages(UserMessage.from("Hello"))
            .parameters(new ProviderParameters())
            .build();

    IllegalArgumentException error =
        assertThrows(IllegalArgumentException.class, () -> ChatCodec.encode("model", request));
    assertTrue(error.getMessage().contains(ProviderParameters.class.getName()));
  }

  /** Stands in for a provider's parameters class, such as {@code OpenAiChatRequestParameters}. */
  private static final class ProviderParameters extends DefaultChatRequestParameters {
    ProviderParameters() {
      super(DefaultChatRequestParameters.builder());
    }
  }
}
