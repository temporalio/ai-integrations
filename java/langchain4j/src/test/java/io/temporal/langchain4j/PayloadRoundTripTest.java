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
import dev.langchain4j.service.tool.ToolExecutionResult;
import io.temporal.api.common.v1.Payload;
import io.temporal.common.converter.DataConverter;
import io.temporal.common.converter.DefaultDataConverter;
import io.temporal.langchain4j.activity.CallMcpToolInput;
import io.temporal.langchain4j.activity.CallMcpToolOutput;
import io.temporal.langchain4j.activity.ChatInput;
import io.temporal.langchain4j.activity.ChatOutput;
import io.temporal.langchain4j.activity.ChatStreamPart;
import io.temporal.langchain4j.activity.ListMcpToolsOutput;
import io.temporal.langchain4j.activity.StreamingChatInput;
import io.temporal.langchain4j.activity.StreamingChatOutput;
import io.temporal.langchain4j.internal.ChatCodec;
import io.temporal.langchain4j.internal.McpCodec;
import io.temporal.langchain4j.model.ModelStreamEvent;
import java.util.Base64;
import java.util.List;
import java.util.Map;
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

  @Test
  void streamingInputAndOutputWithEveryPartType() {
    StreamingChatInput input =
        new StreamingChatInput(
            ChatCodec.encode(
                "model", ChatRequest.builder().messages(UserMessage.from("Hi")).build()),
            "model-output");
    assertEquals(input, roundTrip(input, StreamingChatInput.class));
    assertEquals(
        new StreamingChatInput(input.request(), null),
        roundTrip(new StreamingChatInput(input.request(), null), StreamingChatInput.class));

    StreamingChatOutput output =
        new StreamingChatOutput(
            ChatCodec.encode(ChatResponse.builder().aiMessage(AiMessage.from("Hello")).build()),
            List.of(
                new ChatStreamPart(ChatStreamPart.Type.PARTIAL_THINKING, "hmm", null, null, null),
                new ChatStreamPart(ChatStreamPart.Type.PARTIAL_RESPONSE, "Hel", null, null, null),
                new ChatStreamPart(ChatStreamPart.Type.PARTIAL_TOOL_CALL, "{\"a\"", 0, null, "add"),
                new ChatStreamPart(
                    ChatStreamPart.Type.COMPLETE_TOOL_CALL, "{\"a\":1}", 0, "call-1", "add")));
    assertEquals(output, roundTrip(output, StreamingChatOutput.class));
  }

  @Test
  void modelStreamEvents() {
    ModelStreamEvent part =
        new ModelStreamEvent(
            "workflow",
            "run",
            "1",
            "model",
            2,
            3,
            ModelStreamEvent.Kind.PART,
            new ChatStreamPart(ChatStreamPart.Type.PARTIAL_RESPONSE, "Hel", null, null, null));
    assertEquals(part, roundTrip(part, ModelStreamEvent.class));
    ModelStreamEvent complete =
        new ModelStreamEvent(
            "workflow", "run", "1", "model", 2, 4, ModelStreamEvent.Kind.COMPLETE, null);
    assertEquals(complete, roundTrip(complete, ModelStreamEvent.class));
  }

  @Test
  void mcpToolSpecificationsWithParameterSchemas() {
    ToolSpecification tool =
        ToolSpecification.builder()
            .name("search")
            .description("Searches the docs")
            .parameters(
                JsonObjectSchema.builder()
                    .addStringProperty("query", "What to look for")
                    .addProperty("limit", JsonIntegerSchema.builder().build())
                    .required("query")
                    .build())
            .build();
    ListMcpToolsOutput output =
        roundTrip(new ListMcpToolsOutput(List.of(tool.toJson())), ListMcpToolsOutput.class);
    assertEquals(tool, ToolSpecification.fromJson(output.tools().get(0)));
  }

  @Test
  void mcpToolRequests() {
    ToolExecutionRequest request =
        ToolExecutionRequest.builder().id("call-1").name("search").arguments("{\"q\":1}").build();
    assertEquals(
        request,
        McpCodec.decode(roundTrip(McpCodec.encode("docs", request), CallMcpToolInput.class)));
    ToolExecutionRequest withoutId = ToolExecutionRequest.builder().name("search").build();
    assertEquals(
        withoutId,
        McpCodec.decode(roundTrip(McpCodec.encode("docs", withoutId), CallMcpToolInput.class)));
  }

  @Test
  void mcpToolResultsWithEveryContentTypeAStructuredResultAndAttributes() {
    String png = Base64.getEncoder().encodeToString(new byte[] {(byte) 0x89, 'P', 'N', 'G'});
    ToolExecutionResult result =
        ToolExecutionResult.builder()
            .resultContents(
                List.of(TextContent.from("Found it"), ImageContent.from(png, "image/png")))
            .result(Map.of("hits", List.of(1, 2), "next", Map.of("page", 2)))
            .attributes(Map.of("traceId", "abc", "cost", 3))
            .isError(true)
            .build();

    assertEquals(result, roundTripMcp(result));
  }

  @Test
  void mcpToolResultsWithTextOrNoContent() {
    ToolExecutionResult text = ToolExecutionResult.builder().resultText("plain").build();
    assertEquals(text, roundTripMcp(text));
    ToolExecutionResult empty =
        ToolExecutionResult.builder().resultTextSupplier(() -> null).build();
    ToolExecutionResult decoded = roundTripMcp(empty);
    assertEquals(List.of(), decoded.resultContents());
    assertEquals(empty, decoded);
  }

  @Test
  void mcpErrorsForTheModel() {
    CallMcpToolOutput output =
        roundTrip(McpCodec.errorForLlm("No such record."), CallMcpToolOutput.class);
    assertEquals("No such record.", output.errorForLlm());
  }

  private static ToolExecutionResult roundTripMcp(ToolExecutionResult result) {
    return McpCodec.decode(roundTrip(McpCodec.encode("search", result), CallMcpToolOutput.class));
  }

  /** Stands in for a provider's parameters class, such as {@code OpenAiChatRequestParameters}. */
  private static final class ProviderParameters extends DefaultChatRequestParameters {
    ProviderParameters() {
      super(DefaultChatRequestParameters.builder());
    }
  }
}
