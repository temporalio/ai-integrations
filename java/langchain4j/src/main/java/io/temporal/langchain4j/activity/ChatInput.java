package io.temporal.langchain4j.activity;

import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Input of the {@code langchain4j.Chat} activity: the registered model name and a LangChain4j
 * {@code ChatRequest} in a form the Temporal payload converter can carry. Messages and tool
 * specifications use LangChain4j's own JSON forms.
 *
 * @param model the name the chat model is registered under on the worker
 * @param messages the request messages, as written by {@code ChatMessageSerializer}
 * @param parameters the request parameters, or {@code null} when the request has none
 */
public record ChatInput(String model, String messages, @Nullable Parameters parameters) {

  /**
   * The provider-independent {@code ChatRequestParameters}. Each field is {@code null} when the
   * request leaves it unset.
   *
   * @param modelName the model name sent to the provider
   * @param temperature the sampling temperature
   * @param topP the nucleus sampling probability
   * @param topK the number of tokens sampled from
   * @param frequencyPenalty the frequency penalty
   * @param presencePenalty the presence penalty
   * @param maxOutputTokens the maximum number of output tokens
   * @param stopSequences the stop sequences
   * @param toolSpecifications the tool specifications, each as written by {@code
   *     ToolSpecification.toJson()}
   * @param toolChoice the {@code ToolChoice} name
   * @param responseFormat the response format
   */
  public record Parameters(
      @Nullable String modelName,
      @Nullable Double temperature,
      @Nullable Double topP,
      @Nullable Integer topK,
      @Nullable Double frequencyPenalty,
      @Nullable Double presencePenalty,
      @Nullable Integer maxOutputTokens,
      @Nullable List<String> stopSequences,
      @Nullable List<String> toolSpecifications,
      @Nullable String toolChoice,
      @Nullable ResponseFormat responseFormat) {}

  /**
   * A {@code ResponseFormat}.
   *
   * @param type the {@code ResponseFormatType} name
   * @param schemaName the JSON schema name, or {@code null} without a schema
   * @param schema the JSON schema's root element as a JSON Schema map, or {@code null} without a
   *     schema
   */
  public record ResponseFormat(
      String type, @Nullable String schemaName, @Nullable Map<String, Object> schema) {}
}
