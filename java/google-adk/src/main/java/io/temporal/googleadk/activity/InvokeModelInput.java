package io.temporal.googleadk.activity;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.adk.models.LlmRequest;

/**
 * Input of the {@code googleadk.InvokeModel} activity: the model name, the ADK request, and whether
 * the model should be asked for a streamed response.
 */
public record InvokeModelInput(
    @JsonProperty("model") String model,
    @JsonProperty("request") LlmRequest request,
    @JsonProperty("stream") boolean stream) {

  /** Creates an input. */
  @JsonCreator
  public InvokeModelInput {}
}
