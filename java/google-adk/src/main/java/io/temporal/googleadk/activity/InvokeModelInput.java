package io.temporal.googleadk.activity;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.adk.models.LlmRequest;
import javax.annotation.Nullable;

/**
 * Input of the {@code googleadk.InvokeModel} activity: the model name, the ADK request, and whether
 * the model should be asked for a streamed response. An optional topic enables live delivery to a
 * Workflow Stream; omitting it preserves the ordinary Activity payload shape.
 */
public record InvokeModelInput(
    @JsonProperty("model") String model,
    @JsonProperty("request") LlmRequest request,
    @JsonProperty("stream") boolean stream,
    @JsonProperty("streamingTopic") @JsonInclude(JsonInclude.Include.NON_NULL) @Nullable
        String streamingTopic) {

  /** Creates an input. */
  @JsonCreator
  public InvokeModelInput {}

  /** Creates an input without live delivery to a Workflow Stream. */
  public InvokeModelInput(String model, LlmRequest request, boolean stream) {
    this(model, request, stream, null);
  }
}
