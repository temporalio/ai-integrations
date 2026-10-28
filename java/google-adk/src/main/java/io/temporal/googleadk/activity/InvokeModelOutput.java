package io.temporal.googleadk.activity;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.adk.models.LlmResponse;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Output of the {@code googleadk.InvokeModel} activity: every response the model produced, in
 * order. A non-streamed call yields one response; a streamed call yields the partial chunks
 * followed by the final response, exactly as the model emitted them.
 */
public record InvokeModelOutput(@JsonProperty("responses") List<LlmResponse> responses) {

  /** Creates an output. */
  @JsonCreator
  public InvokeModelOutput {
    responses =
        responses == null ? List.of() : Collections.unmodifiableList(new ArrayList<>(responses));
  }
}
