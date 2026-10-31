package io.temporal.googleadk.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.adk.models.LlmResponse;
import javax.annotation.Nullable;

/**
 * A live model response or the terminal record of one model Activity attempt. The Workflow ID, Run
 * ID, Activity ID, attempt, and zero-based sequence identify a record; a retried Activity keeps its
 * ID, increments the attempt, and restarts the sequence. Terminal records carry no response and are
 * best effort: the Activity result is authoritative.
 */
public record ModelStreamEvent(
    @JsonProperty("workflowId") String workflowId,
    @JsonProperty("runId") String runId,
    @JsonProperty("activityId") String activityId,
    @JsonProperty("model") String model,
    @JsonProperty("attempt") int attempt,
    @JsonProperty("sequence") long sequence,
    @JsonProperty("kind") Kind kind,
    @JsonProperty("response") @Nullable LlmResponse response) {

  /** Creates an event. */
  @JsonCreator
  public ModelStreamEvent {}

  /** What a record carries. */
  public enum Kind {
    /** An ADK response, partial or final. */
    RESPONSE,
    /** The provider completed with at least one response. */
    COMPLETE,
    /** The attempt failed; the Activity's retry policy may start another. */
    FAILED,
    /** The attempt was abandoned: canceled, timed out, paused, reset, or the Worker shut down. */
    CANCELED
  }
}
