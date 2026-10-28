package io.temporal.googleadk.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.adk.models.LlmResponse;
import javax.annotation.Nullable;

/**
 * A live model response or terminal record for one model Activity attempt. The workflow/run,
 * Activity ID, attempt, and zero-based sequence identify a record; a retried Activity restarts its
 * sequence with a higher attempt number. Terminal records have no response. A completed attempt is
 * not necessarily the final agent turn, and the Activity result is authoritative.
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

  /** Creates a live event. */
  @JsonCreator
  public ModelStreamEvent {}

  /** The model response or disposition of this Activity attempt. */
  public enum Kind {
    /** An ADK response, including partial and final responses. */
    RESPONSE,
    /** The provider completed successfully with at least one response. */
    COMPLETE,
    /** The attempt failed; its Activity retry policy may start another attempt. */
    FAILED,
    /** The Activity observed cancellation or timeout while the provider was running. */
    CANCELLED
  }
}
