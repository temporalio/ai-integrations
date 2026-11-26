package io.temporal.langchain4j.model;

import io.temporal.langchain4j.activity.ChatStreamPart;
import javax.annotation.Nullable;

/**
 * A live part of a streaming model's output or the terminal record of one model Activity attempt.
 * The Workflow ID, Run ID, Activity ID, attempt, and zero-based sequence identify a record; a
 * retried Activity keeps its ID, increments the attempt, and restarts the sequence. Terminal
 * records carry no part and are best effort: the Activity result is authoritative.
 *
 * @param workflowId the Workflow ID
 * @param runId the Run ID
 * @param activityId the model Activity's ID
 * @param model the registered model name
 * @param attempt the Activity attempt, starting at 1
 * @param sequence the record's position within the attempt, starting at 0
 * @param kind what the record carries
 * @param part the partial output of a {@link Kind#PART} record, otherwise {@code null}
 */
public record ModelStreamEvent(
    String workflowId,
    String runId,
    String activityId,
    String model,
    int attempt,
    long sequence,
    Kind kind,
    @Nullable ChatStreamPart part) {

  /** What a record carries. */
  public enum Kind {
    /** A partial response, thinking or tool call. */
    PART,
    /** The model completed its response. */
    COMPLETE,
    /** The attempt failed; the Activity's retry policy may start another. */
    FAILED,
    /** The attempt was abandoned: canceled, timed out, paused, reset, or the Worker shut down. */
    CANCELED
  }
}
