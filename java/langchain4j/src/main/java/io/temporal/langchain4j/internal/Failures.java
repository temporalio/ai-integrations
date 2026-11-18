package io.temporal.langchain4j.internal;

import dev.langchain4j.exception.NonRetriableException;
import io.temporal.failure.ApplicationFailure;
import javax.annotation.Nullable;

/** Classifies chat model failures for the activity's retry policy. */
public final class Failures {

  /** The failure type of a chat model error. */
  public static final String MODEL_ERROR = "LangChain4jModelError";

  /** How far down a cause chain a LangChain4j exception is looked for; it also ends a cycle. */
  private static final int MAX_DEPTH = 16;

  private Failures() {}

  /**
   * Converts a chat model exception. An exception that LangChain4j marks as final, a {@link
   * NonRetriableException} such as an authentication or invalid-request error, is not retried; the
   * retry policy decides for everything else.
   */
  public static ApplicationFailure modelFailure(String model, Throwable cause) {
    String detail = cause.getMessage() != null ? cause.getMessage() : cause.toString();
    String message = "Chat model " + model + " failed: " + detail;
    if (findNonRetriable(cause) != null) {
      return ApplicationFailure.newNonRetryableFailureWithCause(message, MODEL_ERROR, cause);
    }
    return ApplicationFailure.newFailureWithCause(message, MODEL_ERROR, cause);
  }

  @Nullable
  private static NonRetriableException findNonRetriable(Throwable throwable) {
    Throwable current = throwable;
    for (int depth = 0; current != null && depth < MAX_DEPTH; depth++) {
      if (current instanceof NonRetriableException nonRetriable) {
        return nonRetriable;
      }
      Throwable next = current.getCause();
      current = next == current ? null : next;
    }
    return null;
  }
}
