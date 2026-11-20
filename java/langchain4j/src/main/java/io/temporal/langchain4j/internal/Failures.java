package io.temporal.langchain4j.internal;

import dev.langchain4j.exception.NonRetriableException;
import io.temporal.failure.ApplicationFailure;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Classifies chat model failures for the activity's retry policy. */
public final class Failures {

  private static final Logger log = LoggerFactory.getLogger(Failures.class);

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
    // The provider's message, stack trace and causes can hold request data, such as the prompt or
    // an
    // API key in a URL, so they stay in the worker log; the history records only the exception
    // type.
    log.warn("Chat model {} failed", model, cause);
    String message = "Chat model " + model + " failed with " + cause.getClass().getName();
    if (findNonRetriable(cause) != null) {
      return ApplicationFailure.newNonRetryableFailure(message, MODEL_ERROR);
    }
    return ApplicationFailure.newFailure(message, MODEL_ERROR);
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
