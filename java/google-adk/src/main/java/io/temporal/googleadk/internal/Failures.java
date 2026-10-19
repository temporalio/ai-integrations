package io.temporal.googleadk.internal;

import com.google.genai.errors.ApiException;
import io.reactivex.rxjava3.exceptions.CompositeException;
import io.reactivex.rxjava3.exceptions.UndeliverableException;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.FailureTypes;

/** Maps model failures to typed, correctly classified application failures. */
public final class Failures {

  private Failures() {}

  /**
   * Converts a model exception. GenAI API errors with status 408, 409, 429, or 5xx are retryable;
   * other 4xx responses are not. Everything else is retried.
   */
  public static ApplicationFailure modelFailure(String model, Throwable cause) {
    Throwable root = unwrap(cause);
    if (root instanceof ApiException) {
      ApiException api = (ApiException) root;
      String message = "Model " + model + " failed with status " + api.code() + " " + api.status();
      if (isRetryableStatus(api.code())) {
        return ApplicationFailure.newFailureWithCause(message, FailureTypes.MODEL_ERROR, cause);
      }
      return ApplicationFailure.newNonRetryableFailureWithCause(
          message, FailureTypes.MODEL_ERROR, cause);
    }
    return ApplicationFailure.newFailureWithCause(
        "Model " + model + " failed: " + root.getMessage(), FailureTypes.MODEL_ERROR, cause);
  }

  /** Returns whether an HTTP status is worth retrying. */
  public static boolean isRetryableStatus(int status) {
    return status == 408 || status == 409 || status == 429 || status >= 500;
  }

  /** Strips the wrappers RxJava adds around an exception raised inside a flow. */
  static Throwable unwrap(Throwable throwable) {
    Throwable current = throwable;
    for (int depth = 0; depth < 8; depth++) {
      if (current instanceof CompositeException) {
        CompositeException composite = (CompositeException) current;
        if (composite.getExceptions().isEmpty()) {
          return current;
        }
        current = composite.getExceptions().get(0);
      } else if (current instanceof UndeliverableException && current.getCause() != null) {
        current = current.getCause();
      } else {
        return current;
      }
    }
    return current;
  }
}
