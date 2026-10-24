package io.temporal.googleadk.internal;

import com.google.genai.errors.ApiException;
import io.reactivex.rxjava3.exceptions.CompositeException;
import io.reactivex.rxjava3.exceptions.UndeliverableException;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.FailureTypes;
import java.util.List;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

/** Maps model failures to typed, correctly classified application failures. */
public final class Failures {

  /** How far down a cause chain the API error is looked for; it also ends a cyclic chain. */
  private static final int MAX_DEPTH = 16;

  private Failures() {}

  /**
   * Converts a model exception. GenAI API errors with status 408, 409, 429, or 5xx are retryable;
   * other 4xx responses are not. Everything else is retried. The API error is found through the
   * wrappers RxJava and futures add: ADK's Gemini model, for example, raises it as a {@code
   * RuntimeException} around an {@code ExecutionException}.
   */
  public static ApplicationFailure modelFailure(String model, Throwable cause) {
    ApiException api = findApiException(cause);
    if (api != null) {
      String message = "Model " + model + " failed with status " + api.code() + " " + api.status();
      if (api.code() < 400 || isRetryableStatus(api.code())) {
        return ApplicationFailure.newFailureWithCause(message, FailureTypes.MODEL_ERROR, cause);
      }
      return ApplicationFailure.newNonRetryableFailureWithCause(
          message, FailureTypes.MODEL_ERROR, cause);
    }
    Throwable root = unwrap(cause);
    String detail = root.getMessage() != null ? root.getMessage() : root.toString();
    return ApplicationFailure.newFailureWithCause(
        "Model " + model + " failed: " + detail, FailureTypes.MODEL_ERROR, cause);
  }

  /** Returns whether an HTTP status is worth retrying. */
  public static boolean isRetryableStatus(int status) {
    return status == 408 || status == 409 || status == 429 || status >= 500;
  }

  /** Finds the GenAI API error in the cause chain, if there is one within reach. */
  @Nullable
  static ApiException findApiException(Throwable throwable) {
    return findApiException(throwable, 0);
  }

  @Nullable
  private static ApiException findApiException(@Nullable Throwable throwable, int depth) {
    if (throwable == null || depth >= MAX_DEPTH) {
      return null;
    }
    if (throwable instanceof ApiException) {
      return (ApiException) throwable;
    }
    if (throwable instanceof CompositeException) {
      for (Throwable member : ((CompositeException) throwable).getExceptions()) {
        ApiException found = findApiException(member, depth + 1);
        if (found != null) {
          return found;
        }
      }
      return null;
    }
    return findApiException(next(throwable), depth + 1);
  }

  /** Strips the generic wrappers RxJava and futures add around an exception. */
  static Throwable unwrap(Throwable throwable) {
    Throwable current = throwable;
    for (int depth = 1; depth < MAX_DEPTH; depth++) {
      Throwable inner = isWrapper(current) ? next(current) : null;
      if (inner == null) {
        return current;
      }
      current = inner;
    }
    return current;
  }

  private static boolean isWrapper(Throwable throwable) {
    return throwable.getClass() == RuntimeException.class
        || throwable instanceof ExecutionException
        || throwable instanceof CompletionException
        || throwable instanceof CompositeException
        || throwable instanceof UndeliverableException;
  }

  @Nullable
  private static Throwable next(Throwable throwable) {
    if (throwable instanceof CompositeException) {
      List<Throwable> exceptions = ((CompositeException) throwable).getExceptions();
      return exceptions.isEmpty() ? null : exceptions.get(0);
    }
    Throwable cause = throwable.getCause();
    return cause == throwable ? null : cause;
  }
}
