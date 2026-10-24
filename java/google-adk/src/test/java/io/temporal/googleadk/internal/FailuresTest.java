package io.temporal.googleadk.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.genai.errors.ClientException;
import com.google.genai.errors.ServerException;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.exceptions.CompositeException;
import io.reactivex.rxjava3.exceptions.UndeliverableException;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.FailureTypes;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import org.junit.jupiter.api.Test;

/** Model failures are classified by HTTP status, through RxJava's and the futures' wrappers. */
class FailuresTest {

  @Test
  void retryableStatuses() {
    for (int status : new int[] {408, 409, 429, 500, 502, 503}) {
      assertTrue(Failures.isRetryableStatus(status), String.valueOf(status));
    }
    for (int status : new int[] {400, 401, 403, 404, 422}) {
      assertFalse(Failures.isRetryableStatus(status), String.valueOf(status));
    }
  }

  @Test
  void clientErrors_areNonRetryable() {
    ApplicationFailure failure =
        Failures.modelFailure("m", new ClientException(403, "PERMISSION_DENIED", "nope"));

    assertEquals(FailureTypes.MODEL_ERROR, failure.getType());
    assertTrue(failure.isNonRetryable());
    assertTrue(failure.getOriginalMessage().contains("403 PERMISSION_DENIED"));
  }

  @Test
  void serverErrors_areRetryable_evenWhenWrappedByRxJava() {
    ApplicationFailure composite =
        Failures.modelFailure(
            "m", new CompositeException(List.of(new ServerException(503, "UNAVAILABLE", "busy"))));
    ApplicationFailure undeliverable =
        Failures.modelFailure(
            "m", new UndeliverableException(new ClientException(400, "INVALID_ARGUMENT", "bad")));

    assertFalse(composite.isNonRetryable());
    assertTrue(composite.getOriginalMessage().contains("503"));
    assertTrue(undeliverable.isNonRetryable());
    assertTrue(undeliverable.getOriginalMessage().contains("400"));
  }

  /**
   * ADK's Gemini model wraps the API call in {@code Flowable.fromFuture}, so a failure reaches the
   * activity as a {@code RuntimeException} around the future's {@code ExecutionException}.
   */
  @Test
  void apiErrors_areClassified_throughFutureWrappers() {
    RuntimeException fromFuture =
        assertThrows(
            RuntimeException.class,
            () ->
                Flowable.fromFuture(
                        CompletableFuture.failedFuture(
                            new ClientException(400, "INVALID_ARGUMENT", "bad")))
                    .toList()
                    .blockingGet());
    assertInstanceOf(ExecutionException.class, fromFuture.getCause());

    ApplicationFailure future = Failures.modelFailure("m", fromFuture);
    ApplicationFailure completion =
        Failures.modelFailure(
            "m", new CompletionException(new ServerException(503, "UNAVAILABLE", "busy")));

    assertTrue(future.isNonRetryable());
    assertTrue(future.getOriginalMessage().contains("400 INVALID_ARGUMENT"));
    assertFalse(completion.isNonRetryable());
    assertTrue(completion.getOriginalMessage().contains("503 UNAVAILABLE"));
  }

  @Test
  void otherExceptions_areRetryable_withTheUnderlyingMessage() {
    ApplicationFailure plain = Failures.modelFailure("m", new IllegalStateException("socket"));
    ApplicationFailure wrapped =
        Failures.modelFailure(
            "m", new RuntimeException(new ExecutionException(new IOException("reset"))));

    assertEquals(FailureTypes.MODEL_ERROR, plain.getType());
    assertFalse(plain.isNonRetryable());
    assertTrue(plain.getOriginalMessage().contains("socket"));
    assertFalse(wrapped.isNonRetryable());
    assertTrue(
        wrapped.getOriginalMessage().endsWith("failed: reset"), wrapped.getOriginalMessage());
  }

  @Test
  void causeChains_areWalkedToABoundedDepth() {
    Throwable deep = new ClientException(400, "INVALID_ARGUMENT", "bad");
    for (int i = 0; i < 15; i++) {
      deep = new RuntimeException("wrapper " + i, deep);
    }
    RuntimeException first = new RuntimeException("first");
    RuntimeException second = new RuntimeException("second", first);
    first.initCause(second);

    assertTrue(Failures.modelFailure("m", deep).isNonRetryable());
    assertFalse(Failures.modelFailure("m", first).isNonRetryable(), "a cycle ends the walk");
  }
}
