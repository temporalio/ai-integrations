package io.temporal.googleadk.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.genai.errors.ClientException;
import com.google.genai.errors.ServerException;
import io.reactivex.rxjava3.exceptions.CompositeException;
import io.reactivex.rxjava3.exceptions.UndeliverableException;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.FailureTypes;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Model failures are classified by HTTP status, through RxJava's wrappers. */
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

  @Test
  void otherExceptions_areRetryable() {
    ApplicationFailure failure = Failures.modelFailure("m", new IllegalStateException("socket"));

    assertEquals(FailureTypes.MODEL_ERROR, failure.getType());
    assertFalse(failure.isNonRetryable());
    assertTrue(failure.getOriginalMessage().contains("socket"));
  }
}
