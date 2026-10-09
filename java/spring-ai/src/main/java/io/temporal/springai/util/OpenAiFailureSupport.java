package io.temporal.springai.util;

import com.openai.errors.OpenAIServiceException;
import io.temporal.failure.ApplicationFailure;
import java.math.BigDecimal;
import java.time.Duration;
import java.time.Instant;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.util.ClassUtils;

/** Preserves structured OpenAI failures and retry delays at the activity boundary. */
public final class OpenAiFailureSupport {
  private static final boolean SDK_PRESENT =
      ClassUtils.isPresent(
          "com.openai.errors.OpenAIServiceException", OpenAiFailureSupport.class.getClassLoader());

  private OpenAiFailureSupport() {}

  /**
   * Converts native HTTP failures to Temporal failures. Other exceptions, including application
   * failures supplied by the caller, retain their existing behavior. The OpenAI SDK is optional.
   *
   * @param error the failure thrown by the model
   * @return a structured application failure, or the original exception
   */
  public static RuntimeException convert(RuntimeException error) {
    if (error instanceof ApplicationFailure || !SDK_PRESENT) {
      return error;
    }
    return NativeFailures.convert(error);
  }

  // Keep references to optional SDK classes out of the outer class's method signatures and code.
  private static final class NativeFailures {
    private static RuntimeException convert(RuntimeException error) {
      Throwable cause = error;
      Set<Throwable> seen = Collections.newSetFromMap(new IdentityHashMap<>());
      while (cause != null && !(cause instanceof OpenAIServiceException)) {
        if (!seen.add(cause) || cause instanceof ApplicationFailure) {
          return error;
        }
        cause = cause.getCause();
      }
      if (!(cause instanceof OpenAIServiceException provider)) {
        return error;
      }

      int status = provider.statusCode();
      Object body = provider.body().isMissing() ? null : provider.body().convert(Object.class);
      boolean quotaExhausted =
          status == 429
              && body instanceof Map<?, ?> errorBody
              && ("insufficient_quota".equals(errorBody.get("code"))
                  || "insufficient_quota".equals(errorBody.get("type")));
      boolean retryable =
          !quotaExhausted && (status == 408 || status == 409 || status == 429 || status >= 500);
      Map<String, List<String>> headers = new HashMap<>();
      provider
          .headers()
          .names()
          .forEach(name -> headers.put(name, provider.headers().values(name)));

      Map<String, Object> details = new HashMap<>();
      details.put("statusCode", status);
      details.put("headers", headers);
      details.put("body", body);

      return ApplicationFailure.newBuilder()
          .setMessage(error.getMessage())
          .setType(provider.getClass().getName())
          .setCause(error)
          .setNonRetryable(!retryable)
          .setNextRetryDelay(
              retryable
                  ? retryDelay(
                      provider.headers().values("retry-after-ms"),
                      provider.headers().values("retry-after"),
                      Instant.now())
                  : null)
          .setDetails(details)
          .build();
    }
  }

  static Duration retryDelay(List<String> milliseconds, List<String> seconds, Instant now) {
    Duration delay = numericDelay(milliseconds, 1_000_000);
    if (delay != null) {
      return delay;
    }
    delay = numericDelay(seconds, 1_000_000_000);
    if (delay != null || seconds.isEmpty()) {
      return delay;
    }
    try {
      Instant deadline =
          ZonedDateTime.parse(seconds.get(0), DateTimeFormatter.RFC_1123_DATE_TIME).toInstant();
      delay = Duration.between(now, deadline);
      return delay.isNegative() || delay.isZero() ? null : delay;
    } catch (DateTimeParseException ignored) {
      return null;
    }
  }

  private static Duration numericDelay(List<String> values, long nanosPerUnit) {
    if (values.isEmpty()) {
      return null;
    }
    try {
      long nanos =
          new BigDecimal(values.get(0)).multiply(BigDecimal.valueOf(nanosPerUnit)).longValueExact();
      return nanos > 0 ? Duration.ofNanos(nanos) : null;
    } catch (NumberFormatException | ArithmeticException ignored) {
      return null;
    }
  }
}
