package io.temporal.springai.util;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import com.openai.core.JsonValue;
import com.openai.core.http.Headers;
import com.openai.errors.OpenAIServiceException;
import io.temporal.failure.ApplicationFailure;
import java.net.URL;
import java.net.URLClassLoader;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.boot.test.context.FilteredClassLoader;

/** Covers provider classification, header edge cases, and the optional SDK boundary. */
class OpenAiFailureSupportTest {
  private static final Instant NOW = Instant.parse("1994-11-06T08:49:23Z");

  @ParameterizedTest
  @ValueSource(ints = {408, 409, 429, 500, 503, 504})
  void transientHttpErrorsAreRetryable(int status) {
    ApplicationFailure failure = failure(status, Map.of("message", "insufficient_quota"));
    assertFalse(failure.isNonRetryable());
  }

  @ParameterizedTest
  @ValueSource(ints = {400, 401, 403, 404, 422})
  void permanentHttpErrorsDoNotRetry(int status) {
    assertTrue(failure(status, Map.of("code", "invalid_request")).isNonRetryable());
  }

  @Test
  void quotaTypeIsPermanentAndKeepsStructuredBody() {
    ApplicationFailure failure = failure(429, Map.of("type", "insufficient_quota"));
    assertTrue(failure.isNonRetryable());
    assertNull(failure.getNextRetryDelay());
    assertEquals(
        Map.of("type", "insufficient_quota"), failure.getDetails().get(0, Map.class).get("body"));
  }

  @Test
  void callerApplicationFailuresAndUnrelatedExceptionsKeepTheirIdentity() {
    RuntimeException custom = new IllegalArgumentException("invalid prompt");
    assertSame(custom, OpenAiFailureSupport.convert(custom));
    ApplicationFailure explicit = ApplicationFailure.newFailure("user failure", "UserFailure");
    assertSame(explicit, OpenAiFailureSupport.convert(explicit));
    RuntimeException wrapped = new RuntimeException(explicit);
    assertSame(wrapped, OpenAiFailureSupport.convert(wrapped));
  }

  @Test
  void loadsAndHandlesErrorsWithoutOpenAiOnTheClasspath() throws Exception {
    URL classes = OpenAiFailureSupport.class.getProtectionDomain().getCodeSource().getLocation();
    try (var parent = new FilteredClassLoader("com.openai", "io.temporal.springai.util");
        var loader = new URLClassLoader(new URL[] {classes}, parent)) {
      Class<?> support = Class.forName(OpenAiFailureSupport.class.getName(), true, loader);
      RuntimeException error = new RuntimeException("offline model failed");
      assertSame(error, support.getMethod("convert", RuntimeException.class).invoke(null, error));
    }
  }

  @ParameterizedTest
  @MethodSource("delays")
  void retryHeadersUseProviderDelayOrFallBackToActivityPolicy(
      List<String> milliseconds, List<String> seconds, Duration expected) {
    assertEquals(expected, OpenAiFailureSupport.retryDelay(milliseconds, seconds, NOW));
  }

  private static Stream<Arguments> delays() {
    return Stream.of(
        Arguments.of(List.of("1500"), List.of("7"), Duration.ofMillis(1500)),
        Arguments.of(List.of("invalid"), List.of("7"), Duration.ofSeconds(7)),
        Arguments.of(List.of("-1"), List.of("7"), Duration.ofSeconds(7)),
        Arguments.of(List.of(), List.of("0.25"), Duration.ofMillis(250)),
        Arguments.of(List.of(), List.of("Sun, 06 Nov 1994 08:49:30 GMT"), Duration.ofSeconds(7)),
        Arguments.of(List.of(), List.of("Sun, 06 Nov 1994 08:49:20 GMT"), null),
        Arguments.of(List.of(), List.of("0"), null),
        Arguments.of(List.of(), List.of("-7"), null),
        Arguments.of(List.of(), List.of("NaN"), null),
        Arguments.of(List.of(), List.of("9999999999999999999999"), null),
        Arguments.of(List.of(), List.of("malformed"), null),
        Arguments.of(List.of(), List.of(), null));
  }

  private static ApplicationFailure failure(int status, Map<String, Object> body) {
    OpenAIServiceException error = mock(OpenAIServiceException.class);
    when(error.statusCode()).thenReturn(status);
    when(error.body()).thenReturn(JsonValue.from(body));
    when(error.headers()).thenReturn(Headers.builder().put("Retry-After", "7").build());
    return assertInstanceOf(
        ApplicationFailure.class, OpenAiFailureSupport.convert(new RuntimeException(error)));
  }
}
