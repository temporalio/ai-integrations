package io.temporal.springai;

import static org.junit.jupiter.api.Assertions.*;

import com.sun.net.httpserver.HttpServer;
import io.temporal.activity.Activity;
import io.temporal.client.WorkflowException;
import io.temporal.client.WorkflowOptions;
import io.temporal.failure.ActivityFailure;
import io.temporal.failure.ApplicationFailure;
import io.temporal.springai.activity.ChatModelActivityImpl;
import io.temporal.springai.chat.TemporalChatClient;
import io.temporal.springai.model.ActivityChatModel;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;

/** Real Spring AI 2 and provider SDK calls against a local HTTP stub. */
@Timeout(30)
class ProviderRateLimitRetryTest {
  private HttpServer server;
  private TestWorkflowEnvironment environment;
  private final List<Integer> attempts = new CopyOnWriteArrayList<>();
  private final AtomicInteger currentAttempt = new AtomicInteger();
  private final List<Long> attemptTimes = new CopyOnWriteArrayList<>();
  private volatile int status = 429;
  private volatile String code = "rate_limit_exceeded";
  private volatile String retryAfter = "7";

  @BeforeEach
  void setUp() throws Exception {
    environment = TestWorkflowEnvironment.newInstance();
    server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    server.createContext(
        "/v1/chat/completions",
        exchange -> {
          exchange.getRequestBody().readAllBytes();
          attempts.add(currentAttempt.get());
          byte[] body =
              ("{\"error\":{\"message\":\"provider failure\",\"type\":\""
                      + code
                      + "\",\"code\":\""
                      + code
                      + "\",\"param\":null}}")
                  .getBytes(StandardCharsets.UTF_8);
          exchange.getResponseHeaders().add("Content-Type", "application/json");
          exchange.getResponseHeaders().add("Retry-After", retryAfter);
          exchange.sendResponseHeaders(status, body.length);
          try (var response = exchange.getResponseBody()) {
            response.write(body);
          }
        });
    server.start();
  }

  @AfterEach
  void tearDown() {
    environment.close();
    server.stop(0);
  }

  @Test
  void rateLimitUsesTemporalAttemptsAndProviderDelay() {
    ApplicationFailure failure = run(0);
    assertEquals(List.of(1, 2, 3), attempts);
    assertEquals("com.openai.errors.RateLimitException", failure.getType());
    assertFalse(failure.isNonRetryable());
    assertEquals(Duration.ofSeconds(7), failure.getNextRetryDelay());
    var details = failure.getDetails().get(0, java.util.Map.class);
    assertEquals(429, details.get("statusCode"));
    assertEquals("rate_limit_exceeded", ((java.util.Map<?, ?>) details.get("body")).get("code"));
    var headers = (java.util.Map<?, ?>) details.get("headers");
    assertTrue(
        headers.entrySet().stream()
            .anyMatch(
                entry ->
                    entry.getKey().toString().equalsIgnoreCase("Retry-After")
                        && entry.getValue().equals(List.of("7"))));
    assertTrue(attemptTimes.get(1) - attemptTimes.get(0) >= 7000);
    assertTrue(attemptTimes.get(2) - attemptTimes.get(1) >= 7000);
  }

  @Test
  void quotaExhaustionDoesNotRetry() {
    code = "insufficient_quota";
    ApplicationFailure failure = run(0);
    assertEquals(List.of(1), attempts);
    assertTrue(failure.isNonRetryable());
  }

  @Test
  void badApiKeyDoesNotRetry() {
    status = 401;
    code = "invalid_api_key";
    ApplicationFailure failure = run(0);
    assertEquals(List.of(1), attempts);
    assertTrue(failure.isNonRetryable());
  }

  @Test
  void serviceFailureUsesOnlyTemporalRetries() {
    status = 503;
    retryAfter = "0.01";
    ApplicationFailure failure = run(0);
    assertEquals(List.of(1, 2, 3), attempts);
    assertFalse(failure.isNonRetryable());
  }

  private ApplicationFailure run(int maxRetries) {
    OpenAiChatModel model =
        OpenAiChatModel.builder()
            .options(
                OpenAiChatOptions.builder()
                    .apiKey("local-test-key")
                    .baseUrl("http://127.0.0.1:" + server.getAddress().getPort() + "/v1")
                    .model("test-model")
                    .maxRetries(maxRetries)
                    .build())
            .build();
    Worker worker = environment.newWorker("provider-retry-test");
    worker.registerWorkflowImplementationTypes(ChatWorkflowImpl.class);
    worker.registerActivitiesImplementations(
        new ChatModelActivityImpl(
            new ChatModel() {
              @Override
              public ChatOptions getOptions() {
                return model.getOptions();
              }

              @Override
              public ChatResponse call(Prompt prompt) {
                currentAttempt.set(Activity.getExecutionContext().getInfo().getAttempt());
                attemptTimes.add(environment.currentTimeMillis());
                return model.call(prompt);
              }
            }));
    environment.start();
    ChatWorkflow workflow =
        environment
            .getWorkflowClient()
            .newWorkflowStub(
                ChatWorkflow.class,
                WorkflowOptions.newBuilder().setTaskQueue("provider-retry-test").build());
    WorkflowException error = assertThrows(WorkflowException.class, workflow::chat);
    ActivityFailure activityFailure = assertInstanceOf(ActivityFailure.class, error.getCause());
    return assertInstanceOf(ApplicationFailure.class, activityFailure.getCause());
  }

  @WorkflowInterface
  public interface ChatWorkflow {
    @WorkflowMethod
    String chat();
  }

  public static class ChatWorkflowImpl implements ChatWorkflow {
    @Override
    public String chat() {
      return TemporalChatClient.builder(ActivityChatModel.forDefault())
          .build()
          .prompt()
          .user("ping")
          .call()
          .content();
    }
  }
}
