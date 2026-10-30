package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.google.adk.agents.LlmAgent;
import com.google.adk.events.Event;
import com.google.adk.models.BaseLlm;
import com.google.adk.models.Gemini;
import com.google.adk.runner.Runner;
import com.google.genai.Client;
import com.google.genai.types.Content;
import com.google.genai.types.HttpOptions;
import com.google.genai.types.HttpRetryOptions;
import com.google.genai.types.Part;
import com.sun.net.httpserver.HttpServer;
import io.temporal.activity.ActivityOptions;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowOptions;
import io.temporal.common.RetryOptions;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.googleadk.model.TemporalModel;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

@Timeout(30)
class ModelAliasWorkflowTest {
  private static final String ALIAS = "fast-alias";
  private static final String MODEL = "gemini-test-model";

  @ParameterizedTest
  @CsvSource({"false,''", "true,''", "false,gemini-callback-model", "true,gemini-callback-model"})
  void geminiUsesItsModelNameUnlessTheCallbackOverridesIt(boolean factory, String override)
      throws Exception {
    List<String> requests = new CopyOnWriteArrayList<>();
    HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    server.createContext(
        "/",
        exchange -> {
          try (exchange) {
            String path = exchange.getRequestURI().getPath();
            requests.add(path);
            exchange.getRequestBody().readAllBytes();
            boolean known =
                path.endsWith("/models/" + MODEL + ":generateContent")
                    || path.endsWith("/models/gemini-callback-model:generateContent");
            String json =
                known
                    ? "{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}"
                    : "{\"error\":{\"code\":404,\"message\":\"Unknown model alias\",\"status\":\"NOT_FOUND\"}}";
            byte[] body = json.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(known ? 200 : 404, body.length);
            exchange.getResponseBody().write(body);
          }
        });
    server.start();
    try (Client provider =
        Client.builder()
            .apiKey("test-key")
            .httpOptions(
                HttpOptions.builder()
                    .baseUrl("http://127.0.0.1:" + server.getAddress().getPort())
                    .apiVersion("v1beta")
                    .retryOptions(HttpRetryOptions.builder().attempts(1).build())
                    .build())
            .build()) {
      AtomicInteger factories = new AtomicInteger();
      Supplier<BaseLlm> supplier =
          () -> {
            factories.incrementAndGet();
            return Gemini.builder().modelName(MODEL).apiClient(provider).build();
          };
      GoogleAdkPlugin.Builder builder = GoogleAdkPlugin.newBuilder();
      if (factory) {
        builder.modelFactory(
            name -> {
              assertEquals(ALIAS, name);
              return supplier.get();
            });
      } else {
        builder.model(ALIAS, supplier);
      }
      GoogleAdkPlugin plugin = builder.build();
      TestEnvironmentOptions options =
          TestEnvironmentOptions.newBuilder()
              .setWorkflowClientOptions(
                  WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
              .build();
      try (TestWorkflowEnvironment environment = TestWorkflowEnvironment.newInstance(options)) {
        String queue = "model-alias-" + UUID.randomUUID();
        Worker worker = environment.newWorker(queue);
        worker.registerWorkflowImplementationTypes(AliasWorkflowImpl.class);
        environment.start();
        AliasWorkflow workflow =
            environment
                .getWorkflowClient()
                .newWorkflowStub(
                    AliasWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(queue).build());
        assertEquals("ok", workflow.ask(override));
        assertEquals(
            List.of(
                "/v1beta/models/" + (override.isEmpty() ? MODEL : override) + ":generateContent"),
            requests);
        assertEquals(1, factories.get());
        WorkflowExecutionHistory history =
            TestSupport.history(environment.getWorkflowClient(), workflow);
        assertEquals(List.of("googleadk.InvokeModel"), TestSupport.scheduledActivityTypes(history));
        try (TestWorkflowEnvironment replay = TestWorkflowEnvironment.newInstance(options)) {
          Worker fresh = replay.newWorker("alias-replay-" + UUID.randomUUID());
          fresh.registerWorkflowImplementationTypes(AliasWorkflowImpl.class);
          fresh.replayWorkflowExecution(history);
        }
        assertEquals(1, requests.size(), "replay must not call Gemini again");
      }
    } finally {
      server.stop(0);
    }
  }

  @WorkflowInterface
  public interface AliasWorkflow {
    @WorkflowMethod
    String ask(String override);
  }

  public static class AliasWorkflowImpl implements AliasWorkflow {
    @Override
    public String ask(String override) {
      LlmAgent agent =
          LlmAgent.builder()
              .name("assistant")
              .model(
                  TemporalModel.newBuilder(ALIAS)
                      .activityOptions(
                          ActivityOptions.newBuilder()
                              .setStartToCloseTimeout(Duration.ofSeconds(10))
                              .setRetryOptions(
                                  RetryOptions.newBuilder().setMaximumAttempts(1).build())
                              .build())
                      .build())
              .beforeModelCallbackSync(
                  (context, request) -> {
                    if (!override.isEmpty()) request.model(override);
                    return Optional.empty();
                  })
              .build();
      Runner runner = TemporalAdk.runnerBuilder().appName("alias").agent(agent).build();
      List<Event> events =
          TemporalAdk.run(runner, "user", "session", Content.fromParts(Part.fromText("Hi")));
      return TestSupport.lastText(events);
    }
  }
}
