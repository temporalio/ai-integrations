package io.temporal.langchain4j;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.exception.AuthenticationException;
import dev.langchain4j.exception.RateLimitException;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.output.OutputParsingException;
import io.temporal.activity.ActivityOptions;
import io.temporal.api.history.v1.HistoryEvent;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowFailedException;
import io.temporal.client.WorkflowOptions;
import io.temporal.client.WorkflowStub;
import io.temporal.common.RetryOptions;
import io.temporal.failure.ActivityFailure;
import io.temporal.failure.ApplicationFailure;
import io.temporal.langchain4j.activity.ChatModelActivity;
import io.temporal.langchain4j.activity.ChatModelActivityImpl;
import io.temporal.langchain4j.internal.Failures;
import io.temporal.langchain4j.model.TemporalChatModel;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.testing.WorkflowReplayer;
import io.temporal.worker.Worker;
import io.temporal.worker.WorkflowImplementationOptions;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/** Runs LangChain4j AI Services in workflows against scripted worker-side chat models. */
class ChatWorkflowTest {

  private static final String TASK_QUEUE = "langchain4j-test";
  private static final ActivityOptions FAST_RETRIES =
      ActivityOptions.newBuilder()
          .setStartToCloseTimeout(Duration.ofSeconds(10))
          .setRetryOptions(
              RetryOptions.newBuilder()
                  .setInitialInterval(Duration.ofMillis(10))
                  .setMaximumAttempts(3)
                  .build())
          .build();

  private TestWorkflowEnvironment testEnv;

  @AfterEach
  void tearDown() {
    if (testEnv != null) {
      testEnv.close();
    }
  }

  private WorkflowClient start(LangChain4jPlugin plugin, Class<?>... workflows) {
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                .build());
    Worker worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(workflows);
    testEnv.start();
    return testEnv.getWorkflowClient();
  }

  private static <T> T stub(WorkflowClient client, Class<T> type) {
    return client.newWorkflowStub(
        type, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
  }

  private static List<String> scheduledActivityTypes(WorkflowClient client, Object stub) {
    String workflowId = WorkflowStub.fromTyped(stub).getExecution().getWorkflowId();
    return client.fetchHistory(workflowId).getEvents().stream()
        .filter(HistoryEvent::hasActivityTaskScheduledEventAttributes)
        .map(e -> e.getActivityTaskScheduledEventAttributes().getActivityType().getName())
        .toList();
  }

  // --- An AI Service with a workflow-side tool ---

  public interface Assistant {
    String chat(String message);
  }

  /** Tools on a workflow object run on the workflow thread, so they must be deterministic. */
  public static final class Calculator {
    @Tool("Adds two numbers")
    int add(@P("a") int a, @P("b") int b) {
      return a + b;
    }
  }

  @WorkflowInterface
  public interface AssistantWorkflow {
    @WorkflowMethod
    String ask(String question);
  }

  public static final class AssistantWorkflowImpl implements AssistantWorkflow {
    @Override
    public String ask(String question) {
      Assistant assistant =
          AiServices.builder(Assistant.class)
              .chatModel(
                  TemporalChatModel.newBuilder("scripted").activityOptions(FAST_RETRIES).build())
              .tools(new Calculator())
              .chatMemory(MessageWindowChatMemory.withMaxMessages(20))
              .build();
      return assistant.chat(question);
    }
  }

  @Test
  void aiServiceCallsModelInActivitiesAndToolsInTheWorkflow() {
    ScriptedChatModel model =
        new ScriptedChatModel(
            request ->
                ChatResponse.builder()
                    .aiMessage(
                        AiMessage.from(
                            ToolExecutionRequest.builder()
                                .id("call-1")
                                .name("add")
                                .arguments("{\"a\":2,\"b\":3}")
                                .build()))
                    .build(),
            request -> ChatResponse.builder().aiMessage(AiMessage.from("The sum is 5.")).build());
    WorkflowClient client =
        start(
            LangChain4jPlugin.newBuilder().chatModel("scripted", model).build(),
            AssistantWorkflowImpl.class);

    AssistantWorkflow workflow = stub(client, AssistantWorkflow.class);
    assertEquals("The sum is 5.", workflow.ask("What is 2 + 3?"));

    assertEquals(
        List.of(ChatModelActivity.CHAT, ChatModelActivity.CHAT),
        scheduledActivityTypes(client, workflow));
    // The first request carried the tool specification; the second, the tool's result.
    assertEquals("add", model.requests.get(0).toolSpecifications().get(0).name());
    ToolExecutionResultMessage result =
        assertInstanceOf(
            ToolExecutionResultMessage.class,
            model.requests.get(1).messages().get(model.requests.get(1).messages().size() - 1));
    assertEquals("5", result.text());

    // AiServices and the workflow-side tool replay deterministically from the history.
    String workflowId = WorkflowStub.fromTyped(workflow).getExecution().getWorkflowId();
    assertDoesNotThrow(
        () ->
            WorkflowReplayer.replayWorkflowExecution(
                client.fetchHistory(workflowId), AssistantWorkflowImpl.class));
  }

  // --- Heartbeats ---

  @WorkflowInterface
  public interface HeartbeatWorkflow {
    @WorkflowMethod
    String ask(String question);
  }

  public static final class HeartbeatWorkflowImpl implements HeartbeatWorkflow {
    @Override
    public String ask(String question) {
      ActivityOptions options =
          ActivityOptions.newBuilder()
              .setStartToCloseTimeout(Duration.ofSeconds(10))
              .setHeartbeatTimeout(Duration.ofMillis(500))
              .setRetryOptions(RetryOptions.newBuilder().setMaximumAttempts(1).build())
              .build();
      Assistant assistant =
          AiServices.create(
              Assistant.class,
              TemporalChatModel.newBuilder("slow").activityOptions(options).build());
      return assistant.chat(question);
    }
  }

  @Test
  void aSlowModelCallHeartbeatsWhenAHeartbeatTimeoutIsSet() {
    ScriptedChatModel model =
        new ScriptedChatModel(
            request -> {
              try {
                Thread.sleep(1500);
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
              }
              return ChatResponse.builder().aiMessage(AiMessage.from("Done.")).build();
            });
    WorkflowClient client =
        start(
            LangChain4jPlugin.newBuilder().chatModel("slow", model).build(),
            HeartbeatWorkflowImpl.class);

    // Without heartbeats, the 500 ms heartbeat timeout would fail the only attempt.
    assertEquals("Done.", stub(client, HeartbeatWorkflow.class).ask("Hi"));
  }

  // --- Structured output with a JSON schema ---

  public record Forecast(String summary, int high) {}

  public interface Forecaster {
    Forecast forecast(String city);
  }

  @WorkflowInterface
  public interface ForecastWorkflow {
    @WorkflowMethod
    Forecast forecast(String city);
  }

  public static final class ForecastWorkflowImpl implements ForecastWorkflow {
    @Override
    public Forecast forecast(String city) {
      Forecaster forecaster =
          AiServices.create(
              Forecaster.class,
              TemporalChatModel.newBuilder("scripted")
                  .supportedCapabilities(Capability.RESPONSE_FORMAT_JSON_SCHEMA)
                  .build());
      return forecaster.forecast(city);
    }
  }

  @Test
  void structuredOutputSendsTheJsonSchemaToTheWorkerModel() {
    ScriptedChatModel model =
        new ScriptedChatModel(
            request ->
                ChatResponse.builder()
                    .aiMessage(AiMessage.from("{\"summary\":\"Snow\",\"high\":-3}"))
                    .build());
    WorkflowClient client =
        start(
            LangChain4jPlugin.newBuilder().chatModel("scripted", model).build(),
            ForecastWorkflowImpl.class);

    assertEquals(new Forecast("Snow", -3), stub(client, ForecastWorkflow.class).forecast("Oslo"));

    var format = model.requests.get(0).responseFormat();
    assertEquals(ResponseFormatType.JSON, format.type());
    assertNotNull(format.jsonSchema());
    assertNotNull(format.jsonSchema().rootElement());
  }

  @Test
  void invalidStructuredOutputFailsTheWorkflowWhenRegisteredAsAFailure() {
    ScriptedChatModel model =
        new ScriptedChatModel(
            request -> ChatResponse.builder().aiMessage(AiMessage.from("Not JSON")).build());
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder()
                        .setPlugins(
                            LangChain4jPlugin.newBuilder().chatModel("scripted", model).build())
                        .build())
                .build());
    Worker worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(
        WorkflowImplementationOptions.newBuilder()
            .setFailWorkflowExceptionTypes(OutputParsingException.class)
            .build(),
        ForecastWorkflowImpl.class);
    testEnv.start();
    WorkflowClient client = testEnv.getWorkflowClient();

    // Without the option, the workflow task would fail on every retry and the workflow would stay
    // running.
    WorkflowFailedException failure =
        assertThrows(
            WorkflowFailedException.class,
            () -> stub(client, ForecastWorkflow.class).forecast("Oslo"));
    ApplicationFailure cause = assertInstanceOf(ApplicationFailure.class, failure.getCause());
    assertEquals(OutputParsingException.class.getName(), cause.getType());
    assertEquals(1, model.requests.size());
  }

  // --- Failure classification ---

  @Test
  void finalProviderErrorsAreNotRetried() {
    AtomicInteger calls = new AtomicInteger();
    ScriptedChatModel model =
        new ScriptedChatModel(
            request -> {
              calls.incrementAndGet();
              throw new AuthenticationException("Invalid API key");
            });
    WorkflowClient client =
        start(
            LangChain4jPlugin.newBuilder().chatModel("scripted", model).build(),
            AssistantWorkflowImpl.class);

    WorkflowFailedException failure =
        assertThrows(
            WorkflowFailedException.class, () -> stub(client, AssistantWorkflow.class).ask("Hi"));
    ApplicationFailure cause = applicationFailure(failure);
    assertEquals(Failures.MODEL_ERROR, cause.getType());
    assertTrue(cause.isNonRetryable());
    assertEquals(1, calls.get());
    // The provider's message stays out of the workflow history.
    assertTrue(cause.getOriginalMessage().contains(AuthenticationException.class.getName()));
    assertFalse(cause.getOriginalMessage().contains("Invalid API key"));
    assertNull(cause.getCause());
  }

  @Test
  void otherProviderErrorsFollowTheRetryPolicy() {
    AtomicInteger calls = new AtomicInteger();
    ScriptedChatModel model =
        new ScriptedChatModel(
            request -> {
              if (calls.incrementAndGet() == 1) {
                throw new RateLimitException("Slow down");
              }
              return ChatResponse.builder().aiMessage(AiMessage.from("Hello!")).build();
            },
            request -> {
              calls.incrementAndGet();
              return ChatResponse.builder().aiMessage(AiMessage.from("Hello!")).build();
            });
    WorkflowClient client =
        start(
            LangChain4jPlugin.newBuilder().chatModel("scripted", model).build(),
            AssistantWorkflowImpl.class);

    assertEquals("Hello!", stub(client, AssistantWorkflow.class).ask("Hi"));
    assertEquals(2, calls.get());
  }

  @Test
  void anUnknownModelNameIsNotRetried() {
    WorkflowClient client =
        start(
            LangChain4jPlugin.newBuilder()
                .chatModel("other", new ScriptedChatModel(request -> null))
                .build(),
            AssistantWorkflowImpl.class);

    WorkflowFailedException failure =
        assertThrows(
            WorkflowFailedException.class, () -> stub(client, AssistantWorkflow.class).ask("Hi"));
    ApplicationFailure cause = applicationFailure(failure);
    assertEquals(ChatModelActivityImpl.UNKNOWN_MODEL, cause.getType());
    assertTrue(cause.isNonRetryable());
  }

  @Test
  void usingTheModelOutsideAWorkflowFails() {
    IllegalStateException error =
        assertThrows(
            IllegalStateException.class, () -> TemporalChatModel.create("scripted").chat("Hi"));
    assertTrue(error.getMessage().contains("only be used from a workflow"));
  }

  private static ApplicationFailure applicationFailure(WorkflowFailedException failure) {
    ActivityFailure activity = assertInstanceOf(ActivityFailure.class, failure.getCause());
    return assertInstanceOf(ApplicationFailure.class, activity.getCause());
  }
}
