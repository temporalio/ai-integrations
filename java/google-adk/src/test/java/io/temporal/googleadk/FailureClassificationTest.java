package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.agents.LlmAgent;
import com.google.adk.models.BaseLlm;
import com.google.adk.models.BaseLlmConnection;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.runner.Runner;
import com.google.genai.errors.ClientException;
import com.google.genai.errors.ServerException;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.reactivex.rxjava3.core.Flowable;
import io.temporal.activity.ActivityOptions;
import io.temporal.api.enums.v1.RetryState;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowFailedException;
import io.temporal.client.WorkflowOptions;
import io.temporal.common.RetryOptions;
import io.temporal.failure.ActivityFailure;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.model.TemporalModel;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/** Model failures reach the workflow as typed application failures with the right retry class. */
class FailureClassificationTest {

  private static final String TASK_QUEUE = "google-adk-failures";

  private TestWorkflowEnvironment testEnv;
  private WorkflowClient client;

  @BeforeEach
  void setUp() {
    GoogleAdkPlugin plugin =
        GoogleAdkPlugin.newBuilder()
            .model(
                "bad_request",
                new ThrowingModel(
                    "bad_request", new ClientException(400, "INVALID_ARGUMENT", "bad prompt")))
            .model(
                "overloaded",
                new ThrowingModel("overloaded", new ServerException(503, "UNAVAILABLE", "busy")))
            .model(
                "bad_request_future",
                new FutureFailingModel(
                    "bad_request_future",
                    new ClientException(400, "INVALID_ARGUMENT", "bad prompt")))
            .model("silent", new SilentModel())
            .build();
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                .build());
    client = testEnv.getWorkflowClient();
    Worker worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(AskWorkflowImpl.class);
    testEnv.start();
  }

  @AfterEach
  void tearDown() {
    testEnv.close();
  }

  @Test
  @Timeout(90)
  void clientError_isNonRetryable() {
    ApplicationFailure failure = failureOf("bad_request");
    assertEquals("googleadk.ModelError", failure.getType());
    assertTrue(failure.isNonRetryable());
    assertTrue(failure.getOriginalMessage().contains("400"), failure.getOriginalMessage());
  }

  @Test
  @Timeout(90)
  void clientErrorRaisedThroughAFuture_isNonRetryable() {
    ApplicationFailure failure = failureOf("bad_request_future");
    assertEquals("googleadk.ModelError", failure.getType());
    assertTrue(failure.isNonRetryable());
    assertTrue(failure.getOriginalMessage().contains("400"), failure.getOriginalMessage());
  }

  @Test
  @Timeout(90)
  void serverError_isRetriedUntilTheRetryPolicyIsExhausted() {
    ActivityFailure activityFailure = activityFailureOf("overloaded");
    assertEquals(RetryState.RETRY_STATE_MAXIMUM_ATTEMPTS_REACHED, activityFailure.getRetryState());
    ApplicationFailure failure = (ApplicationFailure) activityFailure.getCause();
    assertEquals("googleadk.ModelError", failure.getType());
    assertFalse(failure.isNonRetryable());
  }

  @Test
  @Timeout(90)
  void emptyResponse_isRetryable() {
    ActivityFailure activityFailure = activityFailureOf("silent");
    assertEquals(RetryState.RETRY_STATE_MAXIMUM_ATTEMPTS_REACHED, activityFailure.getRetryState());
    ApplicationFailure failure = (ApplicationFailure) activityFailure.getCause();
    assertEquals("googleadk.ModelError", failure.getType());
    assertTrue(failure.getOriginalMessage().contains("no response"), failure.getOriginalMessage());
  }

  @Test
  @Timeout(90)
  void unknownModel_isNonRetryable() {
    ApplicationFailure failure = failureOf("no-such-model");
    assertEquals("googleadk.ModelError", failure.getType());
    assertTrue(failure.isNonRetryable());
    assertTrue(
        failure.getOriginalMessage().contains("Could not create model"),
        failure.getOriginalMessage());
  }

  private ApplicationFailure failureOf(String model) {
    return (ApplicationFailure) activityFailureOf(model).getCause();
  }

  private ActivityFailure activityFailureOf(String model) {
    AskWorkflow workflow =
        client.newWorkflowStub(
            AskWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
    WorkflowFailedException failed =
        assertThrows(WorkflowFailedException.class, () -> workflow.ask(model));
    assertInstanceOf(ActivityFailure.class, failed.getCause());
    ActivityFailure activityFailure = (ActivityFailure) failed.getCause();
    assertEquals("googleadk.InvokeModel", activityFailure.getActivityType());
    assertInstanceOf(ApplicationFailure.class, activityFailure.getCause());
    return activityFailure;
  }

  @WorkflowInterface
  public interface AskWorkflow {
    @WorkflowMethod
    String ask(String model);
  }

  public static class AskWorkflowImpl implements AskWorkflow {
    @Override
    public String ask(String model) {
      TemporalModel temporalModel =
          TemporalModel.newBuilder(model)
              .activityOptions(
                  ActivityOptions.newBuilder()
                      .setStartToCloseTimeout(Duration.ofSeconds(10))
                      .setRetryOptions(
                          RetryOptions.newBuilder()
                              .setInitialInterval(Duration.ofMillis(1))
                              .setBackoffCoefficient(1.0)
                              .setMaximumAttempts(2)
                              .build())
                      .build())
              .build();
      LlmAgent agent =
          LlmAgent.builder().name("assistant").model(temporalModel).instruction("Reply.").build();
      Runner runner = TemporalAdk.runnerBuilder().appName("failures").agent(agent).build();
      return TestSupport.lastText(
          TemporalAdk.run(runner, "user-1", "session-1", Content.fromParts(Part.fromText("hi"))));
    }
  }

  /** A model whose every call fails with the given exception. */
  static final class ThrowingModel extends BaseLlm {
    private final RuntimeException failure;

    ThrowingModel(String name, RuntimeException failure) {
      super(name);
      this.failure = failure;
    }

    @Override
    public Flowable<LlmResponse> generateContent(LlmRequest llmRequest, boolean stream) {
      return Flowable.error(failure);
    }

    @Override
    public BaseLlmConnection connect(LlmRequest llmRequest) {
      throw new UnsupportedOperationException();
    }
  }

  /** A model that fails the way ADK's Gemini model does: through a future wrapped in a flow. */
  static final class FutureFailingModel extends BaseLlm {
    private final RuntimeException failure;

    FutureFailingModel(String name, RuntimeException failure) {
      super(name);
      this.failure = failure;
    }

    @Override
    public Flowable<LlmResponse> generateContent(LlmRequest llmRequest, boolean stream) {
      return Flowable.fromFuture(CompletableFuture.failedFuture(failure));
    }

    @Override
    public BaseLlmConnection connect(LlmRequest llmRequest) {
      throw new UnsupportedOperationException();
    }
  }

  /** A model that completes without producing a response. */
  static final class SilentModel extends BaseLlm {
    SilentModel() {
      super("silent");
    }

    @Override
    public Flowable<LlmResponse> generateContent(LlmRequest llmRequest, boolean stream) {
      return Flowable.empty();
    }

    @Override
    public BaseLlmConnection connect(LlmRequest llmRequest) {
      throw new UnsupportedOperationException();
    }
  }
}
