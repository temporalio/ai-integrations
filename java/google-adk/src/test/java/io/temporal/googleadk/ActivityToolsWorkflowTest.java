package io.temporal.googleadk;

import static io.temporal.googleadk.TestSupport.history;
import static io.temporal.googleadk.TestSupport.scheduledActivityTypes;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.agents.LlmAgent;
import com.google.adk.events.Event;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.runner.Runner;
import com.google.adk.tools.Annotations.Schema;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionCall;
import com.google.genai.types.Part;
import io.temporal.activity.ActivityInterface;
import io.temporal.activity.ActivityMethod;
import io.temporal.activity.ActivityOptions;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowFailedException;
import io.temporal.client.WorkflowOptions;
import io.temporal.client.WorkflowStub;
import io.temporal.common.RetryOptions;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.failure.ActivityFailure;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.activity.ModelActivity;
import io.temporal.googleadk.model.TemporalModel;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.googleadk.tool.ActivityTools;
import io.temporal.googleadk.tool.Hitl;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.QueryMethod;
import io.temporal.workflow.SignalMethod;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Activity-backed tools: a failed activity fails the run with its typed failure, a tool that
 * requires confirmation waits for a signal before the activity runs, and a long-running tool pauses
 * the run until the result arrives.
 */
class ActivityToolsWorkflowTest {

  private static final String TASK_QUEUE = "google-adk-activity-tools";
  private static final String MODEL = "scripted";

  private TestWorkflowEnvironment testEnv;
  private WorkflowClient client;
  private Worker worker;

  @BeforeEach
  void setUp() {
    GoogleAdkPlugin plugin =
        GoogleAdkPlugin.newBuilder()
            .model(MODEL, ScriptedModel.responding(MODEL, ActivityToolsWorkflowTest::respond))
            .build();
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                .build());
    client = testEnv.getWorkflowClient();
    worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(ToolWorkflowImpl.class);
    worker.registerActivitiesImplementations(new JobActivitiesImpl());
    testEnv.start();
  }

  @AfterEach
  void tearDown() {
    testEnv.close();
  }

  private static LlmResponse respond(LlmRequest request) {
    Optional<Map<String, Object>> lookup = TestSupport.functionResponse(request, "lookup");
    if (lookup.isPresent()) {
      Object error = lookup.get().get("error");
      if (error != null) {
        return ScriptedModel.text("I need your approval to look that up.");
      }
      return ScriptedModel.text("Looked up: " + lookup.get().get("result"));
    }
    Optional<Map<String, Object>> job = TestSupport.functionResponse(request, "startJob");
    if (job.isPresent()) {
      return "pending".equals(job.get().get("status"))
          ? ScriptedModel.text("Job started.")
          : ScriptedModel.text("Job finished: " + job.get().get("result"));
    }
    String question = lastUserText(request);
    if (question.contains("fail")) {
      return ScriptedModel.functionCall("boom", Map.of("reason", "test"));
    }
    if (question.contains("job")) {
      return ScriptedModel.functionCall("startJob", Map.of("name", "nightly"));
    }
    return ScriptedModel.functionCall("lookup", Map.of("key", "k1"));
  }

  private static String lastUserText(LlmRequest request) {
    for (int i = request.contents().size() - 1; i >= 0; i--) {
      Content content = request.contents().get(i);
      if (content.role().orElse("").equals("user") && content.text() != null) {
        return content.text();
      }
    }
    return "";
  }

  @Test
  @Timeout(90)
  void failedActivity_failsTheRunWithItsTypedFailure() {
    ToolWorkflow workflow = newWorkflow();

    WorkflowFailedException failed =
        assertThrows(WorkflowFailedException.class, () -> workflow.run("please fail"));

    assertInstanceOf(ActivityFailure.class, failed.getCause());
    ActivityFailure activityFailure = (ActivityFailure) failed.getCause();
    assertEquals("Boom", activityFailure.getActivityType());
    assertInstanceOf(ApplicationFailure.class, activityFailure.getCause());
    ApplicationFailure cause = (ApplicationFailure) activityFailure.getCause();
    assertEquals("AuthError", cause.getType());
    assertTrue(cause.isNonRetryable());
    assertEquals(
        List.of(ModelActivity.INVOKE_MODEL, "Boom"),
        scheduledActivityTypes(
            client.fetchHistory(WorkflowStub.fromTyped(workflow).getExecution().getWorkflowId())),
        "the model is not consulted again after the tool failed");
  }

  @Test
  @Timeout(90)
  void confirmedActivityTool_runsOnlyAfterTheSignal() throws Exception {
    ToolWorkflow workflow = newWorkflow();
    WorkflowClient.start(workflow::run, "look up k1");
    awaitPending(workflow);
    assertEquals(List.of("confirm:lookup"), workflow.pending());

    workflow.decide(true);

    assertEquals(
        "Looked up: value-for-k1", WorkflowStub.fromTyped(workflow).getResult(String.class));
    WorkflowExecutionHistory history = history(client, workflow);
    assertEquals(
        List.of(
            ModelActivity.INVOKE_MODEL, // asks for the tool
            ModelActivity.INVOKE_MODEL, // explains the pause
            "Lookup", // runs after the confirmation
            ModelActivity.INVOKE_MODEL), // answers
        scheduledActivityTypes(history));
    worker.replayWorkflowExecution(history);
  }

  @Test
  @Timeout(90)
  void longRunningActivityTool_pausesUntilTheResultArrives() throws Exception {
    ToolWorkflow workflow = newWorkflow();
    WorkflowClient.start(workflow::run, "start the job");
    awaitPending(workflow);
    assertEquals(List.of("job:startJob"), workflow.pending());

    workflow.jobDone("42");

    assertEquals("Job finished: 42", WorkflowStub.fromTyped(workflow).getResult(String.class));
    WorkflowExecutionHistory history = history(client, workflow);
    assertEquals(
        List.of(
            ModelActivity.INVOKE_MODEL, // asks for the tool
            "StartJob", // returns the interim status
            ModelActivity.INVOKE_MODEL, // acknowledges it; the turn ends with the call pending
            ModelActivity.INVOKE_MODEL), // answers once the result arrives
        scheduledActivityTypes(history));
    worker.replayWorkflowExecution(history);
  }

  private static void awaitPending(ToolWorkflow workflow) throws InterruptedException {
    for (int i = 0; i < 300 && workflow.pending().isEmpty(); i++) {
      Thread.sleep(100);
    }
    assertTrue(!workflow.pending().isEmpty(), "the run must pause");
  }

  private ToolWorkflow newWorkflow() {
    return client.newWorkflowStub(
        ToolWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
  }

  @WorkflowInterface
  public interface ToolWorkflow {
    @WorkflowMethod
    String run(String question);

    @SignalMethod
    void decide(boolean confirmed);

    @SignalMethod
    void jobDone(String result);

    @QueryMethod
    List<String> pending();
  }

  @ActivityInterface
  public interface JobActivities {
    @ActivityMethod
    String boom(@Schema(name = "reason", description = "Why") String reason);

    @ActivityMethod
    String lookup(@Schema(name = "key", description = "The key") String key);

    @ActivityMethod
    Map<String, Object> startJob(@Schema(name = "name", description = "The job") String name);
  }

  public static class JobActivitiesImpl implements JobActivities {
    @Override
    public String boom(String reason) {
      throw ApplicationFailure.newNonRetryableFailure("no access: " + reason, "AuthError");
    }

    @Override
    public String lookup(String key) {
      return "value-for-" + key;
    }

    @Override
    public Map<String, Object> startJob(String name) {
      return Map.of("status", "pending", "job", name);
    }
  }

  public static class ToolWorkflowImpl implements ToolWorkflow {
    private final List<String> pending = new ArrayList<>();
    private Boolean decision;
    private String jobResult;

    @Override
    public String run(String question) {
      JobActivities jobs =
          Workflow.newActivityStub(
              JobActivities.class,
              ActivityOptions.newBuilder()
                  .setStartToCloseTimeout(Duration.ofSeconds(10))
                  .setRetryOptions(RetryOptions.newBuilder().setMaximumAttempts(1).build())
                  .build());
      LlmAgent agent =
          LlmAgent.builder()
              .name("assistant")
              .model(TemporalModel.create(MODEL))
              .instruction("Use the tools.")
              .tools(
                  ActivityTools.create(jobs, "boom"),
                  ActivityTools.create(jobs, "lookup", true),
                  ActivityTools.createLongRunning(jobs, "startJob"))
              .build();
      Runner runner = TemporalAdk.runnerBuilder().appName("tools").agent(agent).build();
      List<Event> events =
          TemporalAdk.run(
              runner, "user-1", "session-1", Content.fromParts(Part.fromText(question)));

      List<Hitl.PendingConfirmation> confirmations = Hitl.pendingConfirmations(events);
      if (!confirmations.isEmpty()) {
        pending.add("confirm:" + confirmations.get(0).toolName());
        Workflow.await(() -> decision != null);
        events =
            TemporalAdk.run(
                runner,
                "user-1",
                "session-1",
                Hitl.confirmationResponse(
                    new Hitl.ConfirmationDecision(
                        confirmations.get(0).functionCallId(), decision)));
        pending.clear();
        return TestSupport.lastText(events);
      }
      List<FunctionCall> longRunning = Hitl.longRunningCalls(events);
      if (!longRunning.isEmpty()) {
        pending.add("job:" + longRunning.get(0).name().orElse(""));
        Workflow.await(() -> jobResult != null);
        events =
            TemporalAdk.run(
                runner,
                "user-1",
                "session-1",
                Hitl.functionResponse(
                    longRunning.get(0), Map.of("status", "done", "result", jobResult)));
        pending.clear();
      }
      return TestSupport.lastText(events);
    }

    @Override
    public void decide(boolean confirmed) {
      this.decision = confirmed;
    }

    @Override
    public void jobDone(String result) {
      this.jobResult = result;
    }

    @Override
    public List<String> pending() {
      return new ArrayList<>(pending);
    }
  }
}
