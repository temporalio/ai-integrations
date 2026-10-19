package io.temporal.googleadk;

import static io.temporal.googleadk.TestSupport.history;
import static io.temporal.googleadk.TestSupport.scheduledActivityTypes;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.agents.LlmAgent;
import com.google.adk.events.Event;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.runner.Runner;
import com.google.adk.tools.Annotations.Schema;
import com.google.adk.tools.FunctionTool;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowOptions;
import io.temporal.client.WorkflowStub;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.googleadk.activity.ModelActivity;
import io.temporal.googleadk.model.TemporalModel;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.googleadk.tool.Hitl;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.QueryMethod;
import io.temporal.workflow.SignalMethod;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * A tool that requires confirmation pauses the run; a signal carries the decision and the next turn
 * resumes through ADK's own confirmation flow. The worker replays every task from history.
 */
class HitlWorkflowTest {

  private static final String TASK_QUEUE = "google-adk-hitl";
  private static final String MODEL = "scripted";
  static final AtomicInteger deletions = new AtomicInteger();

  private TestWorkflowEnvironment testEnv;
  private WorkflowClient client;
  private Worker worker;

  @BeforeEach
  void setUp() {
    deletions.set(0);
    GoogleAdkPlugin plugin =
        GoogleAdkPlugin.newBuilder()
            .model(MODEL, ScriptedModel.responding(MODEL, HitlWorkflowTest::respond))
            .build();
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                .build());
    client = testEnv.getWorkflowClient();
    worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(ApprovalWorkflowImpl.class);
    testEnv.start();
  }

  @AfterEach
  void tearDown() {
    testEnv.close();
  }

  private static LlmResponse respond(LlmRequest request) {
    Optional<Map<String, Object>> response = TestSupport.functionResponse(request, "deleteFile");
    if (response.isEmpty()) {
      return ScriptedModel.functionCall("deleteFile", Map.of("path", "/tmp/report.txt"));
    }
    Object error = response.get().get("error");
    if (error == null) {
      return ScriptedModel.text("Deleted " + response.get().get("deleted") + ".");
    }
    if (String.valueOf(error).contains("rejected")) {
      return ScriptedModel.text("Okay, I will not delete it.");
    }
    return ScriptedModel.text("I need your approval to delete the file.");
  }

  @Test
  @Timeout(90)
  void confirmedTool_runsAfterTheSignalAndReplays() throws Exception {
    ApprovalWorkflow workflow = newWorkflow();
    WorkflowClient.start(workflow::run, "Delete the report");
    awaitPending(workflow);
    assertEquals(List.of("deleteFile: approve"), workflow.pendingTools());

    workflow.decide(true);

    assertEquals(
        "Deleted /tmp/report.txt.", WorkflowStub.fromTyped(workflow).getResult(String.class));
    // An in-workflow tool runs again on a replay, so only "ran at all" is stable here; the
    // model's answer above shows it ran exactly once in the turn that mattered.
    assertTrue(deletions.get() >= 1);
    WorkflowExecutionHistory history = history(client, workflow);
    assertEquals(
        List.of(ModelActivity.INVOKE_MODEL, ModelActivity.INVOKE_MODEL, ModelActivity.INVOKE_MODEL),
        scheduledActivityTypes(history),
        "ask for the tool, explain the pause, answer after the tool ran");
    worker.replayWorkflowExecution(history);
    Path recorded = Path.of("build", "histories", "hitl_workflow.json");
    Files.createDirectories(recorded.getParent());
    Files.writeString(recorded, history.toJson(true), StandardCharsets.UTF_8);
  }

  @Test
  @Timeout(90)
  void rejectedTool_neverRuns() throws Exception {
    ApprovalWorkflow workflow = newWorkflow();
    WorkflowClient.start(workflow::run, "Delete the report");
    awaitPending(workflow);

    workflow.decide(false);

    assertEquals(
        "Okay, I will not delete it.", WorkflowStub.fromTyped(workflow).getResult(String.class));
    assertEquals(0, deletions.get());
    worker.replayWorkflowExecution(history(client, workflow));
  }

  private static void awaitPending(ApprovalWorkflow workflow) throws InterruptedException {
    for (int i = 0; i < 300 && workflow.pendingTools().isEmpty(); i++) {
      Thread.sleep(100);
    }
    assertTrue(!workflow.pendingTools().isEmpty(), "the run must pause for confirmation");
  }

  private ApprovalWorkflow newWorkflow() {
    return client.newWorkflowStub(
        ApprovalWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
  }

  @WorkflowInterface
  public interface ApprovalWorkflow {
    @WorkflowMethod
    String run(String question);

    @SignalMethod
    void decide(boolean confirmed);

    @QueryMethod
    List<String> pendingTools();
  }

  /** A tool with side effects that must be approved by a person first. */
  public static final class FileTools {
    private FileTools() {}

    public static Map<String, Object> deleteFile(
        @Schema(name = "path", description = "The file to delete") String path) {
      deletions.incrementAndGet();
      return Map.of("deleted", path);
    }
  }

  public static class ApprovalWorkflowImpl implements ApprovalWorkflow {
    private final List<String> pendingTools = new ArrayList<>();
    private Boolean decision;

    @Override
    public String run(String question) {
      LlmAgent agent =
          LlmAgent.builder()
              .name("assistant")
              .model(TemporalModel.create(MODEL))
              .instruction("Use the deleteFile tool when asked.")
              .tools(FunctionTool.create(FileTools.class, "deleteFile", true))
              .build();
      Runner runner = TemporalAdk.runnerBuilder().appName("approvals").agent(agent).build();
      List<Event> events =
          TemporalAdk.run(
              runner, "user-1", "session-1", Content.fromParts(Part.fromText(question)));
      List<Hitl.PendingConfirmation> pending = Hitl.pendingConfirmations(events);
      if (pending.isEmpty()) {
        return TestSupport.lastText(events);
      }
      // ADK's hint names the tool; the payload is only set by tools that supply one.
      pending.forEach(
          p ->
              pendingTools.add(
                  p.toolName()
                      + (p.hint() != null && p.hint().contains("approve") ? ": approve" : "")
                      + (p.payload() == null ? "" : ": payload")));
      Workflow.await(() -> decision != null);
      Content answer =
          Hitl.confirmationResponse(
              new Hitl.ConfirmationDecision(pending.get(0).functionCallId(), decision));
      events = TemporalAdk.run(runner, "user-1", "session-1", answer);
      pendingTools.clear();
      return TestSupport.lastText(events);
    }

    @Override
    public void decide(boolean confirmed) {
      this.decision = confirmed;
    }

    @Override
    public List<String> pendingTools() {
      return new ArrayList<>(pendingTools);
    }
  }
}
