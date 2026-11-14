package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.agents.LlmAgent;
import com.google.adk.events.Event;
import com.google.adk.runner.Runner;
import com.google.adk.sessions.Session;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.temporal.api.enums.v1.EventType;
import io.temporal.api.history.v1.HistoryEvent;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowOptions;
import io.temporal.client.WorkflowStub;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.googleadk.model.TemporalModel;
import io.temporal.googleadk.session.Sessions;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/** A conversation survives continue-as-new through an exported session snapshot. */
class ContinueAsNewWorkflowTest {

  private static final String TASK_QUEUE = "google-adk-can";
  private static final String MODEL = "scripted";

  private TestWorkflowEnvironment testEnv;
  private WorkflowClient client;
  private Worker worker;

  private ScriptedModel model;

  @BeforeEach
  void setUp() {
    model =
        ScriptedModel.responding(
            MODEL,
            request -> {
              long userTurns =
                  request.contents().stream()
                      .filter(content -> content.role().orElse("").equals("user"))
                      .count();
              return ScriptedModel.text("turns=" + userTurns);
            });
    GoogleAdkPlugin plugin = GoogleAdkPlugin.newBuilder().model(MODEL, model).build();
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                .build());
    client = testEnv.getWorkflowClient();
    worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(ConversationWorkflowImpl.class);
    testEnv.start();
  }

  @AfterEach
  void tearDown() {
    testEnv.close();
  }

  @Test
  @Timeout(90)
  void sessionSnapshot_carriesTheConversationAcrossContinueAsNew() throws Exception {
    ConversationWorkflow workflow =
        client.newWorkflowStub(
            ConversationWorkflow.class,
            WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());

    String result = workflow.converse(new ConversationInput(null, List.of("one", "two", "three")));

    assertEquals("turns=3", result, "the model must see every earlier turn after two restarts");
    assertEquals(3, model.requests().size(), "one model call per turn across the three runs");
    String workflowId = WorkflowStub.fromTyped(workflow).getExecution().getWorkflowId();
    WorkflowExecutionHistory lastRun = client.fetchHistory(workflowId);
    assertTrue(
        lastRun.getEvents().stream()
            .map(HistoryEvent::getEventType)
            .noneMatch(type -> type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_CONTINUED_AS_NEW),
        "the last run completes normally");
    worker.replayWorkflowExecution(lastRun);
  }

  /** The workflow input: the exported session of the previous run and the remaining messages. */
  public record ConversationInput(@Nullable Session session, List<String> messages) {}

  @WorkflowInterface
  public interface ConversationWorkflow {
    @WorkflowMethod
    String converse(ConversationInput input);
  }

  public static class ConversationWorkflowImpl implements ConversationWorkflow {
    @Override
    public String converse(ConversationInput input) {
      LlmAgent agent =
          LlmAgent.builder()
              .name("assistant")
              .model(TemporalModel.create(MODEL))
              .instruction("Count the turns.")
              .build();
      Runner runner = TemporalAdk.runnerBuilder().appName("chat").agent(agent).build();
      if (input.session() != null) {
        Sessions.importSession(runner.sessionService(), input.session());
      }
      List<Event> events =
          TemporalAdk.run(
              runner,
              "user-1",
              "session-1",
              Content.fromParts(Part.fromText(input.messages().get(0))));
      List<String> remaining = input.messages().subList(1, input.messages().size());
      if (remaining.isEmpty()) {
        return TestSupport.lastText(events);
      }
      Session snapshot =
          Sessions.exportSession(runner.sessionService(), "chat", "user-1", "session-1");
      Workflow.continueAsNew(new ConversationInput(snapshot, remaining));
      throw new IllegalStateException("unreachable");
    }
  }
}
