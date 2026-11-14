package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.google.adk.agents.LlmAgent;
import com.google.adk.events.Event;
import com.google.adk.runner.Runner;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.temporal.activity.ActivityOptions;
import io.temporal.api.history.v1.HistoryEvent;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowOptions;
import io.temporal.client.WorkflowStub;
import io.temporal.common.RetryOptions;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.googleadk.activity.ModelActivity;
import io.temporal.googleadk.model.TemporalModel;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.time.Duration;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/** Runs a vanilla ADK agent inside a workflow through the plugin and replays it. */
class AgentWorkflowTest {

  private static final String TASK_QUEUE = "google-adk-agent";

  private TestWorkflowEnvironment testEnv;
  private WorkflowClient client;
  private Worker worker;

  @BeforeEach
  void setUp() {
    GoogleAdkPlugin plugin =
        GoogleAdkPlugin.newBuilder()
            .model("scripted", ScriptedModel.of("scripted", ScriptedModel.text("Hello from ADK")))
            .build();
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                .build());
    client = testEnv.getWorkflowClient();
    worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(ChatWorkflowImpl.class);
    testEnv.start();
  }

  @AfterEach
  void tearDown() {
    testEnv.close();
  }

  @Test
  @Timeout(90)
  void singleAgentTurn_runsTheModelAsOneActivityAndReplays() throws Exception {
    ChatWorkflow workflow =
        client.newWorkflowStub(
            ChatWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());

    assertEquals("Hello from ADK", workflow.chat("Hi"));

    WorkflowExecutionHistory history =
        client.fetchHistory(WorkflowStub.fromTyped(workflow).getExecution().getWorkflowId());
    long scheduled =
        history.getEvents().stream()
            .filter(HistoryEvent::hasActivityTaskScheduledEventAttributes)
            .filter(
                e ->
                    ModelActivity.INVOKE_MODEL.equals(
                        e.getActivityTaskScheduledEventAttributes().getActivityType().getName()))
            .count();
    assertEquals(1, scheduled, "the model call must be exactly one InvokeModel activity");
    worker.replayWorkflowExecution(history);
  }

  @WorkflowInterface
  public interface ChatWorkflow {
    @WorkflowMethod
    String chat(String message);
  }

  public static class ChatWorkflowImpl implements ChatWorkflow {
    @Override
    public String chat(String message) {
      LlmAgent agent =
          LlmAgent.builder()
              .name("assistant")
              .model(
                  TemporalModel.newBuilder("scripted")
                      .activityOptions(
                          ActivityOptions.newBuilder()
                              .setStartToCloseTimeout(Duration.ofSeconds(30))
                              .setRetryOptions(
                                  RetryOptions.newBuilder().setMaximumAttempts(1).build())
                              .build())
                      .build())
              .instruction("Answer concisely.")
              .build();
      Runner runner = TemporalAdk.runnerBuilder().appName("chat").agent(agent).build();
      List<Event> events =
          TemporalAdk.run(runner, "user-1", "session-1", Content.fromParts(Part.fromText(message)));
      return events.get(events.size() - 1).content().map(Content::text).orElse("");
    }
  }
}
