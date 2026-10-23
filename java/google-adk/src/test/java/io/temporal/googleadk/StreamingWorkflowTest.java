package io.temporal.googleadk;

import static io.temporal.googleadk.TestSupport.history;
import static io.temporal.googleadk.TestSupport.scheduledActivityTypes;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.google.adk.agents.LlmAgent;
import com.google.adk.agents.RunConfig;
import com.google.adk.events.Event;
import com.google.adk.runner.Runner;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowOptions;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.googleadk.activity.ModelActivity;
import io.temporal.googleadk.model.TemporalModel;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/** Streamed model calls keep ADK's partial events, served by a single activity. */
class StreamingWorkflowTest {

  private static final String TASK_QUEUE = "google-adk-streaming";
  private static final String MODEL = "scripted";

  private TestWorkflowEnvironment testEnv;
  private WorkflowClient client;
  private Worker worker;

  @BeforeEach
  void setUp() {
    GoogleAdkPlugin plugin =
        GoogleAdkPlugin.newBuilder()
            .model(
                MODEL,
                ScriptedModel.streaming(
                    MODEL,
                    request ->
                        List.of(
                            ScriptedModel.partialText("Hel"),
                            ScriptedModel.partialText("lo"),
                            ScriptedModel.text("Hello"))))
            .build();
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                .build());
    client = testEnv.getWorkflowClient();
    worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(StreamingWorkflowImpl.class);
    testEnv.start();
  }

  @AfterEach
  void tearDown() {
    testEnv.close();
  }

  @Test
  @Timeout(90)
  void sseStreaming_emitsPartialEventsThenTheFinalResponseFromOneActivity() throws Exception {
    StreamingWorkflow workflow =
        client.newWorkflowStub(
            StreamingWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());

    assertEquals(List.of("partial:Hel", "partial:lo", "final:Hello"), workflow.stream("Hi"));

    WorkflowExecutionHistory history = history(client, workflow);
    assertEquals(List.of(ModelActivity.INVOKE_MODEL), scheduledActivityTypes(history));
    worker.replayWorkflowExecution(history);
  }

  @WorkflowInterface
  public interface StreamingWorkflow {
    @WorkflowMethod
    List<String> stream(String message);
  }

  public static class StreamingWorkflowImpl implements StreamingWorkflow {
    @Override
    public List<String> stream(String message) {
      LlmAgent agent =
          LlmAgent.builder()
              .name("streamer")
              .model(TemporalModel.create(MODEL))
              .instruction("Reply.")
              .build();
      Runner runner = TemporalAdk.runnerBuilder().appName("stream").agent(agent).build();
      RunConfig config =
          RunConfig.builder()
              .streamingMode(RunConfig.StreamingMode.SSE)
              .autoCreateSession(true)
              .build();
      List<Event> events =
          TemporalAdk.run(
              runner, "user-1", "session-1", Content.fromParts(Part.fromText(message)), config);
      return events.stream()
          .map(
              event ->
                  (event.partial().orElse(false) ? "partial:" : "final:")
                      + event.content().map(Content::text).orElse(""))
          .toList();
    }
  }
}
