package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.agents.LlmAgent;
import com.google.adk.events.Event;
import com.google.adk.runner.Runner;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.temporal.activity.ActivityInterface;
import io.temporal.activity.ActivityMethod;
import io.temporal.activity.ActivityOptions;
import io.temporal.api.history.v1.HistoryEvent;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowOptions;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.googleadk.model.TemporalModel;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.worker.WorkerFactory;
import io.temporal.workflow.QueryMethod;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Event ids and timestamps come from the workflow, not the wall clock or a random source: the
 * workflow runs 400 days ahead of real time, and a replay forced by a query on the closed execution
 * mints exactly the ids the original execution recorded.
 */
class DeterminismWorkflowTest {

  private static final String TASK_QUEUE = "google-adk-determinism";
  private static final String MODEL = "scripted";

  private TestWorkflowEnvironment testEnv;
  private WorkflowClient client;

  @BeforeEach
  void setUp() {
    GoogleAdkPlugin plugin =
        GoogleAdkPlugin.newBuilder()
            .model(MODEL, ScriptedModel.of(MODEL, ScriptedModel.text("noted")))
            .build();
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                .build());
    client = testEnv.getWorkflowClient();
    Worker worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(RecordingWorkflowImpl.class);
    worker.registerActivitiesImplementations(new RecorderActivitiesImpl());
    testEnv.start();
  }

  @AfterEach
  void tearDown() {
    testEnv.close();
  }

  @Test
  @Timeout(90)
  void idsAndTimestamps_comeFromTheWorkflowAndReplayIdentically() {
    long wallClockStart = System.currentTimeMillis();
    testEnv.sleep(Duration.ofDays(400));
    RecordingWorkflow workflow =
        client.newWorkflowStub(
            RecordingWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());

    List<String> recorded = workflow.run();

    assertTrue(recorded.size() >= 2, recorded.toString());
    for (String entry : recorded) {
      long timestamp = Long.parseLong(entry.substring(entry.indexOf('|') + 1));
      assertTrue(
          timestamp > wallClockStart + Duration.ofDays(399).toMillis(),
          "timestamps follow the workflow clock, not the wall clock: " + entry);
    }
    WorkflowExecutionHistory history =
        client.fetchHistory(
            io.temporal.client.WorkflowStub.fromTyped(workflow).getExecution().getWorkflowId());
    HistoryEvent scheduled =
        history.getEvents().stream()
            .filter(HistoryEvent::hasActivityTaskScheduledEventAttributes)
            .filter(
                e ->
                    e.getActivityTaskScheduledEventAttributes()
                        .getActivityType()
                        .getName()
                        .equals("Record"))
            .findFirst()
            .orElseThrow();
    @SuppressWarnings("unchecked")
    List<String> input =
        client
            .getOptions()
            .getDataConverter()
            .fromPayloads(
                0,
                java.util.Optional.of(
                    scheduled.getActivityTaskScheduledEventAttributes().getInput()),
                List.class,
                List.class);
    assertEquals(input, recorded, "the result is what the first execution recorded");

    // A fresh worker has no cached execution, so the query replays the history from scratch.
    testEnv.getWorkerFactory().shutdown();
    testEnv.getWorkerFactory().awaitTermination(30, TimeUnit.SECONDS);
    WorkerFactory replayFactory = WorkerFactory.newInstance(client);
    Worker replayWorker = replayFactory.newWorker(TASK_QUEUE);
    replayWorker.registerWorkflowImplementationTypes(RecordingWorkflowImpl.class);
    replayFactory.start();
    try {
      assertEquals(recorded, workflow.recorded(), "a replay mints the same ids and timestamps");
    } finally {
      replayFactory.shutdownNow();
    }
  }

  @WorkflowInterface
  public interface RecordingWorkflow {
    @WorkflowMethod
    List<String> run();

    @QueryMethod
    List<String> recorded();
  }

  @ActivityInterface
  public interface RecorderActivities {
    @ActivityMethod
    void record(List<String> entries);
  }

  public static class RecorderActivitiesImpl implements RecorderActivities {
    @Override
    public void record(List<String> entries) {}
  }

  public static class RecordingWorkflowImpl implements RecordingWorkflow {
    private final List<String> recorded = new ArrayList<>();

    @Override
    public List<String> run() {
      LlmAgent agent =
          LlmAgent.builder().name("assistant").model(TemporalModel.create(MODEL)).build();
      Runner runner = TemporalAdk.runnerBuilder().appName("determinism").agent(agent).build();
      List<Event> events =
          TemporalAdk.run(runner, "user-1", "session-1", Content.fromParts(Part.fromText("hi")));
      for (Event event : events) {
        recorded.add(event.id() + "|" + event.timestamp());
      }
      recorded.add(
          TemporalAdk.uuidProvider().newUuid()
              + "|"
              + TemporalAdk.instantSource().instant().toEpochMilli());
      RecorderActivities recorder =
          Workflow.newActivityStub(
              RecorderActivities.class,
              ActivityOptions.newBuilder().setStartToCloseTimeout(Duration.ofSeconds(10)).build());
      recorder.record(recorded);
      return recorded;
    }

    @Override
    public List<String> recorded() {
      return recorded;
    }
  }
}
