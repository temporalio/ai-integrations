package io.temporal.googleadk;

import static io.temporal.googleadk.TestSupport.history;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.agents.LlmAgent;
import com.google.adk.agents.ParallelAgent;
import com.google.adk.events.Event;
import com.google.adk.runner.Runner;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.temporal.api.enums.v1.EventType;
import io.temporal.api.history.v1.HistoryEvent;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowOptions;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.googleadk.model.TemporalModel;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.googleadk.workflow.WorkflowScheduler;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/** ADK's parallel fan-out runs on workflow threads: siblings wait on their activities together. */
class ParallelAgentWorkflowTest {

  private static final String TASK_QUEUE = "google-adk-parallel";
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
                ScriptedModel.responding(
                    MODEL,
                    request ->
                        ScriptedModel.text(
                            TestSupport.isAgent(request, "worker_a") ? "a done" : "b done")))
            .build();
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                .build());
    client = testEnv.getWorkflowClient();
    worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(FanOutWorkflowImpl.class);
    testEnv.start();
  }

  @AfterEach
  void tearDown() {
    testEnv.close();
  }

  @Test
  @Timeout(90)
  void parallelScheduler_schedulesBothModelCallsBeforeEitherCompletes() throws Exception {
    FanOutWorkflow workflow = newWorkflow();

    assertEquals(List.of("a done", "b done"), workflow.fanOut(true));

    WorkflowExecutionHistory history = history(client, workflow);
    List<EventType> order = activityEventOrder(history);
    assertEquals(
        List.of(
            EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED,
            EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED),
        order.subList(0, 2),
        "both model calls must be scheduled by the same workflow task: " + order);
    worker.replayWorkflowExecution(history);
  }

  @Test
  @Timeout(90)
  void sequentialScheduler_completesTheFirstModelCallBeforeSchedulingTheSecond() throws Exception {
    FanOutWorkflow workflow = newWorkflow();

    assertEquals(List.of("a done", "b done"), workflow.fanOut(false));

    WorkflowExecutionHistory history = history(client, workflow);
    List<EventType> order = activityEventOrder(history);
    int secondScheduled = order.lastIndexOf(EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED);
    int firstCompleted = order.indexOf(EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED);
    assertTrue(
        firstCompleted >= 0 && firstCompleted < secondScheduled,
        "the second model call must wait for the first: " + order);
    worker.replayWorkflowExecution(history);
  }

  private static List<EventType> activityEventOrder(WorkflowExecutionHistory history) {
    return history.getEvents().stream()
        .map(HistoryEvent::getEventType)
        .filter(
            type ->
                type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
                    || type == EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED)
        .collect(Collectors.toList());
  }

  private FanOutWorkflow newWorkflow() {
    return client.newWorkflowStub(
        FanOutWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
  }

  @WorkflowInterface
  public interface FanOutWorkflow {
    @WorkflowMethod
    List<String> fanOut(boolean parallel);
  }

  public static class FanOutWorkflowImpl implements FanOutWorkflow {
    @Override
    public List<String> fanOut(boolean parallel) {
      ParallelAgent root =
          ParallelAgent.builder()
              .name("fan_out")
              .subAgents(agent("worker_a"), agent("worker_b"))
              .build();
      WorkflowScheduler scheduler =
          parallel ? WorkflowScheduler.parallel() : WorkflowScheduler.sequential();
      Runner runner = TemporalAdk.runnerBuilder(scheduler).appName("fan").agent(root).build();
      List<Event> events =
          TemporalAdk.run(runner, "user-1", "session-1", Content.fromParts(Part.fromText("go")));
      return events.stream()
          .filter(event -> event.content().map(Content::text).filter(t -> !t.isEmpty()).isPresent())
          .map(event -> event.content().get().text())
          .sorted()
          .collect(Collectors.toList());
    }

    private static LlmAgent agent(String name) {
      return LlmAgent.builder()
          .name(name)
          .model(TemporalModel.create(MODEL))
          .instruction("Say that you are done.")
          .build();
    }
  }
}
