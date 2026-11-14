package io.temporal.googleadk;

import static io.temporal.googleadk.TestSupport.history;
import static io.temporal.googleadk.TestSupport.timerDurations;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.sessions.BaseSessionService;
import com.google.adk.sessions.Session;
import io.reactivex.rxjava3.core.Flowable;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowFailedException;
import io.temporal.client.WorkflowOptions;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.workflow.WorkflowScheduler;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/** The workflow-side helpers draw from the workflow and replay deterministically. */
class TemporalAdkWorkflowTest {

  private static final String TASK_QUEUE = "google-adk-helpers";

  private TestWorkflowEnvironment testEnv;
  private WorkflowClient client;
  private Worker worker;

  @BeforeEach
  void setUp() {
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder()
                        .setPlugins(GoogleAdkPlugin.newDefaultInstance())
                        .build())
                .build());
    client = testEnv.getWorkflowClient();
    worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(
        HelpersWorkflowImpl.class, AwaitFailureWorkflowImpl.class, IntervalWorkflowImpl.class);
    testEnv.start();
  }

  @AfterEach
  void tearDown() {
    testEnv.close();
  }

  @Test
  @Timeout(90)
  void providersAndSchedulers_useWorkflowTimeAndReplay() throws Exception {
    HelpersWorkflow workflow =
        client.newWorkflowStub(
            HelpersWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());

    List<String> report = workflow.exercise();

    assertEquals("clock-matches-workflow", report.get(0));
    assertEquals("ids-distinct", report.get(1));
    assertEquals("session-rejects-duplicates", report.get(2));
    assertEquals("auto-create-session", report.get(3));
    assertEquals("delayed-parallel:3,1,2", report.get(4));
    assertEquals("delayed-sequential:3,1,2", report.get(5));
    WorkflowExecutionHistory history = history(client, workflow);
    assertTrue(
        history.getEvents().stream().anyMatch(e -> e.hasTimerStartedEventAttributes()),
        "a delayed task waits on a workflow timer");
    worker.replayWorkflowExecution(history);
  }

  /**
   * RxJava's periodic tasks schedule each tick from the worker's clock, so a worker that reported
   * wall-clock time while the workflow skipped ahead stretched every period. Each scheduler emits
   * at one, two, and three seconds of workflow time; a tick trails its second only by the real time
   * the server spends on the workflow task, which the bound below allows for generously, and the
   * timers prove the period.
   */
  @Test
  @Timeout(90)
  void interval_ticksOncePerPeriodOfWorkflowTime() throws Exception {
    IntervalWorkflow workflow =
        client.newWorkflowStub(
            IntervalWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());

    List<Long> offsets = workflow.ticks();

    assertEquals(6, offsets.size(), "three ticks per scheduler: " + offsets);
    for (int i = 0; i < offsets.size(); i++) {
      long second = (i % 3 + 1) * 1000;
      assertTrue(
          offsets.get(i) >= second && offsets.get(i) < second + 10_000,
          "tick " + i + " must follow " + second + " ms of workflow time closely: " + offsets);
    }
    WorkflowExecutionHistory history = history(client, workflow);
    List<Duration> timers = timerDurations(history);
    assertEquals(6, timers.size(), "every tick waits on a workflow timer: " + timers);
    assertTrue(
        timers.stream().allMatch(timer -> timer.compareTo(Duration.ofSeconds(1)) <= 0),
        "a tick's timer never exceeds the period: " + timers);
    worker.replayWorkflowExecution(history);
  }

  @Test
  @Timeout(90)
  void await_rethrowsTheFlowError() {
    AwaitFailureWorkflow workflow =
        client.newWorkflowStub(
            AwaitFailureWorkflow.class,
            WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());

    WorkflowFailedException failed =
        assertThrows(WorkflowFailedException.class, workflow::awaitFailure);

    assertInstanceOf(ApplicationFailure.class, failed.getCause());
    assertEquals("FlowError", ((ApplicationFailure) failed.getCause()).getType());
  }

  @WorkflowInterface
  public interface HelpersWorkflow {
    @WorkflowMethod
    List<String> exercise();
  }

  @WorkflowInterface
  public interface AwaitFailureWorkflow {
    @WorkflowMethod
    void awaitFailure();
  }

  @WorkflowInterface
  public interface IntervalWorkflow {
    /** Returns the workflow time of each tick, relative to the start, per scheduler. */
    @WorkflowMethod
    List<Long> ticks();
  }

  public static class IntervalWorkflowImpl implements IntervalWorkflow {
    @Override
    public List<Long> ticks() {
      List<Long> offsets = new ArrayList<>(ticks(WorkflowScheduler.parallel()));
      offsets.addAll(ticks(WorkflowScheduler.sequential()));
      return offsets;
    }

    private static List<Long> ticks(WorkflowScheduler scheduler) {
      long start = Workflow.currentTimeMillis();
      return TemporalAdk.await(
          Flowable.interval(1, TimeUnit.SECONDS, scheduler)
              .take(3)
              .map(tick -> Workflow.currentTimeMillis() - start));
    }
  }

  public static class AwaitFailureWorkflowImpl implements AwaitFailureWorkflow {
    @Override
    public void awaitFailure() {
      TemporalAdk.await(
          Flowable.error(ApplicationFailure.newNonRetryableFailure("boom", "FlowError")));
    }
  }

  public static class HelpersWorkflowImpl implements HelpersWorkflow {
    @Override
    public List<String> exercise() {
      List<String> report = new ArrayList<>();
      long now = Workflow.currentTimeMillis();
      report.add(
          TemporalAdk.instantSource().instant().toEpochMilli() == now
              ? "clock-matches-workflow"
              : "clock-differs");
      HashSet<String> ids = new HashSet<>();
      for (int i = 0; i < 5; i++) {
        ids.add(TemporalAdk.uuidProvider().newUuid());
      }
      report.add(ids.size() == 5 ? "ids-distinct" : "ids-collide");
      BaseSessionService sessions = TemporalAdk.newSessionService();
      Session first = sessions.createSession("app", "u", Map.of(), "s1").blockingGet();
      boolean rejected;
      try {
        sessions.createSession("app", "u", Map.of(), "s1").blockingGet();
        rejected = false;
      } catch (RuntimeException e) {
        rejected = true;
      }
      report.add(rejected && first.id().equals("s1") ? "session-rejects-duplicates" : "overwrote");
      report.add(
          TemporalAdk.defaultRunConfig().autoCreateSession()
              ? "auto-create-session"
              : "no-auto-create");
      report.add("delayed-parallel:" + delayed(WorkflowScheduler.parallel()));
      report.add("delayed-sequential:" + delayed(WorkflowScheduler.sequential()));
      return report;
    }

    private static String delayed(WorkflowScheduler scheduler) {
      // Equal delays keep the submission order; a shorter delay submitted later runs first.
      List<Integer> items =
          TemporalAdk.await(
              Flowable.merge(
                  Flowable.just(1, 2).delay(2, TimeUnit.SECONDS, scheduler),
                  Flowable.just(3).delay(1, TimeUnit.SECONDS, scheduler)));
      return items.get(0) + "," + items.get(1) + "," + items.get(2);
    }
  }
}
