package io.temporal.googleadk;

import com.google.adk.agents.RunConfig;
import com.google.adk.events.Event;
import com.google.adk.platform.UuidProvider;
import com.google.adk.runner.Runner;
import com.google.adk.sessions.BaseSessionService;
import com.google.adk.sessions.InMemorySessionService;
import com.google.genai.types.Content;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.disposables.Disposable;
import io.temporal.googleadk.workflow.WorkflowScheduler;
import io.temporal.workflow.Workflow;
import java.time.Instant;
import java.time.InstantSource;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Workflow-side entry points for running ADK agents inside a Temporal workflow.
 *
 * <p>{@link #runnerBuilder()} returns ADK's own {@code Runner.Builder} with the workflow's clock,
 * id source, scheduler, and an in-workflow session service installed, so every id and timestamp the
 * runner mints replays identically. {@link #run} drives a run to completion without blocking a
 * workflow thread on an RxJava primitive.
 *
 * <pre>{@code
 * Runner runner = TemporalAdk.runnerBuilder().appName("hello").agent(root).build();
 * List<Event> events =
 *     TemporalAdk.run(runner, "user-1", "session-1", Content.fromParts(Part.fromText("Hi")));
 * }</pre>
 */
public final class TemporalAdk {

  private TemporalAdk() {}

  /** Returns a clock that reads the workflow time. */
  public static InstantSource instantSource() {
    return () -> Instant.ofEpochMilli(Workflow.currentTimeMillis());
  }

  /**
   * Returns a new id source seeded once from the workflow ({@link Workflow#newRandom()}). Each id
   * is a version 4 UUID drawn from that generator, so the number of ids ADK mints never shifts the
   * ids Temporal assigns to activities and timers, which share a counter with {@link
   * Workflow#randomUUID()}. The source must be created on a workflow thread.
   */
  public static UuidProvider uuidProvider() {
    Random random = Workflow.newRandom();
    return () -> {
      long high = (random.nextLong() & 0xFFFFFFFFFFFF0FFFL) | 0x0000000000004000L;
      long low = (random.nextLong() & 0x3FFFFFFFFFFFFFFFL) | 0x8000000000000000L;
      return new UUID(high, low).toString();
    };
  }

  /**
   * Returns an in-memory session service that stamps sessions with the workflow's clock and ids.
   * Temporal history is the durable store; see the README for continue-as-new.
   */
  public static BaseSessionService newSessionService() {
    return newSessionService(uuidProvider());
  }

  private static BaseSessionService newSessionService(UuidProvider uuidProvider) {
    return new InMemorySessionService(
        InMemorySessionService.DuplicateSessionIdBehavior.REJECT, instantSource(), uuidProvider);
  }

  /**
   * Returns a {@code Runner.Builder} configured for workflows, with parallel fan-out on workflow
   * threads. Set {@code appName} and {@code agent} and build it as usual.
   */
  public static Runner.Builder runnerBuilder() {
    return runnerBuilder(WorkflowScheduler.parallel());
  }

  /** Returns a {@code Runner.Builder} configured for workflows with the given scheduler. */
  public static Runner.Builder runnerBuilder(WorkflowScheduler scheduler) {
    Objects.requireNonNull(scheduler, "scheduler");
    UuidProvider ids = uuidProvider();
    return Runner.builder()
        .instantSource(instantSource())
        .uuidProvider(ids)
        .scheduler(scheduler)
        .sessionService(newSessionService(ids));
  }

  /**
   * Runs one turn with {@link #defaultRunConfig()} and returns the events the runner emitted. The
   * session is created on first use.
   */
  public static List<Event> run(Runner runner, String userId, String sessionId, Content message) {
    return run(runner, userId, sessionId, message, defaultRunConfig());
  }

  /**
   * Runs one turn with the given run configuration and returns the events emitted. Unless the
   * configuration sets {@code autoCreateSession}, the session must already exist.
   */
  public static List<Event> run(
      Runner runner, String userId, String sessionId, Content message, RunConfig runConfig) {
    return await(runner.runAsync(userId, sessionId, message, runConfig));
  }

  /** Returns ADK's default run configuration with {@code autoCreateSession} enabled. */
  public static RunConfig defaultRunConfig() {
    return RunConfig.builder().autoCreateSession(true).build();
  }

  /**
   * Subscribes to the flow on the current workflow thread and waits for it to complete with {@link
   * Workflow#await}, returning every item in order. Errors are rethrown.
   */
  public static <T> List<T> await(Flowable<T> flowable) {
    List<T> items = new ArrayList<>();
    AtomicReference<Throwable> failure = new AtomicReference<>();
    AtomicBoolean done = new AtomicBoolean();
    Disposable subscription =
        flowable.subscribe(
            items::add,
            error -> {
              failure.set(error);
              done.set(true);
            },
            () -> done.set(true));
    try {
      Workflow.await(done::get);
    } finally {
      subscription.dispose();
    }
    Throwable error = failure.get();
    if (error instanceof RuntimeException) {
      throw (RuntimeException) error;
    }
    if (error instanceof Error) {
      throw (Error) error;
    }
    if (error != null) {
      throw new IllegalStateException("ADK run failed", error);
    }
    return items;
  }
}
