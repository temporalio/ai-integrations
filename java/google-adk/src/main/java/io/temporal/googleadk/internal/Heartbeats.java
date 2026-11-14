package io.temporal.googleadk.internal;

import io.temporal.activity.ActivityExecutionContext;
import io.temporal.client.ActivityCompletionException;
import java.time.Duration;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Heartbeats an activity in the background while a long model or tool call runs, so a heartbeat
 * timeout configured by the caller detects a dead worker and a cancellation reaches the call. Does
 * nothing when the activity has no heartbeat timeout.
 */
public final class Heartbeats implements AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(Heartbeats.class);
  // A heartbeat may block on an RPC, so a few threads keep one slow call from delaying the rest.
  private static final ScheduledExecutorService EXECUTOR =
      Executors.newScheduledThreadPool(
          2,
          runnable -> {
            Thread thread = new Thread(runnable, "googleadk-heartbeat");
            thread.setDaemon(true);
            return thread;
          });

  @Nullable private final ScheduledFuture<?> ticks;
  private final AtomicReference<ActivityCompletionException> completion;

  private Heartbeats(
      @Nullable ScheduledFuture<?> ticks, AtomicReference<ActivityCompletionException> completion) {
    this.ticks = ticks;
    this.completion = completion;
  }

  /**
   * Starts heartbeating at a third of the activity's heartbeat timeout, if it has one. When the
   * server reports the activity as cancelled or timed out, the ticks stop and {@code onCompletion}
   * runs once so the in-flight call can be abandoned.
   */
  public static Heartbeats start(ActivityExecutionContext context, Runnable onCompletion) {
    AtomicReference<ActivityCompletionException> completion = new AtomicReference<>();
    Duration timeout = context.getInfo().getHeartbeatTimeout();
    if (timeout == null || timeout.isZero() || timeout.isNegative()) {
      return new Heartbeats(null, completion);
    }
    long periodMillis = Math.max(1, timeout.toMillis() / 3);
    AtomicReference<ScheduledFuture<?>> handle = new AtomicReference<>();
    ScheduledFuture<?> ticks =
        EXECUTOR.scheduleAtFixedRate(
            () -> {
              try {
                context.heartbeat(null);
              } catch (ActivityCompletionException e) {
                if (completion.compareAndSet(null, e)) {
                  ScheduledFuture<?> self = handle.get();
                  if (self != null) {
                    self.cancel(false);
                  }
                  onCompletion.run();
                }
              } catch (RuntimeException e) {
                // A failed heartbeat is retried at the next tick; the call itself is unaffected.
                log.debug("Heartbeat failed", e);
              }
            },
            periodMillis,
            periodMillis,
            TimeUnit.MILLISECONDS);
    handle.set(ticks);
    return new Heartbeats(ticks, completion);
  }

  /** Returns the cancellation or timeout the server reported during the call, if any. */
  @Nullable
  public ActivityCompletionException completion() {
    return completion.get();
  }

  @Override
  public void close() {
    if (ticks != null) {
      ticks.cancel(false);
    }
  }
}
