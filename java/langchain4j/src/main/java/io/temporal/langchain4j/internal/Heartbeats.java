package io.temporal.langchain4j.internal;

import io.temporal.activity.ActivityExecutionContext;
import io.temporal.client.ActivityCompletionException;
import java.time.Duration;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Heartbeats an activity in the background while a model call runs, so a heartbeat timeout set by
 * the caller detects a dead worker. Does nothing when the activity has no heartbeat timeout.
 */
public final class Heartbeats implements AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(Heartbeats.class);
  // A heartbeat may block on an RPC, so a few threads keep one slow call from delaying the rest.
  private static final ScheduledExecutorService EXECUTOR =
      Executors.newScheduledThreadPool(
          2,
          runnable -> {
            Thread thread = new Thread(runnable, "langchain4j-heartbeat");
            thread.setDaemon(true);
            return thread;
          });

  @Nullable private final ScheduledFuture<?> ticks;

  private Heartbeats(@Nullable ScheduledFuture<?> ticks) {
    this.ticks = ticks;
  }

  /**
   * Starts heartbeating at a third of the activity's heartbeat timeout, if it has one. The ticks
   * stop once the server reports the activity as cancelled or timed out.
   */
  public static Heartbeats start(ActivityExecutionContext context) {
    Duration timeout = context.getInfo().getHeartbeatTimeout();
    if (timeout == null || timeout.isZero() || timeout.isNegative()) {
      return new Heartbeats(null);
    }
    long periodMillis = Math.max(1, timeout.toMillis() / 3);
    ScheduledFuture<?>[] self = new ScheduledFuture<?>[1];
    ScheduledFuture<?> ticks =
        EXECUTOR.scheduleAtFixedRate(
            () -> {
              try {
                context.heartbeat(null);
              } catch (ActivityCompletionException e) {
                synchronized (self) {
                  if (self[0] != null) {
                    self[0].cancel(false);
                  }
                }
              } catch (RuntimeException e) {
                // A failed heartbeat is retried at the next tick; the call itself is unaffected.
                log.debug("Heartbeat failed", e);
              }
            },
            periodMillis,
            periodMillis,
            TimeUnit.MILLISECONDS);
    synchronized (self) {
      self[0] = ticks;
    }
    return new Heartbeats(ticks);
  }

  @Override
  public void close() {
    if (ticks != null) {
      ticks.cancel(false);
    }
  }
}
