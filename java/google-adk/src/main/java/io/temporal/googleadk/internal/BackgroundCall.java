package io.temporal.googleadk.internal;

import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Runs an RxJava flow on an I/O thread and collects its items on the calling activity thread, so
 * the call can be abandoned when the server cancels the activity.
 *
 * <p>The flow is subscribed exactly once, never cancelled, and its thread is never interrupted:
 * abandoning the call returns the items collected so far and leaves the flow to finish on its own,
 * while whatever it still emits reaches the live subscription and is dropped. That keeps a late
 * failure out of RxJava's undeliverable-error handler, and it keeps the client from retrying: ADK's
 * MCP classes answer an interrupted request by retrying it on a new session, which would run a
 * cancelled tool call a second time, and none of ADK's models abort their HTTP request on an
 * interrupt anyway.
 */
public final class BackgroundCall {

  private static final Logger log = LoggerFactory.getLogger(BackgroundCall.class);

  private BackgroundCall() {}

  /**
   * Subscribes to the flow the supplier returns on an I/O thread and returns its items once it
   * completes, or the items collected so far once {@code abandon} completes, as an unmodifiable
   * list. A failure of the flow is rethrown, a checked one wrapped in a plain {@link
   * RuntimeException} as RxJava's blocking operators do. {@code whenFinished} runs at most once,
   * when the flow terminates, on the thread that terminates it and before the result is delivered,
   * whether or not the call was abandoned first; a runtime exception it throws is logged and
   * changes nothing.
   */
  public static <T> List<T> collect(
      Supplier<Flowable<T>> flow, Flowable<?> abandon, Runnable whenFinished) {
    List<T> items = new CopyOnWriteArrayList<>();
    CompletableFuture<List<T>> result = new CompletableFuture<>();
    AtomicBoolean finished = new AtomicBoolean();
    // Subscribed before the flow, so that an abandonment that already happened completes the
    // result before the I/O thread can start the call.
    abandon.subscribe(ignored -> {}, ignored -> {}, () -> result.complete(snapshot(items)));
    Flowable.defer(
            () -> {
              if (result.isDone()) {
                // Abandoned before the I/O thread picked the call up: do not start it at all.
                return Flowable.<T>empty();
              }
              return flow.get();
            })
        .subscribeOn(Schedulers.io())
        .doOnTerminate(
            () -> {
              if (!finished.compareAndSet(false, true)) {
                return;
              }
              try {
                whenFinished.run();
              } catch (RuntimeException e) {
                log.warn("The completion callback of a background call failed", e);
              }
            })
        .subscribe(
            items::add,
            failure -> {
              if (!result.completeExceptionally(failure)) {
                log.debug("A background call failed after it was abandoned", failure);
              }
            },
            () -> result.complete(snapshot(items)));
    try {
      return result.get();
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      if (cause instanceof RuntimeException) {
        throw (RuntimeException) cause;
      }
      if (cause instanceof Error) {
        throw (Error) cause;
      }
      throw new RuntimeException(cause);
    } catch (InterruptedException e) {
      result.complete(snapshot(items));
      Thread.currentThread().interrupt();
      throw new IllegalStateException("Interrupted while waiting for a background call", e);
    }
  }

  private static <T> List<T> snapshot(List<T> items) {
    return Collections.unmodifiableList(new ArrayList<>(items));
  }
}
