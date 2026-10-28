package io.temporal.googleadk.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.functions.Consumer;
import io.reactivex.rxjava3.plugins.RxJavaPlugins;
import io.reactivex.rxjava3.processors.PublishProcessor;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class BackgroundCallTest {

  private final List<Throwable> undeliverable = new CopyOnWriteArrayList<>();
  private final AtomicInteger finished = new AtomicInteger();
  private final CountDownLatch finishedOnce = new CountDownLatch(1);
  private final CountDownLatch release = new CountDownLatch(1);
  private final CountDownLatch started = new CountDownLatch(1);
  private final PublishProcessor<Object> abandon = PublishProcessor.create();
  private Consumer<? super Throwable> previousHandler;

  @BeforeEach
  void captureUndeliverableErrors() {
    previousHandler = RxJavaPlugins.getErrorHandler();
    RxJavaPlugins.setErrorHandler(undeliverable::add);
  }

  @AfterEach
  void restore() {
    release.countDown();
    RxJavaPlugins.setErrorHandler(previousHandler);
  }

  private void whenFinished() {
    finished.incrementAndGet();
    finishedOnce.countDown();
  }

  /** Emits {@code first}, then blocks until released, then emits {@code late} or fails. */
  private Flowable<String> blockingFlow(boolean failAfterRelease) {
    return Flowable.concat(
        Flowable.just("first"),
        Flowable.fromCallable(
            () -> {
              started.countDown();
              if (!release.await(30, TimeUnit.SECONDS)) {
                throw new IllegalStateException("never released");
              }
              if (failAfterRelease) {
                throw new IOException("late failure");
              }
              return "late";
            }));
  }

  private void abandonOnceStarted() {
    Thread abandoner =
        new Thread(
            () -> {
              try {
                started.await();
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
              }
              abandon.onComplete();
            });
    abandoner.setDaemon(true);
    abandoner.start();
  }

  @Test
  void abandon_returnsTheItemsSoFar_andLeavesTheCallToFinishUninterrupted() throws Exception {
    abandonOnceStarted();

    List<String> items =
        BackgroundCall.collect(() -> blockingFlow(false), abandon, this::whenFinished);

    assertEquals(List.of("first"), items);
    assertThrows(UnsupportedOperationException.class, () -> items.add("mine"));
    assertEquals(0, finished.get(), "the call is still running when the activity thread returns");
    release.countDown();
    assertTrue(finishedOnce.await(10, TimeUnit.SECONDS), "the call finishes on its own");
    Thread.sleep(100);
    assertEquals(1, finished.get());
    assertEquals(List.of("first"), items, "a late item does not reach the returned list");
    assertTrue(undeliverable.isEmpty(), "nothing reaches RxJava's error handler: " + undeliverable);
  }

  @Test
  void lateFailure_afterAbandon_isDropped() throws Exception {
    abandonOnceStarted();

    List<String> items =
        BackgroundCall.collect(() -> blockingFlow(true), abandon, this::whenFinished);

    assertEquals(List.of("first"), items);
    release.countDown();
    assertTrue(finishedOnce.await(10, TimeUnit.SECONDS));
    Thread.sleep(100);
    assertEquals(1, finished.get());
    assertTrue(undeliverable.isEmpty(), "the late failure is dropped: " + undeliverable);
  }

  /** The accounting a caller does in {@code whenFinished} is settled before the caller resumes. */
  @Test
  void whenFinished_runsBeforeTheResultIsDelivered() {
    Runnable slowFinish =
        () -> {
          try {
            Thread.sleep(300);
          } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
          }
          finished.incrementAndGet();
        };

    List<String> items = BackgroundCall.collect(() -> Flowable.just("only"), abandon, slowFinish);
    assertEquals(List.of("only"), items);
    assertEquals(1, finished.get(), "settled on completion");

    assertThrows(
        IllegalStateException.class,
        () ->
            BackgroundCall.collect(
                () -> Flowable.error(new IllegalStateException("boom")), abandon, slowFinish));
    assertEquals(2, finished.get(), "settled on failure");
  }

  /** An abandonment that already happened keeps the call from starting; the accounting settles. */
  @Test
  void abandonedBeforeTheCall_neverStartsTheFlow() throws Exception {
    abandon.onComplete();
    AtomicInteger starts = new AtomicInteger();

    for (int i = 0; i < 2000; i++) {
      List<String> items =
          BackgroundCall.collect(
              () -> {
                starts.incrementAndGet();
                return Flowable.just("started");
              },
              abandon,
              this::whenFinished);
      assertEquals(List.of(), items);
    }

    assertEquals(0, starts.get(), "no flow starts for a call abandoned before it began");
    for (int i = 0; i < 200 && finished.get() < 2000; i++) {
      Thread.sleep(
          50); // The empty flow settles its accounting on the I/O thread, after the return.
    }
    assertEquals(2000, finished.get(), "the accounting still settles for each call");
  }

  @Test
  void checkedFailure_isWrappedInAPlainRuntimeException() {
    IOException failure = new IOException("reset");

    RuntimeException thrown =
        assertThrows(
            RuntimeException.class,
            () ->
                BackgroundCall.collect(() -> Flowable.error(failure), abandon, this::whenFinished));

    assertSame(RuntimeException.class, thrown.getClass());
    assertSame(failure, thrown.getCause());
    assertEquals(1, finished.get());
  }

  @Test
  void whenFinishedThrowing_changesNothing() {
    List<String> items =
        BackgroundCall.collect(
            () -> Flowable.just("only"),
            abandon,
            () -> {
              throw new IllegalStateException("callback failure");
            });

    assertEquals(List.of("only"), items);
    assertTrue(
        undeliverable.isEmpty(), "the callback's failure is logged, not raised: " + undeliverable);
  }

  @Test
  void interruptedCaller_abandonsTheCall() throws Exception {
    Thread caller = Thread.currentThread();
    Thread interrupter =
        new Thread(
            () -> {
              try {
                started.await();
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
              }
              caller.interrupt();
            });
    interrupter.setDaemon(true);
    interrupter.start();

    assertThrows(
        IllegalStateException.class,
        () -> BackgroundCall.collect(() -> blockingFlow(false), abandon, this::whenFinished));

    assertTrue(Thread.interrupted(), "the interrupt flag is restored for the caller");
    assertFalse(abandon.hasComplete());
    release.countDown();
    assertTrue(finishedOnce.await(10, TimeUnit.SECONDS), "the call still finishes on its own");
    assertTrue(undeliverable.isEmpty(), "" + undeliverable);
  }
}
