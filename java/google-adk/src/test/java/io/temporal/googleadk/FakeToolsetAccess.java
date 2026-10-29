package io.temporal.googleadk;

import com.google.adk.tools.BaseToolset;

/** Exposes the package-private {@link FakeToolset} to tests in other packages. */
public final class FakeToolsetAccess {

  private FakeToolsetAccess() {}

  public static BaseToolset newToolset() {
    return new FakeToolset();
  }

  public static void reset() {
    FakeToolset.reset();
  }

  public static int created() {
    return FakeToolset.created.get();
  }

  public static int calls() {
    return FakeToolset.calls.get();
  }

  public static int closed() {
    return FakeToolset.closed.get();
  }

  public static void awaitWaiting() throws InterruptedException {
    if (!FakeToolset.waiting.await(30, java.util.concurrent.TimeUnit.SECONDS)) {
      throw new IllegalStateException("the wait tool never started");
    }
  }

  public static void openGate() {
    FakeToolset.gate.countDown();
  }
}
