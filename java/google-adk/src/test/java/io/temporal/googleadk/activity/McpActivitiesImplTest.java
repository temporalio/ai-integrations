package io.temporal.googleadk.activity;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.agents.ReadonlyContext;
import com.google.adk.tools.BaseTool;
import com.google.adk.tools.BaseToolset;
import com.google.genai.types.FunctionDeclaration;
import io.reactivex.rxjava3.core.Flowable;
import io.temporal.activity.ActivityOptions;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.failure.ActivityFailure;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.FakeToolsetAccess;
import io.temporal.googleadk.converter.GoogleAdkDataConverter;
import io.temporal.testing.TestActivityEnvironment;
import io.temporal.testing.TestEnvironmentOptions;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** The MCP activities serve registered toolsets and fail with typed errors otherwise. */
class McpActivitiesImplTest {

  private TestActivityEnvironment env;
  private McpActivitiesImpl impl;

  /** Counted down when a blocking toolset starts blocking; it returns once {@link #gate} opens. */
  private final CountDownLatch blocked = new CountDownLatch(1);

  private final CountDownLatch gate = new CountDownLatch(1);

  /** Counted down when the blocking-listing toolset is closed. */
  private final CountDownLatch blockingClosed = new CountDownLatch(1);

  @BeforeEach
  void setUp() {
    FakeToolsetAccess.reset();
    env =
        TestActivityEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder()
                        .setDataConverter(GoogleAdkDataConverter.newDefaultInstance())
                        .build())
                .build());
    impl =
        new McpActivitiesImpl(
            Map.of(
                "fake",
                FakeToolsetAccess::newToolset,
                "blocking-factory",
                () -> {
                  block();
                  return FakeToolsetAccess.newToolset();
                },
                "blocking-listing",
                BlockingListingToolset::new,
                "broken",
                () -> {
                  throw new IllegalStateException("cannot start the server");
                }));
    env.registerActivitiesImplementations(impl);
  }

  @AfterEach
  void tearDown() {
    gate.countDown();
    env.close();
  }

  @Test
  void listMcpTools_returnsFullDeclarations() {
    McpActivities activities = env.newActivityStub(McpActivities.class);

    ListMcpToolsOutput output = activities.listMcpTools(new ListMcpToolsInput("fake"));

    assertEquals(2, output.tools().size());
    FunctionDeclaration echo = output.tools().get(0);
    assertEquals("echo", echo.name().orElseThrow());
    assertEquals("Echoes the given text.", echo.description().orElseThrow());
    assertTrue(echo.parameters().isPresent(), "the parameter schema travels with the tool");
    assertEquals(1, FakeToolsetAccess.created());
  }

  @Test
  void callMcpTool_runsTheToolOnce() {
    McpActivities activities = env.newActivityStub(McpActivities.class);

    CallMcpToolOutput output =
        activities.callMcpTool(new CallMcpToolInput("fake", "echo", Map.of("text", "hi")));

    assertEquals(Map.of("echo", "hi"), output.result());
    assertEquals(1, FakeToolsetAccess.calls());
  }

  @Test
  void unknownToolset_failsNonRetryably() {
    McpActivities activities = env.newActivityStub(McpActivities.class);

    ApplicationFailure failure =
        applicationFailure(() -> activities.listMcpTools(new ListMcpToolsInput("missing")));

    assertEquals("googleadk.McpError", failure.getType());
    assertTrue(failure.isNonRetryable());
    assertTrue(failure.getOriginalMessage().contains("registered: ["));
    assertTrue(failure.getOriginalMessage().contains("fake"));
  }

  @Test
  void unknownTool_failsNonRetryably() {
    McpActivities activities = env.newActivityStub(McpActivities.class);

    ApplicationFailure failure =
        applicationFailure(
            () -> activities.callMcpTool(new CallMcpToolInput("fake", "nope", Map.of())));

    assertEquals("googleadk.McpError", failure.getType());
    assertTrue(failure.isNonRetryable());
  }

  @Test
  void closedToolsets_rejectFurtherCallsRetryably() {
    McpActivities activities = env.newActivityStub(McpActivities.class);
    activities.listMcpTools(new ListMcpToolsInput("fake"));

    impl.close();

    assertEquals(1, FakeToolsetAccess.closed());
    ApplicationFailure failure =
        applicationFailure(() -> activities.listMcpTools(new ListMcpToolsInput("fake")));
    assertEquals("googleadk.McpError", failure.getType());
    assertFalse(failure.isNonRetryable(), "another worker can serve the call");
    assertTrue(failure.getOriginalMessage().contains("shutting down"));
  }

  @Test
  void close_waitsForTheInFlightCall() throws Exception {
    McpActivities activities = env.newActivityStub(McpActivities.class);
    AtomicReference<CallMcpToolOutput> output = new AtomicReference<>();
    Thread call =
        new Thread(
            () ->
                output.set(activities.callMcpTool(new CallMcpToolInput("fake", "wait", Map.of()))));
    call.start();
    FakeToolsetAccess.awaitWaiting();

    impl.close();
    assertEquals(0, FakeToolsetAccess.closed(), "the toolset stays open while a call runs");

    FakeToolsetAccess.openGate();
    call.join(30_000);
    assertEquals(Map.of("waited", true), output.get().result());
    assertEquals(1, FakeToolsetAccess.closed(), "released once the last call completed");
  }

  /**
   * Whether the toolset blocks while it is created or while it lists its tools, cancelling the
   * activity releases the activity thread while the toolset is still blocked.
   */
  @ParameterizedTest(name = "{0}")
  @ValueSource(strings = {"blocking-factory", "blocking-listing"})
  void listMcpTools_abandonsTheListingWhenTheActivityIsCancelled(String toolset) throws Exception {
    McpActivities activities =
        env.newActivityStub(
            McpActivities.class,
            ActivityOptions.newBuilder()
                .setStartToCloseTimeout(Duration.ofSeconds(30))
                .setHeartbeatTimeout(Duration.ofMillis(300))
                .build());
    AtomicReference<Throwable> outcome = new AtomicReference<>();
    Thread call =
        new Thread(
            () -> {
              try {
                activities.listMcpTools(new ListMcpToolsInput(toolset));
              } catch (Throwable t) {
                outcome.set(t);
              }
            });
    call.start();
    assertTrue(blocked.await(30, TimeUnit.SECONDS), "the toolset must start blocking");

    env.requestCancelActivity();
    call.join(10_000);

    assertFalse(call.isAlive(), "the activity returns although the toolset is still blocked");
    assertNotNull(outcome.get(), "cancellation surfaces as a failure");
  }

  /** A server that never answers holds back only its own toolset's release, not the others'. */
  @Test
  void close_releasesAnIdleToolsetWhileAnotherIsStillBlocked() throws Exception {
    McpActivities activities =
        env.newActivityStub(
            McpActivities.class,
            ActivityOptions.newBuilder()
                .setStartToCloseTimeout(Duration.ofSeconds(30))
                .setHeartbeatTimeout(Duration.ofMillis(300))
                .build());
    activities.listMcpTools(new ListMcpToolsInput("fake"));
    Thread call =
        new Thread(
            () -> {
              try {
                activities.listMcpTools(new ListMcpToolsInput("blocking-listing"));
              } catch (Throwable ignored) {
                // Cancelled; the listing itself is still blocked.
              }
            });
    call.start();
    assertTrue(blocked.await(30, TimeUnit.SECONDS), "the toolset must start blocking");
    env.requestCancelActivity();
    call.join(10_000);
    assertFalse(call.isAlive(), "the activity returns although the listing is still blocked");

    impl.close();

    assertEquals(1, FakeToolsetAccess.closed(), "the idle toolset is released at once");
    assertEquals(1, blockingClosed.getCount(), "the blocked toolset stays open while it lists");
    gate.countDown();
    assertTrue(
        blockingClosed.await(10, TimeUnit.SECONDS), "the blocked toolset is released when done");
  }

  @Test
  void callMcpTool_acceptsNullArgumentValues() {
    McpActivities activities = env.newActivityStub(McpActivities.class);
    Map<String, Object> args = new HashMap<>();
    args.put("text", null);

    CallMcpToolOutput output = activities.callMcpTool(new CallMcpToolInput("fake", "echo", args));

    assertEquals(Map.of("echo", "null"), output.result());
  }

  @Test
  void reopen_servesTheNextWorkerFactoryWithFreshToolsets() {
    McpActivities activities = env.newActivityStub(McpActivities.class);
    activities.listMcpTools(new ListMcpToolsInput("fake"));
    impl.close();
    assertEquals(1, FakeToolsetAccess.closed());

    impl.reopen();

    assertEquals(2, activities.listMcpTools(new ListMcpToolsInput("fake")).tools().size());
    assertEquals(2, FakeToolsetAccess.created(), "a released toolset is recreated on first use");
  }

  /** Blocks the calling thread until the test opens the gate, as an MCP server start would. */
  private void block() {
    blocked.countDown();
    try {
      if (!gate.await(30, TimeUnit.SECONDS)) {
        throw new IllegalStateException("the gate never opened");
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException(e);
    }
  }

  /** A toolset whose listing blocks the subscribing thread, as ADK's {@code McpToolset} does. */
  private final class BlockingListingToolset implements BaseToolset {
    @Override
    public Flowable<BaseTool> getTools(ReadonlyContext readonlyContext) {
      return Flowable.defer(
          () -> {
            block();
            return Flowable.empty();
          });
    }

    @Override
    public void close() {
      blockingClosed.countDown();
    }
  }

  @Test
  void brokenFactory_failsRetryablyWithItsMessage() {
    McpActivities activities = env.newActivityStub(McpActivities.class);

    ApplicationFailure failure =
        applicationFailure(() -> activities.listMcpTools(new ListMcpToolsInput("broken")));

    assertEquals("googleadk.McpError", failure.getType());
    assertFalse(failure.isNonRetryable(), "the server may come up on a later attempt");
    assertTrue(
        failure.getOriginalMessage().contains("Creating MCP toolset 'broken' failed: cannot start"),
        failure.getOriginalMessage());
  }

  private static ApplicationFailure applicationFailure(Runnable call) {
    ActivityFailure failure = assertThrows(ActivityFailure.class, call::run);
    assertInstanceOf(ApplicationFailure.class, failure.getCause());
    return (ApplicationFailure) failure.getCause();
  }
}
