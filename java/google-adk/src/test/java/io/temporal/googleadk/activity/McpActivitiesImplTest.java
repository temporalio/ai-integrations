package io.temporal.googleadk.activity;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.genai.types.FunctionDeclaration;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.failure.ActivityFailure;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.FakeToolsetAccess;
import io.temporal.googleadk.converter.GoogleAdkDataConverter;
import io.temporal.testing.TestActivityEnvironment;
import io.temporal.testing.TestEnvironmentOptions;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** The MCP activities serve registered toolsets and fail with typed errors otherwise. */
class McpActivitiesImplTest {

  private TestActivityEnvironment env;
  private McpActivitiesImpl impl;

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
    impl = new McpActivitiesImpl(Map.of("fake", FakeToolsetAccess::newToolset));
    env.registerActivitiesImplementations(impl);
  }

  @AfterEach
  void tearDown() {
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
    assertTrue(failure.getOriginalMessage().contains("registered: [fake]"));
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

  private static ApplicationFailure applicationFailure(Runnable call) {
    ActivityFailure failure = assertThrows(ActivityFailure.class, call::run);
    assertInstanceOf(ApplicationFailure.class, failure.getCause());
    return (ApplicationFailure) failure.getCause();
  }
}
