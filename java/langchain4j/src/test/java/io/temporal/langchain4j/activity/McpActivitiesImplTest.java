package io.temporal.langchain4j.activity;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.internal.AsyncNotSupported;
import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.service.tool.ToolExecutionResult;
import io.temporal.activity.ActivityOptions;
import io.temporal.common.RetryOptions;
import io.temporal.testing.TestActivityEnvironment;
import java.lang.reflect.Proxy;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/** Worker-side MCP client lifecycle, cancellation and blocking fallback. */
@Timeout(30)
class McpActivitiesImplTest {

  private static final ActivityOptions OPTIONS =
      ActivityOptions.newBuilder()
          .setStartToCloseTimeout(Duration.ofSeconds(10))
          .setHeartbeatTimeout(Duration.ofMillis(300))
          .setRetryOptions(RetryOptions.newBuilder().setMaximumAttempts(1).build())
          .build();

  private final AtomicInteger created = new AtomicInteger();
  private final AtomicInteger closed = new AtomicInteger();
  private final CompletableFuture<ToolExecutionResult> pending = new CompletableFuture<>();
  private volatile boolean asyncSupported = true;
  private TestActivityEnvironment environment;

  @BeforeEach
  void setUp() {
    environment = TestActivityEnvironment.newInstance();
  }

  @AfterEach
  void tearDown() {
    environment.close();
  }

  /** An MCP client with one tool, whose asynchronous calls never complete. */
  private McpClient newClient() {
    created.incrementAndGet();
    return (McpClient)
        Proxy.newProxyInstance(
            McpClient.class.getClassLoader(),
            new Class<?>[] {McpClient.class},
            (proxy, method, args) ->
                switch (method.getName()) {
                  case "key" -> "fake";
                  case "listTools" ->
                      List.of(
                          ToolSpecification.builder().name("wait").description("Waits").build());
                  case "executeToolAsync" ->
                      asyncSupported
                          ? pending
                          : AsyncNotSupported.failedFuture(McpClient.class, "executeToolAsync");
                  case "executeTool" ->
                      ToolExecutionResult.builder().resultText("blocking").build();
                  case "close" -> {
                    closed.incrementAndGet();
                    yield null;
                  }
                  case "hashCode" -> System.identityHashCode(proxy);
                  case "equals" -> proxy == args[0];
                  case "toString" -> "fake MCP client";
                  default -> throw new UnsupportedOperationException(method.getName());
                });
  }

  private McpActivities register(McpActivitiesImpl impl) {
    environment.registerActivitiesImplementations(impl);
    return environment.newActivityStub(McpActivities.class, OPTIONS);
  }

  @Test
  void closesIdleClientsAndCreatesThemAgainAfterReopen() {
    McpActivitiesImpl impl = new McpActivitiesImpl(Map.of("fake", this::newClient));
    McpActivities activities = register(impl);
    assertEquals(1, activities.listMcpTools(new ListMcpToolsInput("fake")).tools().size());
    activities.listMcpTools(new ListMcpToolsInput("fake"));
    assertEquals(1, created.get(), "a client is created once per worker");

    impl.close();
    assertEquals(1, closed.get());
    // Calls during shutdown fail with a retryable error so another worker serves them.
    assertThrows(
        RuntimeException.class, () -> activities.listMcpTools(new ListMcpToolsInput("fake")));

    impl.reopen();
    activities.listMcpTools(new ListMcpToolsInput("fake"));
    assertEquals(2, created.get());
  }

  @Test
  void aCancellationCancelsTheCallAndReleasesTheActivity() {
    McpActivities activities = register(new McpActivitiesImpl(Map.of("fake", this::newClient)));
    environment.requestCancelActivity();

    assertThrows(
        RuntimeException.class,
        () -> activities.callMcpTool(new CallMcpToolInput("fake", "call-1", "wait", "{}")));
    assertTrue(pending.isCancelled(), "the client's future, and so the server call, is cancelled");
  }

  @Test
  void usesTheBlockingCallWhenTheClientHasNoAsynchronousOne() {
    asyncSupported = false;
    McpActivities activities = register(new McpActivitiesImpl(Map.of("fake", this::newClient)));

    CallMcpToolOutput output =
        activities.callMcpTool(new CallMcpToolInput("fake", "call-1", "wait", "{}"));
    assertTrue(output.contents().contains("blocking"), output.contents());
  }
}
