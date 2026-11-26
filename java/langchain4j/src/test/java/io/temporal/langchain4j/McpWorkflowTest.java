package io.temporal.langchain4j;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.mcp.McpToolProvider;
import dev.langchain4j.mcp.client.DefaultMcpClient;
import dev.langchain4j.mcp.client.transport.stdio.StdioMcpTransport;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.service.AiServices;
import io.temporal.activity.ActivityOptions;
import io.temporal.api.history.v1.HistoryEvent;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowOptions;
import io.temporal.client.WorkflowStub;
import io.temporal.common.RetryOptions;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.langchain4j.activity.ChatModelActivity;
import io.temporal.langchain4j.activity.McpActivities;
import io.temporal.langchain4j.activity.McpActivitiesImpl;
import io.temporal.langchain4j.internal.Failures;
import io.temporal.langchain4j.mcp.TemporalMcpClient;
import io.temporal.langchain4j.model.TemporalChatModel;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.testing.WorkflowReplayer;
import io.temporal.worker.Worker;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

/**
 * Lists and calls the tools of a real MCP server over stdio through activities, with LangChain4j's
 * own {@code DefaultMcpClient} on the worker and {@code McpToolProvider} in the workflow.
 */
@Timeout(60)
class McpWorkflowTest {

  private static final String TASK_QUEUE = "langchain4j-mcp";
  private static final ActivityOptions FAST_RETRIES =
      ActivityOptions.newBuilder()
          .setStartToCloseTimeout(Duration.ofSeconds(20))
          .setRetryOptions(
              RetryOptions.newBuilder()
                  .setInitialInterval(Duration.ofMillis(10))
                  .setMaximumAttempts(3)
                  .build())
          .build();

  @TempDir static Path temporary;

  private static final List<ChatRequest> requests = new CopyOnWriteArrayList<>();
  private static TestWorkflowEnvironment testEnv;
  private static WorkflowClient client;

  @BeforeAll
  static void startServer() {
    Path arguments = TestClasspath.argumentFile(temporary, McpTestServer.class, entry -> true);
    LangChain4jPlugin plugin =
        LangChain4jPlugin.newBuilder()
            .chatModel("scripted", new ScriptedChatModel(McpWorkflowTest::respond))
            .mcpClient(
                "test",
                () ->
                    DefaultMcpClient.builder()
                        .key("test")
                        .transport(
                            StdioMcpTransport.builder()
                                .command(List.of(TestClasspath.java(), "@" + arguments))
                                .build())
                        .build())
            .build();
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                .build());
    Worker worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(McpWorkflowImpl.class);
    testEnv.start();
    client = testEnv.getWorkflowClient();
  }

  @AfterAll
  static void stopServer() {
    if (testEnv != null) {
      // Shutting the worker factory down closes the MCP client and so the server process.
      testEnv.close();
    }
  }

  /**
   * Asks for the tool named in the user message ({@code call <tool> <text>}), then answers with the
   * tool's result.
   */
  private static ChatResponse respond(ChatRequest request) {
    requests.add(request);
    ChatMessage last = request.messages().get(request.messages().size() - 1);
    if (last instanceof ToolExecutionResultMessage result) {
      return ChatResponse.builder().aiMessage(AiMessage.from(result.text())).build();
    }
    String[] words = ((UserMessage) last).singleText().split(" ", 3);
    return ChatResponse.builder()
        .aiMessage(
            AiMessage.from(
                ToolExecutionRequest.builder()
                    .id("call-1")
                    .name(words[1])
                    .arguments("{\"text\":\"" + words[2] + "\"}")
                    .build()))
        .build();
  }

  public interface Assistant {
    String chat(String message);
  }

  @WorkflowInterface
  public interface McpWorkflow {
    @WorkflowMethod
    String ask(String mcpClient, String question);
  }

  public static final class McpWorkflowImpl implements McpWorkflow {
    @Override
    public String ask(String mcpClient, String question) {
      Assistant assistant =
          AiServices.builder(Assistant.class)
              .chatModel(
                  TemporalChatModel.newBuilder("scripted").activityOptions(FAST_RETRIES).build())
              .toolProvider(
                  McpToolProvider.builder()
                      .mcpClients(
                          TemporalMcpClient.newBuilder(mcpClient)
                              .activityOptions(FAST_RETRIES)
                              .build())
                      .failIfOneServerFails(true)
                      .build())
              .build();
      try {
        return assistant.chat(question);
      } catch (RuntimeException e) {
        return "failed: " + e.getClass().getSimpleName() + ": " + e.getMessage();
      }
    }
  }

  private static McpWorkflow stub() {
    return client.newWorkflowStub(
        McpWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
  }

  private static WorkflowExecutionHistory history(McpWorkflow workflow) {
    return client.fetchHistory(WorkflowStub.fromTyped(workflow).getExecution().getWorkflowId());
  }

  private static List<String> scheduledActivityTypes(WorkflowExecutionHistory history) {
    return history.getEvents().stream()
        .filter(HistoryEvent::hasActivityTaskScheduledEventAttributes)
        .map(e -> e.getActivityTaskScheduledEventAttributes().getActivityType().getName())
        .toList();
  }

  private static List<Integer> callAttempts(WorkflowExecutionHistory history) {
    return history.getEvents().stream()
        .filter(HistoryEvent::hasActivityTaskStartedEventAttributes)
        .map(e -> e.getActivityTaskStartedEventAttributes().getAttempt())
        .toList();
  }

  @Test
  void listsAndCallsServerToolsThroughActivitiesAndReplays() {
    requests.clear();
    McpWorkflow workflow = stub();
    assertEquals("echo: hello", workflow.ask("test", "call echo hello"));

    WorkflowExecutionHistory history = history(workflow);
    assertEquals(
        List.of(
            McpActivities.LIST_MCP_TOOLS,
            ChatModelActivity.CHAT,
            McpActivities.CALL_MCP_TOOL,
            ChatModelActivity.CHAT),
        scheduledActivityTypes(history));
    // The model saw the server's tools with their parameter schemas.
    List<ToolSpecification> tools = requests.get(0).toolSpecifications();
    assertEquals(
        List.of("broken", "echo", "lookup", "strict"),
        tools.stream().map(ToolSpecification::name).sorted().toList());
    assertTrue(tools.get(0).parameters().properties().containsKey("text"));

    // The tool provider and the MCP calls replay from the history without the server.
    assertDoesNotThrow(
        () -> WorkflowReplayer.replayWorkflowExecution(history, McpWorkflowImpl.class));
  }

  @Test
  void anErrorTheServerWroteForTheModelReachesTheModel() {
    McpWorkflow workflow = stub();
    // The server's isError result is LangChain4j's McpApplicationErrorException, which AI Services
    // send to the model; the activity itself succeeds.
    assertEquals("No record matches 42.", workflow.ask("test", "call lookup 42"));
    assertEquals(List.of(1, 1, 1, 1), callAttempts(history(workflow)));
  }

  @Test
  void otherServerErrorsAreRetriedAndReachTheModelWithoutTheServerMessage() {
    McpWorkflow workflow = stub();
    String answer = workflow.ask("test", "call broken now");

    // AI Services send the failure's message to the model by default; it names only the type.
    assertEquals(
        "MCP tool 'broken' of client 'test' failed with"
            + " dev.langchain4j.exception.ToolExecutionException",
        answer);
    WorkflowExecutionHistory history = history(workflow);
    assertEquals(List.of(1, 1, 3, 1), callAttempts(history));
    assertFalse(history.toJson(false).contains(McpTestServer.SECRET));
  }

  @Test
  void rejectedArgumentsFailTheInvocationWithoutRetries() {
    McpWorkflow workflow = stub();
    String answer = workflow.ask("test", "call strict now");

    // AI Services fail the invocation on a ToolArgumentsException by default, with its cause, as
    // for the ToolArgumentsException that LangChain4j's own MCP client throws.
    assertEquals(
        "failed: RuntimeException: MCP tool 'strict' of client 'test' failed: the server"
            + " rejected the tool arguments",
        answer);
    WorkflowExecutionHistory history = history(workflow);
    assertTrue(history.toJson(false).contains(Failures.MCP_TOOL_ARGUMENTS_ERROR));
    assertEquals(List.of(1, 1, 1), callAttempts(history));
    assertFalse(history.toJson(false).contains(McpTestServer.SECRET));
  }

  @Test
  void anUnknownClientIsNotRetried() {
    McpWorkflow workflow = stub();
    String answer = workflow.ask("missing", "call echo hello");

    assertTrue(answer.startsWith("failed: RuntimeException: Failed to retrieve tools"), answer);
    WorkflowExecutionHistory history = history(workflow);
    assertEquals(List.of(McpActivities.LIST_MCP_TOOLS), scheduledActivityTypes(history));
    assertEquals(List.of(1), callAttempts(history));
    assertTrue(history.toJson(false).contains(McpActivitiesImpl.UNKNOWN_CLIENT));
  }
}
