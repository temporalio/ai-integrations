package io.temporal.googleadk;

import static io.temporal.googleadk.TestSupport.history;
import static io.temporal.googleadk.TestSupport.scheduledActivityTypes;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.agents.LlmAgent;
import com.google.adk.events.Event;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.runner.Runner;
import com.google.adk.tools.BaseTool;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowOptions;
import io.temporal.client.WorkflowStub;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.googleadk.activity.McpActivities;
import io.temporal.googleadk.activity.ModelActivity;
import io.temporal.googleadk.mcp.TemporalMcpToolset;
import io.temporal.googleadk.model.TemporalModel;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.googleadk.tool.Hitl;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.QueryMethod;
import io.temporal.workflow.SignalMethod;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/** MCP toolsets are listed and called through activities served by the worker's toolset. */
class McpWorkflowTest {

  private static final String TASK_QUEUE = "google-adk-mcp";
  private static final String MODEL = "scripted";

  private TestWorkflowEnvironment testEnv;
  private WorkflowClient client;
  private Worker worker;

  @BeforeEach
  void setUp() {
    FakeToolset.reset();
    GoogleAdkPlugin plugin =
        GoogleAdkPlugin.newBuilder()
            .model(MODEL, ScriptedModel.responding(MODEL, McpWorkflowTest::respond))
            .mcpToolset("fake", FakeToolset::new)
            .build();
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                .build());
    client = testEnv.getWorkflowClient();
    worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(
        McpWorkflowImpl.class, FilteredToolsWorkflowImpl.class, ConfirmedMcpWorkflowImpl.class);
    testEnv.start();
  }

  @AfterEach
  void tearDown() {
    testEnv.close();
  }

  private static LlmResponse respond(LlmRequest request) {
    Optional<Map<String, Object>> echo = TestSupport.functionResponse(request, "echo");
    if (echo.isPresent()) {
      return echo.get().containsKey("error")
          ? ScriptedModel.text("I need your approval to call the server.")
          : ScriptedModel.text("The server said hi.");
    }
    boolean advertised =
        request.config().flatMap(c -> c.tools()).orElse(List.of()).stream()
            .flatMap(tool -> tool.functionDeclarations().orElse(List.of()).stream())
            .anyMatch(declaration -> declaration.name().orElse("").equals("echo"));
    assertTrue(advertised, "the MCP tool must be advertised to the model with its schema");
    return ScriptedModel.functionCall("echo", Map.of("text", "hi"));
  }

  @Test
  @Timeout(90)
  void mcpTool_isListedAndCalledThroughActivities() throws Exception {
    McpWorkflow workflow = newWorkflow();

    assertEquals("The server said hi.", workflow.ask(false, "say hi"));

    WorkflowExecutionHistory history = history(client, workflow);
    assertEquals(
        List.of(
            McpActivities.LIST_MCP_TOOLS, // the tools are listed before each model call
            ModelActivity.INVOKE_MODEL,
            McpActivities.CALL_MCP_TOOL,
            McpActivities.LIST_MCP_TOOLS,
            ModelActivity.INVOKE_MODEL),
        scheduledActivityTypes(history));
    assertEquals(1, FakeToolset.created.get(), "the worker creates the toolset once");
    assertEquals(1, FakeToolset.calls.get());
    worker.replayWorkflowExecution(history);
  }

  @Test
  @Timeout(90)
  void toolNames_filterTheAdvertisedTools() {
    FilteredToolsWorkflow workflow =
        client.newWorkflowStub(
            FilteredToolsWorkflow.class,
            WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());

    assertEquals(List.of("wait"), workflow.advertisedTools(List.of("wait")));
  }

  @Test
  @Timeout(90)
  void requireConfirmation_holdsTheCallUntilTheSignal() throws Exception {
    ConfirmedMcpWorkflow workflow =
        client.newWorkflowStub(
            ConfirmedMcpWorkflow.class,
            WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
    WorkflowClient.start(workflow::askWithConfirmation, "say hi");
    for (int i = 0; i < 300 && workflow.pendingTools().isEmpty(); i++) {
      Thread.sleep(100);
    }
    assertEquals(List.of("echo"), workflow.pendingTools());
    assertEquals(0, FakeToolset.calls.get(), "the server is not called before approval");

    workflow.decide(true);

    assertEquals("The server said hi.", WorkflowStub.fromTyped(workflow).getResult(String.class));
    assertEquals(1, FakeToolset.calls.get());
    assertEquals(
        List.of(
            McpActivities.LIST_MCP_TOOLS,
            ModelActivity.INVOKE_MODEL, // asks for the tool
            McpActivities.LIST_MCP_TOOLS,
            ModelActivity.INVOKE_MODEL, // explains the pause
            McpActivities.LIST_MCP_TOOLS,
            McpActivities.CALL_MCP_TOOL, // runs after the confirmation
            McpActivities.LIST_MCP_TOOLS,
            ModelActivity.INVOKE_MODEL), // answers
        scheduledActivityTypes(history(client, workflow)));
    worker.replayWorkflowExecution(history(client, workflow));
  }

  @Test
  @Timeout(90)
  void cachedTools_listOnce() throws Exception {
    McpWorkflow workflow = newWorkflow();

    assertEquals("The server said hi.", workflow.ask(true, "say hi"));

    assertEquals(
        List.of(
            McpActivities.LIST_MCP_TOOLS,
            ModelActivity.INVOKE_MODEL,
            McpActivities.CALL_MCP_TOOL,
            ModelActivity.INVOKE_MODEL),
        scheduledActivityTypes(history(client, workflow)));
  }

  @Test
  @Timeout(90)
  void toolsets_areClosedWhenTheWorkerShutsDown() {
    McpWorkflow workflow = newWorkflow();
    workflow.ask(false, "say hi");
    assertEquals(0, FakeToolset.closed.get());

    testEnv.close();

    assertEquals(1, FakeToolset.closed.get());
  }

  private McpWorkflow newWorkflow() {
    return client.newWorkflowStub(
        McpWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
  }

  @WorkflowInterface
  public interface McpWorkflow {
    @WorkflowMethod
    String ask(boolean cacheTools, String message);
  }

  @WorkflowInterface
  public interface FilteredToolsWorkflow {
    @WorkflowMethod
    List<String> advertisedTools(List<String> toolNames);
  }

  @WorkflowInterface
  public interface ConfirmedMcpWorkflow {
    @WorkflowMethod
    String askWithConfirmation(String message);

    @SignalMethod
    void decide(boolean confirmed);

    @QueryMethod
    List<String> pendingTools();
  }

  public static class FilteredToolsWorkflowImpl implements FilteredToolsWorkflow {
    @Override
    public List<String> advertisedTools(List<String> toolNames) {
      TemporalMcpToolset toolset =
          TemporalMcpToolset.newBuilder("fake").toolNames(toolNames).build();
      return TemporalAdk.await(toolset.getTools(null)).stream().map(BaseTool::name).toList();
    }
  }

  public static class ConfirmedMcpWorkflowImpl implements ConfirmedMcpWorkflow {
    private final List<String> pendingTools = new ArrayList<>();
    private Boolean decision;

    @Override
    public String askWithConfirmation(String message) {
      LlmAgent agent =
          LlmAgent.builder()
              .name("assistant")
              .model(TemporalModel.create(MODEL))
              .instruction("Use the echo tool.")
              .tools(TemporalMcpToolset.newBuilder("fake").requireConfirmation(true).build())
              .build();
      Runner runner = TemporalAdk.runnerBuilder().appName("mcp").agent(agent).build();
      List<Event> events =
          TemporalAdk.run(runner, "user-1", "session-1", Content.fromParts(Part.fromText(message)));
      List<Hitl.PendingConfirmation> pending = Hitl.pendingConfirmations(events);
      if (pending.isEmpty()) {
        return TestSupport.lastText(events);
      }
      pendingTools.add(pending.get(0).toolName());
      Workflow.await(() -> decision != null);
      events =
          TemporalAdk.run(
              runner,
              "user-1",
              "session-1",
              Hitl.confirmationResponse(
                  new Hitl.ConfirmationDecision(pending.get(0).functionCallId(), decision)));
      pendingTools.clear();
      return TestSupport.lastText(events);
    }

    @Override
    public void decide(boolean confirmed) {
      this.decision = confirmed;
    }

    @Override
    public List<String> pendingTools() {
      return new ArrayList<>(pendingTools);
    }
  }

  public static class McpWorkflowImpl implements McpWorkflow {
    @Override
    public String ask(boolean cacheTools, String message) {
      LlmAgent agent =
          LlmAgent.builder()
              .name("assistant")
              .model(TemporalModel.create(MODEL))
              .instruction("Use the echo tool.")
              .tools(TemporalMcpToolset.newBuilder("fake").cacheTools(cacheTools).build())
              .build();
      Runner runner = TemporalAdk.runnerBuilder().appName("mcp").agent(agent).build();
      List<Event> events =
          TemporalAdk.run(runner, "user-1", "session-1", Content.fromParts(Part.fromText(message)));
      return TestSupport.lastText(events);
    }
  }
}
