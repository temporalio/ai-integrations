package io.temporal.googleadk;

import static io.temporal.googleadk.TestSupport.history;
import static io.temporal.googleadk.TestSupport.scheduledActivityIds;
import static io.temporal.googleadk.TestSupport.scheduledActivityTypes;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.agents.LlmAgent;
import com.google.adk.events.Event;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.runner.Runner;
import com.google.adk.tools.Annotations.Schema;
import com.google.adk.tools.FunctionTool;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.temporal.activity.ActivityInterface;
import io.temporal.activity.ActivityMethod;
import io.temporal.activity.ActivityOptions;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowOptions;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.googleadk.activity.ModelActivity;
import io.temporal.googleadk.model.TemporalModel;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.googleadk.tool.ActivityTools;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.time.Duration;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * The cardinal end-to-end test: a vanilla two-agent ADK tree (a router that transfers to a weather
 * agent) with an in-workflow function tool and an activity-backed tool, driven through ADK's own
 * {@code Runner} inside a workflow. The worker replays every workflow task from history (cache size
 * 0), so an un-wrapped side effect would run twice and change the counts below.
 */
class ToolCallWorkflowTest {

  private static final String TASK_QUEUE = "google-adk-tools";
  private static final String MODEL = "scripted";

  private final ScriptedModel model =
      ScriptedModel.responding(MODEL, ToolCallWorkflowTest::respond);
  private TestWorkflowEnvironment testEnv;
  private WorkflowClient client;
  private Worker worker;

  @BeforeEach
  void setUp() {
    GoogleAdkPlugin plugin = GoogleAdkPlugin.newBuilder().model(MODEL, model).build();
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                .build());
    client = testEnv.getWorkflowClient();
    worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(AssistantWorkflowImpl.class);
    worker.registerActivitiesImplementations(new WeatherActivitiesImpl());
    testEnv.start();
  }

  @AfterEach
  void tearDown() {
    testEnv.close();
  }

  /** The scripted model: it decides from the request which agent is asking and what it has. */
  private static LlmResponse respond(LlmRequest request) {
    if (TestSupport.isAgent(request, "router")) {
      return ScriptedModel.functionCall("transfer_to_agent", Map.of("agent_name", "weather_agent"));
    }
    if (TestSupport.isAgent(request, "weather_agent")) {
      if (TestSupport.hasFunctionResponse(request, "getWeather")) {
        return ScriptedModel.text("It is sunny in Paris.");
      }
      return ScriptedModel.functionCall("getWeather", Map.of("city", "Paris"));
    }
    if (TestSupport.hasFunctionResponse(request, "addNumbers")) {
      return ScriptedModel.text("The sum is 5.");
    }
    return ScriptedModel.functionCall("addNumbers", Map.of("a", 2, "b", 3));
  }

  @Test
  @Timeout(90)
  void transferAndActivityTool_runEveryModelCallAndToolCallAsActivities() throws Exception {
    AssistantWorkflow workflow = newWorkflow();

    assertEquals("It is sunny in Paris.", workflow.ask("weather", "What is the weather in Paris?"));

    WorkflowExecutionHistory history = history(client, workflow);
    assertEquals(
        List.of(
            ModelActivity.INVOKE_MODEL, // router decides to transfer
            ModelActivity.INVOKE_MODEL, // weather agent asks for the tool
            "GetWeather", // the activity-backed tool
            ModelActivity.INVOKE_MODEL), // weather agent answers
        scheduledActivityTypes(history));
    assertEquals(
        4,
        new HashSet<>(scheduledActivityIds(history)).size(),
        "every activity must be scheduled exactly once");
    assertTrue(
        model.requests().get(1).tools().containsKey("getWeather"),
        "the worker-side model sees the activity-backed tool in the request's tool map");
    worker.replayWorkflowExecution(history);
  }

  @Test
  @Timeout(90)
  void inWorkflowTool_runsOnTheWorkflowThreadBetweenTwoModelActivities() throws Exception {
    AssistantWorkflow workflow = newWorkflow();

    assertEquals("The sum is 5.", workflow.ask("math", "What is 2 plus 3?"));

    WorkflowExecutionHistory history = history(client, workflow);
    assertEquals(
        List.of(ModelActivity.INVOKE_MODEL, ModelActivity.INVOKE_MODEL),
        scheduledActivityTypes(history));
    LlmRequest first = model.requests().get(0);
    assertEquals(
        List.of("addNumbers"),
        List.copyOf(first.tools().keySet()),
        "the worker-side model sees the in-workflow tool in the request's tool map");
    assertEquals(
        FunctionTool.create(MathTools.class, "addNumbers").declaration(),
        first.tools().get("addNumbers").declaration());
    worker.replayWorkflowExecution(history);
  }

  private AssistantWorkflow newWorkflow() {
    return client.newWorkflowStub(
        AssistantWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
  }

  @WorkflowInterface
  public interface AssistantWorkflow {
    @WorkflowMethod
    String ask(String scenario, String question);
  }

  @ActivityInterface
  public interface WeatherActivities {
    @ActivityMethod
    String getWeather(@Schema(name = "city", description = "The city to look up") String city);
  }

  public static class WeatherActivitiesImpl implements WeatherActivities {
    @Override
    public String getWeather(String city) {
      return "sunny in " + city;
    }
  }

  /** A plain ADK function tool that runs on the workflow thread. */
  public static final class MathTools {
    private MathTools() {}

    public static Map<String, Object> addNumbers(
        @Schema(name = "a", description = "First number") int a,
        @Schema(name = "b", description = "Second number") int b) {
      return Map.of("sum", a + b);
    }
  }

  public static class AssistantWorkflowImpl implements AssistantWorkflow {
    @Override
    public String ask(String scenario, String question) {
      LlmAgent root = "weather".equals(scenario) ? weatherTree() : mathAgent();
      Runner runner = TemporalAdk.runnerBuilder().appName("assistant").agent(root).build();
      List<Event> events =
          TemporalAdk.run(
              runner, "user-1", "session-1", Content.fromParts(Part.fromText(question)));
      assertTrue(events.size() >= 2, "a tool-calling turn emits several events");
      return TestSupport.lastText(events);
    }

    private static LlmAgent weatherTree() {
      WeatherActivities weather =
          Workflow.newActivityStub(
              WeatherActivities.class,
              ActivityOptions.newBuilder().setStartToCloseTimeout(Duration.ofSeconds(30)).build());
      LlmAgent weatherAgent =
          LlmAgent.builder()
              .name("weather_agent")
              .description("Answers weather questions.")
              .model(TemporalModel.create(MODEL))
              .instruction("Use the getWeather tool.")
              .tools(ActivityTools.create(weather, "getWeather"))
              .build();
      return LlmAgent.builder()
          .name("router")
          .description("Routes questions to specialists.")
          .model(TemporalModel.create(MODEL))
          .instruction("Transfer weather questions to the weather agent.")
          .subAgents(weatherAgent)
          .build();
    }

    private static LlmAgent mathAgent() {
      return LlmAgent.builder()
          .name("math_agent")
          .model(TemporalModel.create(MODEL))
          .instruction("Use the addNumbers tool.")
          .tools(FunctionTool.create(MathTools.class, "addNumbers"))
          .build();
    }
  }
}
