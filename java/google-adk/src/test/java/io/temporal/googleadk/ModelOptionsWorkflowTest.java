package io.temporal.googleadk;

import static io.temporal.googleadk.TestSupport.history;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.google.adk.agents.LlmAgent;
import com.google.adk.runner.Runner;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.temporal.api.history.v1.HistoryEvent;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowOptions;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.common.converter.DefaultDataConverter;
import io.temporal.googleadk.model.TemporalModel;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/** Model resolution through the factory, and the activity summaries shown in the UI. */
class ModelOptionsWorkflowTest {

  private static final String TASK_QUEUE = "google-adk-model-options";

  private TestWorkflowEnvironment testEnv;
  private WorkflowClient client;

  @BeforeEach
  void setUp() {
    GoogleAdkPlugin plugin =
        GoogleAdkPlugin.newBuilder()
            .model(
                "registered",
                ScriptedModel.of("registered", ScriptedModel.text("from registration")))
            .modelFactory(
                name -> ScriptedModel.of(name, ScriptedModel.text("from factory: " + name)))
            .build();
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                .build());
    client = testEnv.getWorkflowClient();
    Worker worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(ModelsWorkflowImpl.class);
    testEnv.start();
  }

  @AfterEach
  void tearDown() {
    testEnv.close();
  }

  @Test
  @Timeout(90)
  void explicitRegistrationWinsOverTheFactory_andTheFactoryServesTheRest() {
    ModelsWorkflow workflow = newWorkflow();

    assertEquals(
        List.of("from registration", "from factory: gemini-x"),
        workflow.ask(List.of("registered", "gemini-x"), false));
  }

  @Test
  @Timeout(90)
  void summaries_defaultToTheModelNameAndCanBeComputedFromTheRequest() {
    ModelsWorkflow workflow = newWorkflow();
    workflow.ask(List.of("registered", "gemini-x"), true);

    WorkflowExecutionHistory history = history(client, workflow);
    List<String> summaries =
        history.getEvents().stream()
            .filter(HistoryEvent::hasActivityTaskScheduledEventAttributes)
            .map(
                e ->
                    DefaultDataConverter.STANDARD_INSTANCE.fromPayload(
                        e.getUserMetadata().getSummary(), String.class, String.class))
            .collect(Collectors.toList());
    assertEquals(List.of("InvokeModel: registered", "custom: gemini-x"), summaries);
  }

  private ModelsWorkflow newWorkflow() {
    return client.newWorkflowStub(
        ModelsWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
  }

  @WorkflowInterface
  public interface ModelsWorkflow {
    @WorkflowMethod
    List<String> ask(List<String> models, boolean customSummaryForSecond);
  }

  public static class ModelsWorkflowImpl implements ModelsWorkflow {
    @Override
    public List<String> ask(List<String> models, boolean customSummaryForSecond) {
      List<String> answers = new java.util.ArrayList<>();
      for (int i = 0; i < models.size(); i++) {
        TemporalModel.Builder builder = TemporalModel.newBuilder(models.get(i));
        if (customSummaryForSecond && i == 1) {
          builder.summary(request -> "custom: " + request.model().orElse("?"));
        }
        LlmAgent agent =
            LlmAgent.builder()
                .name("agent_" + i)
                .model(builder.build())
                .instruction("Reply.")
                .build();
        Runner runner = TemporalAdk.runnerBuilder().appName("models").agent(agent).build();
        answers.add(
            TestSupport.lastText(
                TemporalAdk.run(
                    runner, "user-1", "session-" + i, Content.fromParts(Part.fromText("hi")))));
      }
      return answers;
    }
  }
}
