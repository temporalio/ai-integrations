package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.assertNotNull;

import io.temporal.client.WorkflowClientOptions;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

/**
 * Replays a recorded history with the plugin installed, so a change to how the integration
 * schedules activities or mints ids surfaces as a non-determinism error here. The history was
 * recorded by {@link HitlWorkflowTest} (a paused tool confirmation answered by a signal).
 */
class ReplayHistoryTest {

  @Test
  void hitlHistory_replaysWithThePlugin() throws Exception {
    GoogleAdkPlugin plugin =
        GoogleAdkPlugin.newBuilder()
            .model("scripted", ScriptedModel.of("scripted", ScriptedModel.text("unused")))
            .build();
    TestWorkflowEnvironment testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                .build());
    try (InputStream history = getClass().getResourceAsStream("/histories/hitl_workflow.json")) {
      assertNotNull(history, "recorded history is missing");
      Worker worker = testEnv.newWorker("replay");
      worker.registerWorkflowImplementationTypes(HitlWorkflowTest.ApprovalWorkflowImpl.class);
      worker.replayWorkflowExecution(
          WorkflowExecutionHistory.fromJson(
              new String(history.readAllBytes(), StandardCharsets.UTF_8)));
    } finally {
      testEnv.close();
    }
  }
}
