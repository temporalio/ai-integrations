package io.temporal.springai;

import static org.junit.jupiter.api.Assertions.*;

import io.temporal.client.WorkflowOptions;
import io.temporal.client.WorkflowStub;
import io.temporal.springai.activity.ChatModelActivityImpl;
import io.temporal.springai.chat.TemporalChatClient;
import io.temporal.springai.model.ActivityChatModel;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.testing.WorkflowReplayer;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;

class GeminiThoughtSignatureContinuationTest {
  @Test
  void toolContinuationRestoresBinarySignaturesAndPreservesUnsignedParts() throws Exception {
    byte[] first = {0, 1, -1, 42};
    byte[] last = {7, 8, -128};
    AtomicInteger calls = new AtomicInteger();
    ChatModel provider =
        prompt -> {
          if (calls.getAndIncrement() == 0) {
            return new ChatResponse(
                List.of(
                    new Generation(
                        AssistantMessage.builder()
                            .content("")
                            .properties(
                                Map.of("thoughtSignatures", Arrays.asList(first, null, last)))
                            .toolCalls(
                                List.of(
                                    new AssistantMessage.ToolCall(
                                        "call", "function", "echo", "{}")))
                            .build())));
          }
          AssistantMessage assistant =
              prompt.getInstructions().stream()
                  .filter(AssistantMessage.class::isInstance)
                  .map(AssistantMessage.class::cast)
                  .findFirst()
                  .orElseThrow();
          List<?> signatures =
              assertInstanceOf(List.class, assistant.getMetadata().get("thoughtSignatures"));
          assertEquals(3, signatures.size());
          assertArrayEquals(first, assertInstanceOf(byte[].class, signatures.get(0)));
          assertNull(signatures.get(1));
          assertArrayEquals(last, assertInstanceOf(byte[].class, signatures.get(2)));
          return new ChatResponse(List.of(new Generation(new AssistantMessage("done"))));
        };
    try (TestWorkflowEnvironment env = TestWorkflowEnvironment.newInstance()) {
      var worker = env.newWorker("gemini-thinking");
      worker.registerWorkflowImplementationTypes(ThinkingWorkflowImpl.class);
      worker.registerActivitiesImplementations(new ChatModelActivityImpl(provider));
      env.start();
      var workflow =
          env.getWorkflowClient()
              .newWorkflowStub(
                  ThinkingWorkflow.class,
                  WorkflowOptions.newBuilder()
                      .setTaskQueue("gemini-thinking")
                      .setWorkflowExecutionTimeout(Duration.ofSeconds(10))
                      .build());
      assertEquals("done", workflow.chat());
      WorkflowReplayer.replayWorkflowExecution(
          env.getWorkflowClient()
              .fetchHistory(WorkflowStub.fromTyped(workflow).getExecution().getWorkflowId()),
          ThinkingWorkflowImpl.class);
    }
    assertEquals(2, calls.get());
  }

  @WorkflowInterface
  public interface ThinkingWorkflow {
    @WorkflowMethod
    String chat();
  }

  public static class ThinkingWorkflowImpl implements ThinkingWorkflow {
    @Override
    public String chat() {
      ToolCallback tool =
          new ToolCallback() {
            @Override
            public ToolDefinition getToolDefinition() {
              return ToolDefinition.builder()
                  .name("echo")
                  .description("Echo")
                  .inputSchema("{\"type\":\"object\"}")
                  .build();
            }

            @Override
            public String call(String input) {
              return "echo";
            }
          };
      return TemporalChatClient.builder(ActivityChatModel.forDefault())
          .defaultTools(tool)
          .build()
          .prompt()
          .user("ping")
          .call()
          .content();
    }
  }
}
