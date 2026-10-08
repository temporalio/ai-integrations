package io.temporal.springai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.*;

import com.anthropic.client.AnthropicClient;
import com.anthropic.client.AnthropicClientAsync;
import com.anthropic.core.JsonValue;
import com.anthropic.core.RequestOptions;
import com.anthropic.core.http.Headers;
import com.anthropic.core.http.HttpResponseFor;
import com.anthropic.models.messages.*;
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
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.ai.anthropic.AnthropicChatModel;
import org.springframework.ai.anthropic.AnthropicChatOptions;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;

/**
 * Exercises native Anthropic request construction across the Temporal tool loop without
 * credentials.
 */
class AnthropicThinkingContinuationTest {
  @Test
  @SuppressWarnings("unchecked")
  void toolContinuationPreservesSignedAndRedactedThinkingInOrder() throws Exception {
    AnthropicClient sdk = mock(AnthropicClient.class, RETURNS_DEEP_STUBS);
    Usage usage =
        Usage.builder()
            .inputTokens(1)
            .outputTokens(1)
            .cacheCreation(Optional.empty())
            .cacheCreationInputTokens(0)
            .cacheReadInputTokens(0)
            .inferenceGeo(Optional.empty())
            .outputTokensDetails(Optional.empty())
            .serverToolUse(Optional.empty())
            .serviceTier(Optional.empty())
            .build();
    Message first =
        message("first", usage, StopReason.TOOL_USE)
            .addContent(
                ThinkingBlock.builder()
                    .thinking("considering")
                    .signature("first-signature")
                    .build())
            .addContent(RedactedThinkingBlock.builder().data("opaque-redacted-data").build())
            .addContent(
                ThinkingBlock.builder().thinking("ready").signature("second-signature").build())
            .addContent(
                ToolUseBlock.builder()
                    .id("toolu")
                    .name("echo")
                    .caller(DirectCaller.builder().build())
                    .input(JsonValue.from(Map.of()))
                    .build())
            .build();
    Message last =
        message("last", usage, StopReason.END_TURN)
            .addContent(TextBlock.builder().text("done").citations(List.of()).build())
            .build();
    HttpResponseFor<Message> rawResponse = mock(HttpResponseFor.class);
    when(rawResponse.parse()).thenReturn(first, last);
    when(rawResponse.headers()).thenReturn(Headers.builder().build());
    when(sdk.messages()
            .withRawResponse()
            .create(any(MessageCreateParams.class), nullable(RequestOptions.class)))
        .thenReturn(rawResponse);
    AnthropicChatModel provider =
        AnthropicChatModel.builder()
            .anthropicClient(sdk)
            .anthropicClientAsync(mock(AnthropicClientAsync.class))
            .options(
                AnthropicChatOptions.builder()
                    .model("claude-test")
                    .maxTokens(2048)
                    .thinkingEnabled(1024)
                    .build())
            .build();
    try (TestWorkflowEnvironment env = TestWorkflowEnvironment.newInstance()) {
      var worker = env.newWorker("anthropic-thinking");
      worker.registerWorkflowImplementationTypes(ThinkingWorkflowImpl.class);
      worker.registerActivitiesImplementations(new ChatModelActivityImpl(provider));
      env.start();
      var workflow =
          env.getWorkflowClient()
              .newWorkflowStub(
                  ThinkingWorkflow.class,
                  WorkflowOptions.newBuilder()
                      .setTaskQueue("anthropic-thinking")
                      .setWorkflowExecutionTimeout(Duration.ofSeconds(10))
                      .build());
      assertEquals("done", workflow.chat());
      WorkflowReplayer.replayWorkflowExecution(
          env.getWorkflowClient()
              .fetchHistory(WorkflowStub.fromTyped(workflow).getExecution().getWorkflowId()),
          ThinkingWorkflowImpl.class);
    }
    var requests = ArgumentCaptor.forClass(MessageCreateParams.class);
    verify(sdk.messages().withRawResponse(), times(2))
        .create(requests.capture(), nullable(RequestOptions.class));
    var assistant =
        requests.getAllValues().get(1).messages().stream()
            .filter(m -> m.role().asString().equals("assistant"))
            .findFirst()
            .orElseThrow();
    var blocks = assistant.content().asBlockParams();
    assertEquals(4, blocks.size());
    assertEquals("considering", blocks.get(0).asThinking().thinking());
    assertEquals("first-signature", blocks.get(0).asThinking().signature());
    assertEquals("opaque-redacted-data", blocks.get(1).asRedactedThinking().data());
    assertEquals("ready", blocks.get(2).asThinking().thinking());
    assertEquals("second-signature", blocks.get(2).asThinking().signature());
    assertTrue(blocks.get(3).isToolUse());
    assertEquals("toolu", blocks.get(3).asToolUse().id());
  }

  private Message.Builder message(String id, Usage usage, StopReason reason) {
    return Message.builder()
        .id(id)
        .model("claude-test")
        .usage(usage)
        .stopReason(reason)
        .stopSequence(Optional.empty())
        .container(Optional.empty())
        .stopDetails(Optional.empty());
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
          .defaultToolCallbacks(tool)
          .build()
          .prompt()
          .user("ping")
          .call()
          .content();
    }
  }
}
