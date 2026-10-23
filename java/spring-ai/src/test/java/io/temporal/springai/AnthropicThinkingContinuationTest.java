package io.temporal.springai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.*;

import com.anthropic.client.AnthropicClient;
import com.anthropic.client.AnthropicClientAsync;
import com.anthropic.core.ObjectMappers;
import com.anthropic.core.RequestOptions;
import com.anthropic.core.http.Headers;
import com.anthropic.core.http.HttpResponseFor;
import com.anthropic.models.messages.*;
import com.fasterxml.jackson.core.JsonProcessingException;
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
  void toolContinuationPreservesSignedAndRedactedThinkingInOrder() throws Exception {
    AnthropicClient sdk = mock(AnthropicClient.class, RETURNS_DEEP_STUBS);
    Message first =
        message(
            "first",
            "tool_use",
            """
            [
              {"type":"thinking","thinking":"considering","signature":"first-signature"},
              {"type":"redacted_thinking","data":"opaque-redacted-data"},
              {"type":"thinking","thinking":"ready","signature":"second-signature"},
              {"type":"tool_use","id":"toolu","name":"echo","caller":{"type":"direct"},"input":{}}
            ]
            """);
    Message last =
        message(
            "last",
            "end_turn",
            """
            [{"type":"text","text":"done","citations":[]}]
            """);
    HttpResponseFor<Message> rawResponse = mock();
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

  private Message message(String id, String reason, String content) throws JsonProcessingException {
    // Decode a provider response so new nullable SDK fields do not require version-specific
    // setters.
    return ObjectMappers.jsonMapper()
        .readValue(
            """
            {
              "id":"%s","type":"message","role":"assistant","model":"claude-test",
              "content":%s,"stop_reason":"%s","stop_sequence":null,
              "container":null,"stop_details":null,"diagnostics":null,
              "usage":{"input_tokens":1,"output_tokens":1,
                       "cache_creation_input_tokens":0,"cache_read_input_tokens":0}
            }
            """
                .formatted(id, content, reason),
            Message.class);
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
