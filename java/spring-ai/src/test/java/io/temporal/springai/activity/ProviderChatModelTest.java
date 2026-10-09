package io.temporal.springai.activity;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

import com.openai.client.OpenAIClient;
import com.openai.client.OpenAIClientAsync;
import com.openai.core.RequestOptions;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import com.openai.models.chat.completions.ChatCompletionMessage;
import io.temporal.springai.model.ChatModelTypes.*;
import io.temporal.springai.util.ChatOptionsCodec;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;

/** Exercises the real provider's request construction with an offline native SDK client. */
class ProviderChatModelTest {
  private OpenAIClient client;

  @BeforeEach
  void setUp() {
    client = mock(OpenAIClient.class, RETURNS_DEEP_STUBS);
    ChatCompletion reply =
        ChatCompletion.builder()
            .id("reply")
            .created(0)
            .model("worker-model")
            .addChoice(
                ChatCompletion.Choice.builder()
                    .index(0)
                    .finishReason(ChatCompletion.Choice.FinishReason.STOP)
                    .logprobs(Optional.empty())
                    .message(
                        ChatCompletionMessage.builder()
                            .content("pong")
                            .refusal(Optional.empty())
                            .build())
                    .build())
            .build();
    when(client
            .chat()
            .completions()
            .create(any(ChatCompletionCreateParams.class), any(RequestOptions.class)))
        .thenReturn(reply);
  }

  @Test
  void noOptions_usesSelectedModelsProviderDefaults() {
    OpenAiChatModel model =
        model(OpenAiChatOptions.builder().model("selected-model").temperature(0.4).build());
    ChatModelActivityImpl activity =
        new ChatModelActivityImpl(
            Map.of(
                "default",
                mock(org.springframework.ai.chat.model.ChatModel.class),
                "selected",
                model),
            "default");
    ChatModelActivityOutput output =
        activity.callChatModel(new ChatModelActivityInput("selected", messages(), null, List.of()));

    assertEquals("pong", output.generations().get(0).message().rawContent());
    ChatCompletionCreateParams request = request();
    assertEquals("selected-model", request.model().asString());
    assertEquals(0.4, request.temperature().orElseThrow());
    assertEquals(0.4, model.getOptions().getTemperature());
  }

  @Test
  void commonOverrides_keepProviderDefaultsAndReplaceWorkerTools() {
    ToolCallback workerTool = mock(ToolCallback.class);
    when(workerTool.getToolDefinition())
        .thenReturn(
            ToolDefinition.builder()
                .name("worker_tool")
                .description("Worker tool")
                .inputSchema("{\"type\":\"object\"}")
                .build());
    OpenAiChatModel model =
        model(
            OpenAiChatOptions.builder()
                .model("worker-model")
                .temperature(0.4)
                .reasoningEffort("medium")
                .toolCallbacks(workerTool)
                .build());
    ModelOptions overrides = new ModelOptions(null, null, 256, null, null, 0.7, null, null);
    FunctionTool tool =
        new FunctionTool(
            new FunctionTool.Function("workflow_tool", "Workflow tool", "{\"type\":\"object\"}"));

    new ChatModelActivityImpl(model)
        .callChatModel(new ChatModelActivityInput(messages(), overrides, List.of(tool)));

    ChatCompletionCreateParams request = request();
    assertEquals("worker-model", request.model().asString());
    assertEquals(0.7, request.temperature().orElseThrow());
    assertEquals(256, request.maxTokens().orElseThrow());
    assertEquals("medium", request.reasoningEffort().orElseThrow().asString());
    assertEquals(1, request.tools().orElseThrow().size());
    assertEquals(
        "workflow_tool", request.tools().orElseThrow().get(0).asFunction().function().name());
    assertEquals(List.of(workerTool), model.getOptions().getToolCallbacks());
    assertEquals(0.4, model.getOptions().getTemperature());
  }

  @Test
  void providerOverrides_keepUnsetWorkerOptions() {
    OpenAiChatModel model =
        model(OpenAiChatOptions.builder().model("worker-model").temperature(0.4).build());
    OpenAiChatOptions caller =
        OpenAiChatOptions.builder().model("caller-model").reasoningEffort("high").build();
    ModelOptions overrides =
        new ModelOptions(
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            OpenAiChatOptions.class.getName(),
            ChatOptionsCodec.mapper().writeValueAsString(caller));

    new ChatModelActivityImpl(model)
        .callChatModel(new ChatModelActivityInput(messages(), overrides, List.of()));

    ChatCompletionCreateParams request = request();
    assertEquals("caller-model", request.model().asString());
    assertEquals(0.4, request.temperature().orElseThrow());
    assertEquals("high", request.reasoningEffort().orElseThrow().asString());
    assertEquals("worker-model", model.getOptions().getModel());
  }

  private OpenAiChatModel model(OpenAiChatOptions options) {
    return OpenAiChatModel.builder()
        .openAiClient(client)
        .openAiClientAsync(mock(OpenAIClientAsync.class))
        .options(options)
        .build();
  }

  private ChatCompletionCreateParams request() {
    ArgumentCaptor<ChatCompletionCreateParams> captor =
        ArgumentCaptor.forClass(ChatCompletionCreateParams.class);
    verify(client.chat().completions()).create(captor.capture(), any(RequestOptions.class));
    return captor.getValue();
  }

  private List<Message> messages() {
    return List.of(new Message("ping", Message.Role.USER));
  }
}
