package io.temporal.springai.model;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;

class WorkflowChatOptionsTest {
  @Test
  void providerAndAdditionalCallbacksAreMergedOnceWithoutMutatingDefaults() {
    ToolCallback providerTool = tool("provider");
    ToolCallback additionalTool = tool("additional");
    OpenAiChatOptions defaults =
        OpenAiChatOptions.builder()
            .model("test")
            .reasoningEffort("medium")
            .toolCallbacks(providerTool)
            .toolContext("default", "value")
            .build();
    ToolCallingChatOptions merged =
        new WorkflowChatOptions()
            .mutate()
            .combineWith(defaults.mutate())
            .toolCallbacks(additionalTool)
            .toolContext("additional", "value")
            .build();
    assertEquals(List.of(providerTool, additionalTool), merged.getToolCallbacks());
    assertEquals(Map.of("default", "value", "additional", "value"), merged.getToolContext());
    assertEquals("medium", assertInstanceOf(OpenAiChatOptions.class, merged).getReasoningEffort());
    assertEquals(List.of(providerTool), defaults.getToolCallbacks());
  }

  @Test
  void chatClientDefaultProviderOptionsProduceOneDefinitionPerCallback() {
    ToolCallback tool = tool("echo");
    AtomicReference<Prompt> captured = new AtomicReference<>();
    ChatModel model =
        new ChatModel() {
          @Override
          public ChatOptions getOptions() {
            return new WorkflowChatOptions();
          }

          @Override
          public ChatResponse call(Prompt prompt) {
            captured.set(prompt);
            return new ChatResponse(List.of(new Generation(new AssistantMessage("ok"))));
          }
        };
    assertEquals(
        "ok",
        ChatClient.builder(model)
            .defaultOptions(OpenAiChatOptions.builder().toolCallbacks(tool))
            .build()
            .prompt()
            .user("ping")
            .call()
            .content());
    ToolCallingChatOptions options =
        assertInstanceOf(ToolCallingChatOptions.class, captured.get().getOptions());
    assertEquals(List.of(tool), options.getToolCallbacks());
    assertEquals(1, ToolCallingManager.builder().build().resolveToolDefinitions(options).size());
  }

  private ToolCallback tool(String name) {
    return new ToolCallback() {
      @Override
      public ToolDefinition getToolDefinition() {
        return ToolDefinition.builder()
            .name(name)
            .description(name)
            .inputSchema("{\"type\":\"object\"}")
            .build();
      }

      @Override
      public String call(String input) {
        return "ok";
      }
    };
  }
}
