package io.temporal.springai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowOptions;
import io.temporal.springai.activity.ChatModelActivityImpl;
import io.temporal.springai.model.ActivityChatModel;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.ai.anthropic.AnthropicCacheOptions;
import org.springframework.ai.anthropic.AnthropicCacheStrategy;
import org.springframework.ai.anthropic.AnthropicCacheTtl;
import org.springframework.ai.anthropic.AnthropicChatOptions;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.openai.OpenAiChatOptions;

/** Verifies that real immutable provider options survive the Temporal activity boundary. */
class ProviderOptionsPassthroughTest {

  private static final String TASK_QUEUE = "test-spring-ai-provider-options";

  private TestWorkflowEnvironment testEnv;
  private WorkflowClient client;
  private CapturingChatModel model;

  @BeforeEach
  void setUp() {
    testEnv = TestWorkflowEnvironment.newInstance();
    client = testEnv.getWorkflowClient();
    model = new CapturingChatModel();
  }

  @AfterEach
  void tearDown() {
    testEnv.close();
  }

  @Test
  void immutableProviderOptions_surviveActivityRoundTrip() {
    Worker worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(CustomOptionsWorkflowImpl.class);
    worker.registerActivitiesImplementations(new ChatModelActivityImpl(model));
    testEnv.start();

    ChatWorkflow workflow =
        client.newWorkflowStub(
            ChatWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
    assertEquals("pong", workflow.chat("ping"));

    ChatOptions received = model.capturedOptions.get();
    assertNotNull(received, "activity should receive a non-null ChatOptions");
    OpenAiChatOptions custom =
        assertInstanceOf(
            OpenAiChatOptions.class,
            received,
            "activity should receive the exact caller subclass, not a ToolCallingChatOptions");
    assertEquals(
        "high",
        custom.getReasoningEffort(),
        "provider-specific field should survive the round-trip");
    // Common fields should also come through.
    assertEquals(0.7, custom.getTemperature(), 1e-9);
    assertEquals(256, custom.getMaxTokens());
  }

  @Test
  void immutableProviderOptions_surviveChatClientDefaultOptions() {
    Worker worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(ChatClientWorkflowImpl.class);
    worker.registerActivitiesImplementations(new ChatModelActivityImpl(model));
    testEnv.start();

    ChatWorkflow workflow =
        client.newWorkflowStub(
            ChatWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
    assertEquals("pong", workflow.chat("ping"));

    ChatOptions received = model.capturedOptions.get();
    OpenAiChatOptions custom =
        assertInstanceOf(
            OpenAiChatOptions.class, received, "subclass should survive the ChatClient path too");
    assertEquals("medium", custom.getReasoningEffort());
    assertEquals(0.5, custom.getTemperature(), 1e-9);
  }

  @Test
  void defaultAnthropicOptions_surviveActivityRoundTrip() {
    runAnthropicWorkflow(DefaultAnthropicOptionsWorkflowImpl.class);
    AnthropicChatOptions received =
        assertInstanceOf(AnthropicChatOptions.class, model.capturedOptions.get());
    assertEquals("claude-test", received.getModel());
    assertEquals(2048, received.getMaxTokens());
    assertNull(received.getThinking());
    assertEquals(AnthropicCacheStrategy.NONE, received.getCacheOptions().getStrategy());
  }

  @Test
  void anthropicThinkingAndCacheOptions_surviveChatClientRoundTrip() {
    runAnthropicWorkflow(AnthropicOptionsWorkflowImpl.class);
    AnthropicChatOptions received =
        assertInstanceOf(AnthropicChatOptions.class, model.capturedOptions.get());
    assertEquals(
        AnthropicChatOptions.builder().thinkingEnabled(1024).build().getThinking(),
        received.getThinking());
    assertEquals(AnthropicCacheStrategy.SYSTEM_ONLY, received.getCacheOptions().getStrategy());
    assertEquals(
        AnthropicCacheTtl.ONE_HOUR,
        received.getCacheOptions().getMessageTypeTtl().get(MessageType.SYSTEM));
    assertEquals(
        32, received.getCacheOptions().getMessageTypeMinContentLengths().get(MessageType.SYSTEM));
    assertTrue(received.getCacheOptions().isMultiBlockSystemCaching());
  }

  private void runAnthropicWorkflow(Class<? extends ChatWorkflow> implementation) {
    Worker worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(implementation);
    worker.registerActivitiesImplementations(new ChatModelActivityImpl(model));
    testEnv.start();
    ChatWorkflow workflow =
        client.newWorkflowStub(
            ChatWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
    assertEquals("pong", workflow.chat("ping"));
  }

  @Test
  void nullChatOptions_usesCommonFieldFallback() {
    // Sanity: a workflow that doesn't set any prompt-level options still works. The activity
    // gets the plugin's default ToolCallingChatOptions and the capturing model confirms it.
    Worker worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(NoOptionsWorkflowImpl.class);
    worker.registerActivitiesImplementations(new ChatModelActivityImpl(model));
    testEnv.start();

    ChatWorkflow workflow =
        client.newWorkflowStub(
            ChatWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
    assertEquals("pong", workflow.chat("hi"));

    ChatOptions received = model.capturedOptions.get();
    assertNotNull(received, "activity should receive default options even when caller set none");
    // In the fallback path we build a plain ToolCallingChatOptions — no OpenAiChatOptions, no
    // user-provided fields.
    assertNull(received.getTemperature(), "no temperature should be set in the fallback path");
  }

  @WorkflowInterface
  public interface ChatWorkflow {
    @WorkflowMethod
    String chat(String message);
  }

  public static class CustomOptionsWorkflowImpl implements ChatWorkflow {
    @Override
    public String chat(String message) {
      OpenAiChatOptions opts =
          OpenAiChatOptions.builder()
              .temperature(0.7)
              .maxTokens(256)
              .reasoningEffort("high")
              .build();
      // Call ActivityChatModel directly with our custom ChatOptions — same ChatOptions
      // arrives at the activity side. The sibling ChatClient-based test exercises the
      // idiomatic Spring AI entry point.
      ActivityChatModel chatModel = ActivityChatModel.forDefault();
      ChatResponse response =
          chatModel.call(
              new Prompt(
                  List.of(new org.springframework.ai.chat.messages.UserMessage(message)), opts));
      return response.getResult().getOutput().getText();
    }
  }

  public static class ChatClientWorkflowImpl implements ChatWorkflow {
    @Override
    public String chat(String message) {
      OpenAiChatOptions opts =
          OpenAiChatOptions.builder().temperature(0.5).reasoningEffort("medium").build();
      ActivityChatModel chatModel = ActivityChatModel.forDefault();
      ChatClient chatClient = ChatClient.builder(chatModel).defaultOptions(opts.mutate()).build();
      return chatClient.prompt().user(message).call().content();
    }
  }

  public static class NoOptionsWorkflowImpl implements ChatWorkflow {
    @Override
    public String chat(String message) {
      ActivityChatModel chatModel = ActivityChatModel.forDefault();
      ChatResponse response =
          chatModel.call(
              new Prompt(List.of(new org.springframework.ai.chat.messages.UserMessage(message))));
      return response.getResult().getOutput().getText();
    }
  }

  public static class DefaultAnthropicOptionsWorkflowImpl implements ChatWorkflow {
    @Override
    public String chat(String message) {
      AnthropicChatOptions options =
          AnthropicChatOptions.builder().model("claude-test").maxTokens(2048).build();
      return ActivityChatModel.forDefault()
          .call(new Prompt(message, options))
          .getResult()
          .getOutput()
          .getText();
    }
  }

  public static class AnthropicOptionsWorkflowImpl implements ChatWorkflow {
    @Override
    public String chat(String message) {
      AnthropicChatOptions options =
          AnthropicChatOptions.builder()
              .model("claude-test")
              .maxTokens(2048)
              .thinkingEnabled(1024)
              .cacheOptions(
                  AnthropicCacheOptions.builder()
                      .strategy(AnthropicCacheStrategy.SYSTEM_ONLY)
                      .messageTypeTtl(MessageType.SYSTEM, AnthropicCacheTtl.ONE_HOUR)
                      .messageTypeMinContentLength(MessageType.SYSTEM, 32)
                      .multiBlockSystemCaching(true)
                      .build())
              .build();
      return ChatClient.builder(ActivityChatModel.forDefault())
          .defaultOptions(options.mutate())
          .build()
          .prompt()
          .user(message)
          .call()
          .content();
    }
  }

  private static class CapturingChatModel implements ChatModel {
    final AtomicReference<ChatOptions> capturedOptions = new AtomicReference<>();

    @Override
    public ChatResponse call(Prompt prompt) {
      capturedOptions.set(prompt.getOptions());
      return ChatResponse.builder()
          .generations(List.of(new Generation(new AssistantMessage("pong"))))
          .build();
    }

    @Override
    public reactor.core.publisher.Flux<ChatResponse> stream(Prompt prompt) {
      throw new UnsupportedOperationException();
    }
  }
}
