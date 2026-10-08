package io.temporal.springai.activity;

import io.temporal.springai.model.ChatModelTypes;
import io.temporal.springai.model.ChatModelTypes.Message;
import io.temporal.springai.util.ChatOptionsCodec;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.messages.*;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.DefaultChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.content.Media;
import org.springframework.ai.model.tool.DefaultToolCallingChatOptions;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.CollectionUtils;
import org.springframework.util.MimeType;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.json.JsonMapper;

/**
 * Implementation of {@link ChatModelActivity} that delegates to a Spring AI {@link ChatModel}.
 *
 * <p>This implementation handles the conversion between Temporal-serializable types ({@link
 * ChatModelTypes}) and Spring AI types.
 *
 * <p>Supports multiple chat models. The model to use is determined by the {@code modelName} field
 * in the input. If no model name is specified, the default model is used.
 */
public class ChatModelActivityImpl implements ChatModelActivity {

  private static final Logger log = LoggerFactory.getLogger(ChatModelActivityImpl.class);

  /**
   * Reads the caller's {@link ChatOptions} back out of the serialized JSON carried on {@link
   * ChatModelTypes.ModelOptions}. Plain Jackson — the workflow side wrote the blob with a matching
   * Jackson 3 mapper.
   */
  private static final JsonMapper OPTIONS_MAPPER = ChatOptionsCodec.mapper();

  private final Map<String, ChatModel> chatModels;
  private final String defaultModelName;

  /**
   * Creates an activity implementation with a single chat model.
   *
   * @param chatModel the chat model to use
   */
  public ChatModelActivityImpl(ChatModel chatModel) {
    this.chatModels = Map.of(ChatModelTypes.DEFAULT_MODEL_NAME, chatModel);
    this.defaultModelName = ChatModelTypes.DEFAULT_MODEL_NAME;
  }

  /**
   * Creates an activity implementation with multiple chat models.
   *
   * @param chatModels map of model names to chat models
   * @param defaultModelName the name of the default model to use when none is specified
   */
  public ChatModelActivityImpl(Map<String, ChatModel> chatModels, String defaultModelName) {
    this.chatModels = chatModels;
    this.defaultModelName = defaultModelName;
  }

  @Override
  public ChatModelTypes.ChatModelActivityOutput callChatModel(
      ChatModelTypes.ChatModelActivityInput input) {
    ChatModel chatModel = resolveChatModel(input.modelName());
    Prompt prompt = createPrompt(input, chatModel.getOptions());
    ChatResponse response = chatModel.call(prompt);
    return toOutput(response);
  }

  private ChatModel resolveChatModel(String modelName) {
    String name = (modelName != null && !modelName.isEmpty()) ? modelName : defaultModelName;
    ChatModel model = chatModels.get(name);
    if (model == null) {
      throw new IllegalArgumentException(
          "No chat model with name '" + name + "'. Available models: " + chatModels.keySet());
    }
    return model;
  }

  private Prompt createPrompt(ChatModelTypes.ChatModelActivityInput input, ChatOptions defaults) {
    List<org.springframework.ai.chat.messages.Message> messages =
        input.messages().stream().map(this::toSpringMessage).collect(Collectors.toList());

    ChatOptions callerOptions = tryRehydrateChatOptions(input.modelOptions());
    if (callerOptions == null) {
      callerOptions = commonOptions(input.modelOptions());
    }

    // Spring AI 2 providers expect their own options type and do not merge prompt options
    // with model defaults. Start from the selected worker model, then apply caller overrides.
    // A custom model with only generic defaults can still accept a caller's provider subtype.
    ChatOptions.Builder<?> optionsBuilder;
    if (defaults == null
        || defaults.getClass() == DefaultChatOptions.class
        || defaults.getClass() == DefaultToolCallingChatOptions.class) {
      optionsBuilder = callerOptions.mutate();
      if (defaults != null) {
        optionsBuilder.combineWith(defaults.mutate().combineWith(callerOptions.mutate()));
      }
    } else {
      optionsBuilder = defaults.mutate().combineWith(callerOptions.mutate());
    }

    if (optionsBuilder instanceof ToolCallingChatOptions.Builder<?> toolOptions) {
      // Tools execute in the workflow. Replace even worker-configured callbacks with only
      // the definitions carried by this request, without mutating the model's defaults.
      toolOptions.toolCallbacks(stubToolCallbacks(input));
    } else if (!CollectionUtils.isEmpty(input.tools())) {
      log.debug(
          "ChatOptions {} does not support tool callbacks.", callerOptions.getClass().getName());
    }

    return Prompt.builder().messages(messages).chatOptions(optionsBuilder.build()).build();
  }

  private ChatOptions commonOptions(ChatModelTypes.ModelOptions opts) {
    ToolCallingChatOptions.Builder<?> builder = ToolCallingChatOptions.builder();
    if (opts != null) {
      if (opts.model() != null) builder.model(opts.model());
      if (opts.temperature() != null) builder.temperature(opts.temperature());
      if (opts.maxTokens() != null) builder.maxTokens(opts.maxTokens());
      if (opts.topP() != null) builder.topP(opts.topP());
      if (opts.topK() != null) builder.topK(opts.topK());
      if (opts.frequencyPenalty() != null) builder.frequencyPenalty(opts.frequencyPenalty());
      if (opts.presencePenalty() != null) builder.presencePenalty(opts.presencePenalty());
      if (opts.stopSequences() != null) builder.stopSequences(opts.stopSequences());
    }
    return builder.build();
  }

  private List<ToolCallback> stubToolCallbacks(ChatModelTypes.ChatModelActivityInput input) {
    if (CollectionUtils.isEmpty(input.tools())) {
      return List.of();
    }
    return input.tools().stream()
        .map(
            tool ->
                createStubToolCallback(
                    tool.function().name(),
                    tool.function().description(),
                    tool.function().jsonSchema()))
        .collect(Collectors.toList());
  }

  /**
   * Attempts to rehydrate the caller's exact {@link ChatOptions} subclass from the serialized blob
   * in {@code modelOptions}. Returns {@code null} if the blob is absent or rehydration fails, in
   * which case the caller should use the common-field fallback.
   */
  private ChatOptions tryRehydrateChatOptions(ChatModelTypes.ModelOptions modelOptions) {
    if (modelOptions == null
        || modelOptions.chatOptionsClass() == null
        || modelOptions.chatOptionsJson() == null) {
      return null;
    }
    String className = modelOptions.chatOptionsClass();
    try {
      Class<?> cls = Class.forName(className, true, Thread.currentThread().getContextClassLoader());
      if (!ChatOptions.class.isAssignableFrom(cls)) {
        log.warn(
            "Serialized ChatOptions class {} is not a ChatOptions; falling back to common fields.",
            className);
        return null;
      }
      return (ChatOptions) OPTIONS_MAPPER.readValue(modelOptions.chatOptionsJson(), cls);
    } catch (ClassNotFoundException e) {
      log.warn(
          "Could not load ChatOptions class {} on the activity side; falling back to common"
              + " fields. This typically means spring-ai-<provider> is not on this worker's"
              + " classpath.",
          className);
      return null;
    } catch (JacksonException e) {
      log.warn(
          "Could not deserialize ChatOptions of type {} on the activity side; falling back to"
              + " common fields. Cause: {}",
          className,
          e.getMessage());
      return null;
    }
  }

  private org.springframework.ai.chat.messages.Message toSpringMessage(Message message) {
    return switch (message.role()) {
      case SYSTEM -> new SystemMessage(message.rawContent());
      case USER -> {
        UserMessage.Builder builder = UserMessage.builder().text(message.rawContent());
        if (!CollectionUtils.isEmpty(message.mediaContents())) {
          builder.media(
              message.mediaContents().stream().map(this::toMedia).collect(Collectors.toList()));
        }
        yield builder.build();
      }
      case ASSISTANT ->
          AssistantMessage.builder()
              .content(message.rawContent())
              .properties(assistantMetadata(message))
              .toolCalls(
                  message.toolCalls() != null
                      ? message.toolCalls().stream()
                          .map(
                              tc ->
                                  new AssistantMessage.ToolCall(
                                      tc.id(),
                                      tc.type(),
                                      tc.function().name(),
                                      tc.function().arguments()))
                          .collect(Collectors.toList())
                      : List.of())
              .media(
                  message.mediaContents() != null
                      ? message.mediaContents().stream()
                          .map(this::toMedia)
                          .collect(Collectors.toList())
                      : List.of())
              .build();
      case TOOL ->
          ToolResponseMessage.builder()
              .responses(
                  List.of(
                      new ToolResponseMessage.ToolResponse(
                          message.toolCallId(), message.name(), message.rawContent())))
              .build();
    };
  }

  private Map<String, Object> assistantMetadata(Message message) {
    if (message.metadata() == null) {
      return Map.of();
    }
    Map<String, Object> metadata = new HashMap<>(message.metadata());
    Object thinking = metadata.get("anthropicThinkingContents");
    if (thinking instanceof List<?> blocks && !blocks.isEmpty()) {
      // Temporal decodes metadata as JSON maps. Anthropic expects its typed records in
      // this property, even on a plain AssistantMessage. Restore them only on the worker
      // so the plugin and workflow remain independent of the optional provider module.
      try {
        Class<?> contentType =
            Class.forName(
                "org.springframework.ai.anthropic.AnthropicChatModel$AnthropicThinkingContent",
                true,
                Thread.currentThread().getContextClassLoader());
        metadata.put(
            "anthropicThinkingContents",
            OPTIONS_MAPPER.convertValue(
                blocks,
                OPTIONS_MAPPER.getTypeFactory().constructCollectionType(List.class, contentType)));
      } catch (ClassNotFoundException | JacksonException e) {
        throw new IllegalArgumentException("Could not restore Anthropic thinking continuation", e);
      }
    }
    return metadata;
  }

  private Media toMedia(ChatModelTypes.MediaContent mediaContent) {
    MimeType mimeType = MimeType.valueOf(mediaContent.mimeType());
    if (mediaContent.uri() != null) {
      try {
        return new Media(mimeType, new URI(mediaContent.uri()));
      } catch (URISyntaxException e) {
        throw new RuntimeException("Invalid media URI: " + mediaContent.uri(), e);
      }
    } else if (mediaContent.data() != null) {
      return new Media(mimeType, new ByteArrayResource(mediaContent.data()));
    }
    throw new IllegalArgumentException("Media content must have either uri or data");
  }

  private ChatModelTypes.ChatModelActivityOutput toOutput(ChatResponse response) {
    List<ChatModelTypes.ChatModelActivityOutput.Generation> generations =
        response.getResults().stream()
            .map(
                gen ->
                    new ChatModelTypes.ChatModelActivityOutput.Generation(
                        fromAssistantMessage(gen.getOutput())))
            .collect(Collectors.toList());

    ChatModelTypes.ChatModelActivityOutput.ChatResponseMetadata metadata = null;
    if (response.getMetadata() != null) {
      var rateLimit = response.getMetadata().getRateLimit();
      var usage = response.getMetadata().getUsage();

      metadata =
          new ChatModelTypes.ChatModelActivityOutput.ChatResponseMetadata(
              response.getMetadata().getModel(),
              rateLimit != null
                  ? new ChatModelTypes.ChatModelActivityOutput.ChatResponseMetadata.RateLimit(
                      rateLimit.getRequestsLimit(),
                      rateLimit.getRequestsRemaining(),
                      rateLimit.getRequestsReset(),
                      rateLimit.getTokensLimit(),
                      rateLimit.getTokensRemaining(),
                      rateLimit.getTokensReset())
                  : null,
              usage != null
                  ? new ChatModelTypes.ChatModelActivityOutput.ChatResponseMetadata.Usage(
                      usage.getPromptTokens() != null ? usage.getPromptTokens().intValue() : null,
                      usage.getCompletionTokens() != null
                          ? usage.getCompletionTokens().intValue()
                          : null,
                      usage.getTotalTokens() != null ? usage.getTotalTokens().intValue() : null)
                  : null);
    }

    return new ChatModelTypes.ChatModelActivityOutput(generations, metadata);
  }

  private Message fromAssistantMessage(AssistantMessage assistantMessage) {
    List<Message.ToolCall> toolCalls = null;
    if (!CollectionUtils.isEmpty(assistantMessage.getToolCalls())) {
      toolCalls =
          assistantMessage.getToolCalls().stream()
              .map(
                  tc ->
                      new Message.ToolCall(
                          tc.id(),
                          tc.type(),
                          new Message.ChatCompletionFunction(tc.name(), tc.arguments())))
              .collect(Collectors.toList());
    }

    List<ChatModelTypes.MediaContent> mediaContents = null;
    if (!CollectionUtils.isEmpty(assistantMessage.getMedia())) {
      mediaContents =
          assistantMessage.getMedia().stream().map(this::fromMedia).collect(Collectors.toList());
    }

    return new Message(
        assistantMessage.getText(),
        Message.Role.ASSISTANT,
        null,
        null,
        toolCalls,
        mediaContents,
        assistantMessage.getMetadata());
  }

  private ChatModelTypes.MediaContent fromMedia(Media media) {
    String mimeType = media.getMimeType().toString();
    if (media.getData() instanceof String uri) {
      return new ChatModelTypes.MediaContent(mimeType, uri);
    } else if (media.getData() instanceof byte[] data) {
      ChatModelTypes.checkMediaSize(data);
      return new ChatModelTypes.MediaContent(mimeType, data);
    }
    throw new IllegalArgumentException(
        "Unsupported media data type: " + media.getData().getClass());
  }

  /**
   * Creates a stub ToolCallback that provides a tool definition but throws if called. This is used
   * because Spring AI's ChatModel API requires ToolCallbacks, but we only need to inform the model
   * about available tools. Actual execution happens in the workflow's ChatClient advisor.
   */
  private ToolCallback createStubToolCallback(String name, String description, String inputSchema) {
    ToolDefinition toolDefinition =
        ToolDefinition.builder()
            .name(name)
            .description(description)
            .inputSchema(inputSchema)
            .build();

    return new ToolCallback() {
      @Override
      public ToolDefinition getToolDefinition() {
        return toolDefinition;
      }

      @Override
      public String call(String toolInput) {
        throw new UnsupportedOperationException(
            "Tool execution must be handled by the workflow's ChatClient advisor.");
      }
    };
  }
}
