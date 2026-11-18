package io.temporal.langchain4j;

import dev.langchain4j.model.chat.ChatModel;
import io.temporal.common.SimplePlugin;
import io.temporal.langchain4j.activity.ChatModelActivityImpl;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Temporal plugin for LangChain4j.
 *
 * <p>Register it on the {@code WorkflowClientOptions}; workers created from that client inherit it.
 * The plugin registers the {@code langchain4j.Chat} activity, which serves each {@code
 * TemporalChatModel} call with the {@link ChatModel} registered under the same name.
 *
 * <pre>{@code
 * LangChain4jPlugin plugin =
 *     LangChain4jPlugin.newBuilder()
 *         .chatModel("openai", () -> OpenAiChatModel.builder()
 *             .apiKey(System.getenv("OPENAI_API_KEY"))
 *             .modelName("gpt-4o-mini")
 *             .maxRetries(0)
 *             .build())
 *         .build();
 * WorkflowClient client =
 *     WorkflowClient.newInstance(
 *         service, WorkflowClientOptions.newBuilder().setPlugins(plugin).build());
 * }</pre>
 */
public final class LangChain4jPlugin extends SimplePlugin {

  /** The plugin name reported to Temporal. */
  public static final String NAME = "io.temporal.langchain4j";

  private LangChain4jPlugin(Builder builder) {
    super(
        SimplePlugin.newBuilder(NAME)
            .registerActivitiesImplementations(new ChatModelActivityImpl(builder.chatModels)));
  }

  /** Starts building a plugin. */
  public static Builder newBuilder() {
    return new Builder();
  }

  /** Builder for {@link LangChain4jPlugin}. */
  public static final class Builder {
    private final Map<String, Supplier<ChatModel>> chatModels = new LinkedHashMap<>();

    private Builder() {}

    /**
     * Serves {@code TemporalChatModel.create(name)} with a chat model created once per worker, on
     * first use, by the supplier. Turn off the provider client's own retries (for example {@code
     * maxRetries(0)}) so that each activity attempt sends one request and Temporal owns retries.
     */
    public Builder chatModel(String name, Supplier<ChatModel> chatModel) {
      Objects.requireNonNull(name, "name");
      if (name.isEmpty()) {
        throw new IllegalArgumentException("name must not be empty");
      }
      chatModels.put(name, Objects.requireNonNull(chatModel, "chatModel"));
      return this;
    }

    /** Serves {@code TemporalChatModel.create(name)} with the given chat model. */
    public Builder chatModel(String name, ChatModel chatModel) {
      Objects.requireNonNull(chatModel, "chatModel");
      return chatModel(name, () -> chatModel);
    }

    /** Builds the plugin. */
    public LangChain4jPlugin build() {
      return new LangChain4jPlugin(this);
    }
  }
}
