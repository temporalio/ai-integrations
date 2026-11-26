package io.temporal.langchain4j;

import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import io.temporal.common.SimplePlugin;
import io.temporal.langchain4j.activity.ChatModelActivityImpl;
import io.temporal.langchain4j.activity.McpActivitiesImpl;
import io.temporal.langchain4j.activity.StreamingChatModelActivityImpl;
import io.temporal.worker.WorkerFactory;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Supplier;
import javax.annotation.Nonnull;

/**
 * Temporal plugin for LangChain4j.
 *
 * <p>Register it on the {@code WorkflowClientOptions}; workers created from that client inherit it.
 * The plugin registers the {@code langchain4j.Chat} activity, which serves each {@code
 * TemporalChatModel} call with the {@link ChatModel} registered under the same name, {@code
 * langchain4j.StreamingChat}, which serves {@code TemporalStreamingChatModel} with a registered
 * {@link StreamingChatModel}, and the MCP activities, which serve {@code TemporalMcpClient} with a
 * registered {@code McpClient}.
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

  private final McpActivitiesImpl mcpActivities;

  private LangChain4jPlugin(Builder builder) {
    this(builder, new McpActivitiesImpl(builder.mcpClients));
  }

  private LangChain4jPlugin(Builder builder, McpActivitiesImpl mcpActivities) {
    super(
        SimplePlugin.newBuilder(NAME)
            .registerActivitiesImplementations(
                new ChatModelActivityImpl(builder.chatModels),
                new StreamingChatModelActivityImpl(builder.streamingChatModels),
                mcpActivities));
    this.mcpActivities = mcpActivities;
  }

  /** Starts building a plugin. */
  public static Builder newBuilder() {
    return new Builder();
  }

  @Override
  public void startWorkerFactory(
      @Nonnull WorkerFactory factory, @Nonnull Consumer<WorkerFactory> next) {
    // A plugin instance may serve one worker factory after another.
    mcpActivities.reopen();
    super.startWorkerFactory(factory, next);
  }

  @Override
  public void shutdownWorkerFactory(
      @Nonnull WorkerFactory factory, @Nonnull Consumer<WorkerFactory> next) {
    try {
      super.shutdownWorkerFactory(factory, next);
    } finally {
      // Clients are closed once their last in-flight MCP activity completes.
      mcpActivities.close();
    }
  }

  /** Builder for {@link LangChain4jPlugin}. */
  public static final class Builder {
    private final Map<String, Supplier<ChatModel>> chatModels = new LinkedHashMap<>();
    private final Map<String, Supplier<StreamingChatModel>> streamingChatModels =
        new LinkedHashMap<>();
    private final Map<String, Supplier<McpClient>> mcpClients = new LinkedHashMap<>();

    private Builder() {}

    /**
     * Serves {@code TemporalChatModel.create(name)} with a chat model created once per worker, on
     * first use, by the supplier. Turn off the provider client's own retries (for example {@code
     * maxRetries(0)}) so that each activity attempt sends one request and Temporal owns retries.
     */
    public Builder chatModel(String name, Supplier<ChatModel> chatModel) {
      chatModels.put(requireName(name), Objects.requireNonNull(chatModel, "chatModel"));
      return this;
    }

    /** Serves {@code TemporalChatModel.create(name)} with the given chat model. */
    public Builder chatModel(String name, ChatModel chatModel) {
      Objects.requireNonNull(chatModel, "chatModel");
      return chatModel(name, () -> chatModel);
    }

    /**
     * Serves {@code TemporalStreamingChatModel.create(name)} with a streaming chat model created
     * once per worker, on first use, by the supplier. Turn off the provider client's own retries,
     * as for {@link #chatModel(String, Supplier)}.
     */
    public Builder streamingChatModel(
        String name, Supplier<StreamingChatModel> streamingChatModel) {
      streamingChatModels.put(
          requireName(name), Objects.requireNonNull(streamingChatModel, "streamingChatModel"));
      return this;
    }

    /** Serves {@code TemporalStreamingChatModel.create(name)} with the given model. */
    public Builder streamingChatModel(String name, StreamingChatModel streamingChatModel) {
      Objects.requireNonNull(streamingChatModel, "streamingChatModel");
      return streamingChatModel(name, () -> streamingChatModel);
    }

    /**
     * Serves {@code TemporalMcpClient.create(name)} with an MCP client created once per worker, on
     * first use, by the supplier, and closed when the worker factory shuts down. Requires {@code
     * dev.langchain4j:langchain4j-mcp} on the worker's classpath.
     */
    public Builder mcpClient(String name, Supplier<McpClient> mcpClient) {
      mcpClients.put(requireName(name), Objects.requireNonNull(mcpClient, "mcpClient"));
      return this;
    }

    /** Builds the plugin. */
    public LangChain4jPlugin build() {
      return new LangChain4jPlugin(this);
    }

    private static String requireName(String name) {
      Objects.requireNonNull(name, "name");
      if (name.isEmpty()) {
        throw new IllegalArgumentException("name must not be empty");
      }
      return name;
    }
  }
}
