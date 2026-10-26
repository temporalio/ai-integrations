package io.temporal.googleadk;

import com.google.adk.models.BaseLlm;
import com.google.adk.models.LlmRegistry;
import com.google.adk.tools.BaseToolset;
import io.temporal.common.SimplePlugin;
import io.temporal.googleadk.activity.McpActivitiesImpl;
import io.temporal.googleadk.activity.ModelActivityImpl;
import io.temporal.googleadk.converter.GoogleAdkDataConverter;
import io.temporal.worker.WorkerFactory;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import javax.annotation.Nonnull;

/**
 * Temporal plugin for Google ADK Java.
 *
 * <p>Register it on the {@code WorkflowClientOptions}; workers created from that client inherit it.
 * The plugin registers the model activity on every worker and teaches the client's data converter
 * to serialize ADK types. Models are resolved on the worker by name: explicit registrations first,
 * then the configured factory, which defaults to ADK's {@code LlmRegistry}.
 *
 * <pre>{@code
 * GoogleAdkPlugin plugin =
 *     GoogleAdkPlugin.newBuilder()
 *         .model("gemini-2.5-flash", () -> Gemini.builder().modelName("gemini-2.5-flash").build())
 *         .build();
 * WorkflowClient client =
 *     WorkflowClient.newInstance(
 *         service, WorkflowClientOptions.newBuilder().setPlugins(plugin).build());
 * }</pre>
 */
public final class GoogleAdkPlugin extends SimplePlugin {

  /** The plugin name reported to Temporal. */
  public static final String NAME = "io.temporal.google-adk";

  private final McpActivitiesImpl mcpActivities;

  private GoogleAdkPlugin(Builder builder) {
    this(builder, new McpActivitiesImpl(builder.toolsets));
  }

  private GoogleAdkPlugin(Builder builder, McpActivitiesImpl mcpActivities) {
    super(toSimplePluginBuilder(builder, mcpActivities));
    this.mcpActivities = mcpActivities;
  }

  /** Starts building a plugin. */
  public static Builder newBuilder() {
    return new Builder();
  }

  /** Creates a plugin that resolves every model through ADK's {@code LlmRegistry}. */
  public static GoogleAdkPlugin newDefaultInstance() {
    return newBuilder().build();
  }

  private static SimplePlugin.Builder toSimplePluginBuilder(
      Builder builder, McpActivitiesImpl mcpActivities) {
    Map<String, Supplier<BaseLlm>> models = new LinkedHashMap<>(builder.models);
    Function<String, BaseLlm> factory = builder.modelFactory;
    Function<String, BaseLlm> resolver =
        name -> {
          Supplier<BaseLlm> registered = models.get(name);
          return registered != null ? registered.get() : factory.apply(name);
        };
    SimplePlugin.Builder simple =
        SimplePlugin.newBuilder(NAME)
            .registerActivitiesImplementations(new ModelActivityImpl(resolver), mcpActivities);
    if (builder.customizeDataConverter) {
      simple.customizeDataConverter(GoogleAdkDataConverter::withAdkTypes);
    }
    return simple;
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
      // Toolsets are released once the last in-flight MCP activity completes.
      mcpActivities.close();
    }
  }

  /** Builder for {@link GoogleAdkPlugin}. */
  public static final class Builder {
    private final Map<String, Supplier<BaseLlm>> models = new LinkedHashMap<>();
    private final Map<String, Supplier<BaseToolset>> toolsets = new LinkedHashMap<>();
    private Function<String, BaseLlm> modelFactory = LlmRegistry::getLlm;
    private boolean customizeDataConverter = true;

    private Builder() {}

    /** Serves the given model name with a model created once, on first use, by the supplier. */
    public Builder model(String modelName, Supplier<BaseLlm> model) {
      models.put(
          Objects.requireNonNull(modelName, "modelName"), Objects.requireNonNull(model, "model"));
      return this;
    }

    /** Serves the given model name with the given model instance. */
    public Builder model(String modelName, BaseLlm model) {
      Objects.requireNonNull(model, "model");
      return model(modelName, () -> model);
    }

    /**
     * Sets the factory used for model names without an explicit registration. Defaults to ADK's
     * {@code LlmRegistry}, which builds Gemini models from the environment.
     */
    public Builder modelFactory(Function<String, BaseLlm> modelFactory) {
      this.modelFactory = Objects.requireNonNull(modelFactory, "modelFactory");
      return this;
    }

    /**
     * Serves the named MCP toolset with a toolset created once per worker, on first use, by the
     * supplier. Workflows reach it with {@code TemporalMcpToolset.create(name)}. Any ADK {@code
     * BaseToolset} works; MCP toolsets are the common case. The worker lists and calls its tools
     * without a context, so filter tools on the workflow side with {@code
     * TemporalMcpToolset.Builder#toolFilter} or {@code toolNames} rather than with a context-aware
     * predicate.
     */
    public Builder mcpToolset(String name, Supplier<BaseToolset> toolset) {
      toolsets.put(
          Objects.requireNonNull(name, "name"), Objects.requireNonNull(toolset, "toolset"));
      return this;
    }

    /**
     * Controls whether the plugin replaces the client's JSON payload converter with one that
     * understands ADK types. Disable it only when the client's data converter already does, for
     * example a codec converter built around {@code GoogleAdkDataConverter.newDefaultInstance()}.
     */
    public Builder customizeDataConverter(boolean customizeDataConverter) {
      this.customizeDataConverter = customizeDataConverter;
      return this;
    }

    /** Builds the plugin. */
    public GoogleAdkPlugin build() {
      return new GoogleAdkPlugin(this);
    }
  }
}
