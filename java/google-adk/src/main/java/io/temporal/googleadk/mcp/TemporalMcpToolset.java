package io.temporal.googleadk.mcp;

import com.google.adk.agents.ReadonlyContext;
import com.google.adk.tools.BaseTool;
import com.google.adk.tools.BaseToolset;
import com.google.adk.tools.ToolContext;
import com.google.genai.types.FunctionDeclaration;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import io.temporal.activity.ActivityOptions;
import io.temporal.googleadk.activity.CallMcpToolInput;
import io.temporal.googleadk.activity.CallMcpToolOutput;
import io.temporal.googleadk.activity.ListMcpToolsInput;
import io.temporal.googleadk.activity.ListMcpToolsOutput;
import io.temporal.googleadk.activity.McpActivities;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.WorkflowLocal;
import io.temporal.workflow.unsafe.WorkflowUnsafe;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Predicate;

/**
 * An ADK toolset whose tools live in an MCP server reached from the worker.
 *
 * <p>The workflow never connects to the server. Listing the tools runs the {@code
 * googleadk.ListMcpTools} activity and calling one runs {@code googleadk.CallMcpTool}; both are
 * served by the toolset the plugin creates on the worker under the same name:
 *
 * <pre>{@code
 * // Worker
 * GoogleAdkPlugin plugin =
 *     GoogleAdkPlugin.newBuilder()
 *         .mcpToolset("docs", () -> new McpToolset(new StdioServerParameters(...)))
 *         .build();
 * // Workflow
 * LlmAgent agent =
 *     LlmAgent.builder()
 *         .name("assistant")
 *         .model(TemporalModel.create("gemini-2.5-flash"))
 *         .tools(TemporalMcpToolset.create("docs"))
 *         .build();
 * }</pre>
 *
 * <p>ADK asks a toolset for its tools before every model call. By default each request runs the
 * list activity again so the workflow sees the server's current tools; {@link
 * Builder#cacheTools(boolean)} keeps the first answer for the rest of the workflow execution
 * instead. The cache is scoped to the execution, so a toolset held by an agent that several
 * executions share stays deterministic.
 */
public final class TemporalMcpToolset implements BaseToolset {

  /** The activity options used when none are configured: a one-minute start-to-close timeout. */
  public static final ActivityOptions DEFAULT_ACTIVITY_OPTIONS =
      ActivityOptions.newBuilder().setStartToCloseTimeout(Duration.ofMinutes(1)).build();

  private final String name;
  private final ActivityOptions activityOptions;
  private final Predicate<FunctionDeclaration> toolFilter;
  private final boolean requireConfirmation;
  private final boolean cacheTools;
  // Per workflow execution: an agent, and so its toolset, is often shared by every execution.
  private final WorkflowLocal<List<BaseTool>> cachedTools = new WorkflowLocal<>();

  private TemporalMcpToolset(Builder builder) {
    this.name = builder.name;
    this.activityOptions = builder.activityOptions;
    this.toolFilter = builder.toolFilter;
    this.requireConfirmation = builder.requireConfirmation;
    this.cacheTools = builder.cacheTools;
  }

  /** Creates a toolset for the registered name with the default options. */
  public static TemporalMcpToolset create(String name) {
    return newBuilder(name).build();
  }

  /** Starts building a toolset for the registered name. */
  public static Builder newBuilder(String name) {
    return new Builder(name);
  }

  /** Returns the toolset name the worker resolves. */
  public String name() {
    return name;
  }

  @Override
  public Flowable<BaseTool> getTools(ReadonlyContext readonlyContext) {
    return Flowable.defer(
        () -> {
          if (cacheTools && cachedTools.get() != null) {
            return Flowable.fromIterable(cachedTools.get());
          }
          McpActivities activities = Workflow.newActivityStub(McpActivities.class, listOptions());
          ListMcpToolsOutput output = activities.listMcpTools(new ListMcpToolsInput(name));
          List<BaseTool> tools = new ArrayList<>();
          for (FunctionDeclaration declaration : output.tools()) {
            if (toolFilter.test(declaration)) {
              tools.add(new ProxyTool(declaration));
            }
          }
          if (cacheTools) {
            cachedTools.set(List.copyOf(tools));
          }
          return Flowable.fromIterable(tools);
        });
  }

  @Override
  public void close() {
    if (cacheTools && WorkflowUnsafe.isWorkflowThread()) {
      cachedTools.set(null);
    }
  }

  private ActivityOptions listOptions() {
    return ActivityOptions.newBuilder(activityOptions).setSummary("ListMcpTools: " + name).build();
  }

  private ActivityOptions callOptions(String tool) {
    return ActivityOptions.newBuilder(activityOptions)
        .setSummary("CallMcpTool: " + name + "/" + tool)
        .build();
  }

  /** A tool whose calls run the {@code googleadk.CallMcpTool} activity. */
  private final class ProxyTool extends BaseTool {
    private final FunctionDeclaration declaration;

    ProxyTool(FunctionDeclaration declaration) {
      super(declaration.name().orElseThrow(), declaration.description().orElse(""));
      this.declaration = declaration;
    }

    @Override
    public Optional<FunctionDeclaration> declaration() {
      return Optional.of(declaration);
    }

    @Override
    public Single<Map<String, Object>> runAsync(Map<String, Object> args, ToolContext toolContext) {
      return Single.defer(
          () -> {
            if (requireConfirmation) {
              if (toolContext.toolConfirmation().isEmpty()) {
                toolContext.requestConfirmation(
                    "Please approve or reject the tool call "
                        + name()
                        + "() by responding with a FunctionResponse with an expected"
                        + " ToolConfirmation payload.");
                return Single.just(
                    Map.of(
                        "error",
                        "This tool call requires confirmation, please approve or reject."));
              }
              if (!toolContext.toolConfirmation().get().confirmed()) {
                return Single.just(Map.of("error", "This tool call is rejected."));
              }
            }
            McpActivities activities =
                Workflow.newActivityStub(McpActivities.class, callOptions(name()));
            CallMcpToolOutput output =
                activities.callMcpTool(new CallMcpToolInput(name, name(), args));
            return Single.just(output.result());
          });
    }
  }

  /** Builder for {@link TemporalMcpToolset}. */
  public static final class Builder {
    private final String name;
    private ActivityOptions activityOptions = DEFAULT_ACTIVITY_OPTIONS;
    private Predicate<FunctionDeclaration> toolFilter = declaration -> true;
    private boolean requireConfirmation;
    private boolean cacheTools;

    private Builder(String name) {
      this.name = Objects.requireNonNull(name, "name");
      if (name.isEmpty()) {
        throw new IllegalArgumentException("name must not be empty");
      }
    }

    /** Sets the activity options for listing and calling tools; summaries are set per call. */
    public Builder activityOptions(ActivityOptions activityOptions) {
      this.activityOptions = Objects.requireNonNull(activityOptions, "activityOptions");
      return this;
    }

    /** Keeps only the tools whose declaration passes the filter. */
    public Builder toolFilter(Predicate<FunctionDeclaration> toolFilter) {
      this.toolFilter = Objects.requireNonNull(toolFilter, "toolFilter");
      return this;
    }

    /** Keeps only the named tools. */
    public Builder toolNames(List<String> toolNames) {
      List<String> names = List.copyOf(toolNames);
      return toolFilter(declaration -> declaration.name().map(names::contains).orElse(false));
    }

    /** Asks the user to confirm every call before it runs (see {@code Hitl}). */
    public Builder requireConfirmation(boolean requireConfirmation) {
      this.requireConfirmation = requireConfirmation;
      return this;
    }

    /** Lists the tools once per workflow execution instead of before every model call. */
    public Builder cacheTools(boolean cacheTools) {
      this.cacheTools = cacheTools;
      return this;
    }

    /** Builds the toolset. */
    public TemporalMcpToolset build() {
      return new TemporalMcpToolset(this);
    }
  }
}
