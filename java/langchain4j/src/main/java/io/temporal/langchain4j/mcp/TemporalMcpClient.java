package io.temporal.langchain4j.mcp;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.exception.ToolArgumentsException;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.invocation.InvocationContext;
import dev.langchain4j.mcp.client.McpApplicationErrorException;
import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.mcp.client.McpGetPromptResult;
import dev.langchain4j.mcp.client.McpPrompt;
import dev.langchain4j.mcp.client.McpReadResourceResult;
import dev.langchain4j.mcp.client.McpResource;
import dev.langchain4j.mcp.client.McpResourceTemplate;
import dev.langchain4j.mcp.client.McpRoot;
import dev.langchain4j.service.tool.ToolExecutionResult;
import io.temporal.activity.ActivityOptions;
import io.temporal.failure.ActivityFailure;
import io.temporal.failure.ApplicationFailure;
import io.temporal.langchain4j.activity.CallMcpToolOutput;
import io.temporal.langchain4j.activity.ListMcpToolsInput;
import io.temporal.langchain4j.activity.McpActivities;
import io.temporal.langchain4j.internal.Failures;
import io.temporal.langchain4j.internal.McpCodec;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.unsafe.WorkflowUnsafe;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A LangChain4j {@link McpClient} whose tools are listed and called as Temporal activities.
 *
 * <p>The workflow never connects to the MCP server. Listing the tools runs the {@code
 * langchain4j.ListMcpTools} activity and calling one runs {@code langchain4j.CallMcpTool}; the
 * worker serves both with the {@code McpClient} registered under the same name with {@code
 * LangChain4jPlugin}. Give it to LangChain4j's own {@code McpToolProvider}, so its filters, name
 * mapping and error handling apply unchanged:
 *
 * <pre>{@code
 * Assistant assistant =
 *     AiServices.builder(Assistant.class)
 *         .chatModel(TemporalChatModel.create("openai"))
 *         .toolProvider(
 *             McpToolProvider.builder().mcpClients(TemporalMcpClient.create("docs")).build())
 *         .build();
 * }</pre>
 *
 * <p>Only tools are supported: resources, prompts, roots and health checks throw {@link
 * UnsupportedOperationException}. The {@code InvocationContext} passed to a call is not sent to the
 * worker, because its id and timestamp differ on replay.
 */
public final class TemporalMcpClient implements McpClient {

  /** The activity options used when none are configured: a one-minute start-to-close timeout. */
  public static final ActivityOptions DEFAULT_ACTIVITY_OPTIONS =
      ActivityOptions.newBuilder().setStartToCloseTimeout(Duration.ofMinutes(1)).build();

  private final String name;
  private final ActivityOptions activityOptions;

  private TemporalMcpClient(Builder builder) {
    this.name = builder.name;
    this.activityOptions = builder.activityOptions;
  }

  /** Creates a client for the given registration name with the default activity options. */
  public static TemporalMcpClient create(String name) {
    return newBuilder(name).build();
  }

  /** Starts building a client for the given registration name. */
  public static Builder newBuilder(String name) {
    return new Builder(name);
  }

  /** Returns the name the worker's MCP client is registered under. */
  @Override
  public String key() {
    return name;
  }

  @Override
  public List<ToolSpecification> listTools() {
    McpActivities activities = activities("ListMcpTools: " + name);
    return activities.listMcpTools(new ListMcpToolsInput(name)).tools().stream()
        .map(ToolSpecification::fromJson)
        .toList();
  }

  @Override
  public List<ToolSpecification> listTools(InvocationContext invocationContext) {
    return listTools();
  }

  /**
   * Calls the tool in the {@code langchain4j.CallMcpTool} activity. Errors are reported the way
   * LangChain4j's MCP client reports them, so the AI Service's tool error handlers apply: an error
   * the server wrote for the model is a {@link McpApplicationErrorException}, a rejection of the
   * arguments is a {@link ToolArgumentsException}, and any other failure, once the activity's retry
   * policy gives up, is a {@link ToolExecutionException} that names only the exception type. A
   * cancellation is rethrown as it is.
   */
  @Override
  public ToolExecutionResult executeTool(ToolExecutionRequest executionRequest) {
    McpActivities activities = activities("CallMcpTool: " + name + "/" + executionRequest.name());
    CallMcpToolOutput output;
    try {
      output = activities.callMcpTool(McpCodec.encode(name, executionRequest));
    } catch (ActivityFailure failure) {
      if (!(failure.getCause() instanceof ApplicationFailure cause)) {
        throw failure;
      }
      // AI Services hand a tool error's cause to the error handler, so the activity failure, whose
      // message names worker details, is kept only as a suppressed exception.
      RuntimeException error =
          Failures.MCP_TOOL_ARGUMENTS_ERROR.equals(cause.getType())
              ? new ToolArgumentsException(cause.getOriginalMessage())
              : new ToolExecutionException(cause.getOriginalMessage());
      error.addSuppressed(failure);
      throw error;
    }
    if (output.errorForLlm() != null) {
      throw new McpApplicationErrorException(output.errorForLlm());
    }
    return McpCodec.decode(output);
  }

  @Override
  public ToolExecutionResult executeTool(
      ToolExecutionRequest executionRequest, InvocationContext invocationContext) {
    return executeTool(executionRequest);
  }

  @Override
  public List<McpResource> listResources() {
    throw unsupported("resources");
  }

  @Override
  public List<McpResource> listResources(InvocationContext invocationContext) {
    throw unsupported("resources");
  }

  @Override
  public List<McpResourceTemplate> listResourceTemplates() {
    throw unsupported("resources");
  }

  @Override
  public List<McpResourceTemplate> listResourceTemplates(InvocationContext invocationContext) {
    throw unsupported("resources");
  }

  @Override
  public McpReadResourceResult readResource(String uri) {
    throw unsupported("resources");
  }

  @Override
  public McpReadResourceResult readResource(String uri, InvocationContext invocationContext) {
    throw unsupported("resources");
  }

  @Override
  @Deprecated
  public void subscribeToResource(String uri) {
    throw unsupported("resources");
  }

  @Override
  @Deprecated
  public void unsubscribeFromResource(String uri) {
    throw unsupported("resources");
  }

  @Override
  public long subscribeToResources(List<String> uris) {
    throw unsupported("resources");
  }

  @Override
  public void unsubscribeFromResources(long subscriptionId) {
    throw unsupported("resources");
  }

  @Override
  public List<McpPrompt> listPrompts() {
    throw unsupported("prompts");
  }

  @Override
  public McpGetPromptResult getPrompt(String name, Map<String, Object> arguments) {
    throw unsupported("prompts");
  }

  @Override
  public void checkHealth() {
    throw unsupported("health checks");
  }

  @Override
  @Deprecated
  public void setRoots(List<McpRoot> roots) {
    throw unsupported("roots");
  }

  /** Does nothing: the worker owns the MCP connection and closes it when it shuts down. */
  @Override
  public void close() {}

  private McpActivities activities(String summary) {
    if (!WorkflowUnsafe.isWorkflowThread()) {
      throw new IllegalStateException(
          "TemporalMcpClient '"
              + name
              + "' runs MCP calls as Temporal activities and can only be used from a workflow;"
              + " use the underlying McpClient directly outside workflows");
    }
    return Workflow.newActivityStub(
        McpActivities.class,
        ActivityOptions.newBuilder(activityOptions).setSummary(summary).build());
  }

  private UnsupportedOperationException unsupported(String feature) {
    return new UnsupportedOperationException(
        "TemporalMcpClient '"
            + name
            + "' supports only MCP tools; MCP "
            + feature
            + " are not supported in workflows");
  }

  /** Builder for {@link TemporalMcpClient}. */
  public static final class Builder {
    private final String name;
    private ActivityOptions activityOptions = DEFAULT_ACTIVITY_OPTIONS;

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

    /** Builds the client. */
    public TemporalMcpClient build() {
      return new TemporalMcpClient(this);
    }
  }
}
