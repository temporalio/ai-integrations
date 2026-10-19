package io.temporal.googleadk.activity;

import com.google.adk.tools.BaseTool;
import com.google.adk.tools.BaseToolset;
import com.google.genai.types.FunctionDeclaration;
import io.reactivex.rxjava3.processors.PublishProcessor;
import io.reactivex.rxjava3.schedulers.Schedulers;
import io.temporal.activity.Activity;
import io.temporal.client.ActivityCompletionException;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.FailureTypes;
import io.temporal.googleadk.internal.Heartbeats;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Worker-side implementation of {@link McpActivities}. Each registered toolset is created at most
 * once per worker process, on first use, outside the lock that guards the in-flight count. {@link
 * #close()} marks the toolsets closed; they are released once the last in-flight call completes,
 * and later calls fail with a retryable error so another worker picks them up.
 */
public final class McpActivitiesImpl implements McpActivities, AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(McpActivitiesImpl.class);

  private final Map<String, Supplier<BaseToolset>> factories;
  private final Map<String, Toolset> toolsets = new LinkedHashMap<>();
  private int inFlight;
  private boolean closed;

  /** A registered toolset, created on first use, and the tools it offered when last listed. */
  private static final class Toolset {
    private final String name;
    private final Supplier<BaseToolset> factory;
    private volatile BaseToolset created;
    volatile Map<String, BaseTool> tools = Map.of();

    Toolset(String name, Supplier<BaseToolset> factory) {
      this.name = name;
      this.factory = factory;
    }

    /** Returns the toolset, creating it on first use without holding the activities' monitor. */
    BaseToolset get() {
      BaseToolset existing = created;
      if (existing != null) {
        return existing;
      }
      synchronized (this) {
        if (created == null) {
          try {
            created = Objects.requireNonNull(factory.get(), "toolset factory returned null");
          } catch (RuntimeException e) {
            throw ApplicationFailure.newFailureWithCause(
                "Creating MCP toolset '" + name + "' failed: " + e.getMessage(),
                FailureTypes.MCP_ERROR,
                e);
          }
        }
        return created;
      }
    }

    void close() throws Exception {
      BaseToolset existing = created;
      if (existing != null) {
        existing.close();
      }
    }
  }

  /** Creates the activity implementation from toolset factories keyed by toolset name. */
  public McpActivitiesImpl(Map<String, Supplier<BaseToolset>> factories) {
    this.factories = Map.copyOf(Objects.requireNonNull(factories, "factories"));
  }

  @Override
  public ListMcpToolsOutput listMcpTools(ListMcpToolsInput input) {
    String name = input == null ? null : input.toolset();
    Toolset toolset = enter(name);
    PublishProcessor<Object> abandon = PublishProcessor.create();
    try (Heartbeats heartbeats =
        Heartbeats.start(Activity.getExecutionContext(), abandon::onComplete)) {
      Map<String, BaseTool> tools = refresh(toolset, abandon);
      ActivityCompletionException completion = heartbeats.completion();
      if (completion != null) {
        throw completion;
      }
      List<FunctionDeclaration> declarations = new ArrayList<>();
      for (BaseTool tool : tools.values()) {
        declarations.add(
            tool.declaration()
                .orElseThrow(
                    () ->
                        ApplicationFailure.newNonRetryableFailure(
                            "Tool '"
                                + tool.name()
                                + "' of toolset '"
                                + name
                                + "' has no declaration",
                            FailureTypes.MCP_ERROR)));
      }
      return new ListMcpToolsOutput(declarations);
    } catch (ApplicationFailure | ActivityCompletionException e) {
      throw e;
    } catch (RuntimeException e) {
      throw ApplicationFailure.newFailureWithCause(
          "Listing tools of toolset '" + name + "' failed: " + e.getMessage(),
          FailureTypes.MCP_ERROR,
          e);
    } finally {
      exit();
    }
  }

  @Override
  public CallMcpToolOutput callMcpTool(CallMcpToolInput input) {
    if (input == null || input.tool() == null) {
      throw ApplicationFailure.newNonRetryableFailure(
          "CallMcpTool requires a tool name", FailureTypes.MCP_ERROR);
    }
    Toolset toolset = enter(input.toolset());
    PublishProcessor<Object> abandon = PublishProcessor.create();
    try (Heartbeats heartbeats =
        Heartbeats.start(Activity.getExecutionContext(), abandon::onComplete)) {
      BaseTool tool = toolset.tools.get(input.tool());
      if (tool == null) {
        tool = refresh(toolset, abandon).get(input.tool());
      }
      if (tool == null && heartbeats.completion() == null) {
        throw ApplicationFailure.newNonRetryableFailure(
            "Toolset '" + input.toolset() + "' has no tool named '" + input.tool() + "'",
            FailureTypes.MCP_ERROR);
      }
      Map<String, Object> result =
          tool == null
              ? null
              : tool.runAsync(input.args(), null)
                  .subscribeOn(Schedulers.io())
                  .toFlowable()
                  .takeUntil(abandon)
                  .firstElement()
                  .blockingGet();
      ActivityCompletionException completion = heartbeats.completion();
      if (completion != null) {
        throw completion;
      }
      return new CallMcpToolOutput(result);
    } catch (ApplicationFailure | ActivityCompletionException e) {
      throw e;
    } catch (RuntimeException e) {
      throw ApplicationFailure.newFailureWithCause(
          "Tool '"
              + input.tool()
              + "' of toolset '"
              + input.toolset()
              + "' failed: "
              + e.getMessage(),
          FailureTypes.MCP_ERROR,
          e);
    } finally {
      exit();
    }
  }

  /** Lists the toolset's tools and remembers them by name; stops early once abandoned. */
  private static Map<String, BaseTool> refresh(Toolset toolset, PublishProcessor<Object> abandon) {
    List<BaseTool> listed = toolset.get().getTools(null).takeUntil(abandon).toList().blockingGet();
    Map<String, BaseTool> byName = new LinkedHashMap<>();
    for (BaseTool tool : listed) {
      byName.put(tool.name(), tool);
    }
    toolset.tools = byName;
    return byName;
  }

  /** Resolves the named toolset holder and counts the call as in flight. */
  private Toolset enter(String name) {
    if (name == null || name.isEmpty()) {
      throw ApplicationFailure.newNonRetryableFailure(
          "An MCP activity requires a toolset name", FailureTypes.MCP_ERROR);
    }
    Supplier<BaseToolset> factory = factories.get(name);
    if (factory == null) {
      throw ApplicationFailure.newNonRetryableFailure(
          "No MCP toolset named '"
              + name
              + "' is registered on this worker; registered: "
              + factories.keySet(),
          FailureTypes.MCP_ERROR);
    }
    synchronized (this) {
      if (closed) {
        throw ApplicationFailure.newFailure(
            "MCP toolsets on this worker are closed because it is shutting down; retry on another"
                + " worker",
            FailureTypes.MCP_ERROR);
      }
      inFlight++;
      return toolsets.computeIfAbsent(name, n -> new Toolset(n, factory));
    }
  }

  private synchronized void exit() {
    inFlight--;
    if (closed && inFlight == 0) {
      release();
    }
  }

  /**
   * Marks the toolsets closed. They are released now if no call is in flight, otherwise when the
   * last in-flight call completes.
   */
  @Override
  public synchronized void close() {
    closed = true;
    if (inFlight == 0) {
      release();
    }
  }

  /** Accepts calls again after {@link #close()}; released toolsets are recreated on first use. */
  public synchronized void reopen() {
    closed = false;
  }

  private void release() {
    for (Map.Entry<String, Toolset> entry : toolsets.entrySet()) {
      try {
        entry.getValue().close();
      } catch (Exception e) {
        log.warn("Closing MCP toolset '{}' failed", entry.getKey(), e);
      }
    }
    toolsets.clear();
  }
}
