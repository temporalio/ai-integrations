package io.temporal.googleadk.activity;

import com.google.adk.tools.BaseTool;
import com.google.adk.tools.BaseToolset;
import com.google.genai.types.FunctionDeclaration;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.processors.PublishProcessor;
import io.temporal.activity.Activity;
import io.temporal.client.ActivityCompletionException;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.FailureTypes;
import io.temporal.googleadk.internal.BackgroundCall;
import io.temporal.googleadk.internal.Heartbeats;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Worker-side implementation of {@link McpActivities}. Each registered toolset is created at most
 * once per worker process, on first use, outside the lock that guards the in-flight counts. {@link
 * #close()} marks the toolsets closed and releases every idle one at once; a toolset with a call in
 * flight is released when that call completes, so a server that never answers keeps only its own
 * toolset alive, and later calls fail with a retryable error so another worker picks them up.
 */
public final class McpActivitiesImpl implements McpActivities, AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(McpActivitiesImpl.class);

  private final Map<String, Supplier<BaseToolset>> factories;
  private final Map<String, Toolset> toolsets = new LinkedHashMap<>();
  private boolean closed;

  /** A registered toolset, created on first use, and the tools it offered when last listed. */
  private static final class Toolset {
    private final String name;
    private final Supplier<BaseToolset> factory;
    private volatile BaseToolset created;
    volatile Map<String, BaseTool> tools = Map.of();

    /**
     * Background calls on this toolset that have not finished; guarded by the activities' monitor.
     */
    int inFlight;

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
    Toolset toolset = resolve(name);
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
    }
  }

  @Override
  public CallMcpToolOutput callMcpTool(CallMcpToolInput input) {
    if (input == null || input.tool() == null) {
      throw ApplicationFailure.newNonRetryableFailure(
          "CallMcpTool requires a tool name", FailureTypes.MCP_ERROR);
    }
    Toolset toolset = resolve(input.toolset());
    PublishProcessor<Object> abandon = PublishProcessor.create();
    try (Heartbeats heartbeats =
        Heartbeats.start(Activity.getExecutionContext(), abandon::onComplete)) {
      BaseTool cached = toolset.tools.get(input.tool());
      BaseTool tool = cached != null ? cached : refresh(toolset, abandon).get(input.tool());
      ActivityCompletionException cancelledWhileListing = heartbeats.completion();
      if (cancelledWhileListing != null) {
        // Do not start a call for an activity the worker already knows is cancelled.
        throw cancelledWhileListing;
      }
      if (tool == null) {
        throw ApplicationFailure.newNonRetryableFailure(
            "Toolset '" + input.toolset() + "' has no tool named '" + input.tool() + "'",
            FailureTypes.MCP_ERROR);
      }
      // Run like the listing: a tool may do its blocking work before it returns a result.
      List<Map<String, Object>> results =
          background(
              toolset,
              () -> Single.defer(() -> tool.runAsync(input.args(), null)).toFlowable(),
              abandon);
      ActivityCompletionException completion = heartbeats.completion();
      if (completion != null) {
        throw completion;
      }
      return new CallMcpToolOutput(results.isEmpty() ? null : results.get(0));
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
    }
  }

  /**
   * Lists the toolset's tools and remembers them by name; once abandoned, the collection stops
   * while the listing finishes on its own. Creating the toolset and subscribing to its listing may
   * both block (ADK's {@code McpToolset} starts its server and initializes its session on
   * subscription), so both happen on an I/O thread, and the activity thread is released at once
   * when the call is abandoned.
   */
  private Map<String, BaseTool> refresh(Toolset toolset, PublishProcessor<Object> abandon) {
    List<BaseTool> listed = background(toolset, () -> toolset.get().getTools(null), abandon);
    Map<String, BaseTool> byName = new LinkedHashMap<>();
    for (BaseTool tool : listed) {
      byName.put(tool.name(), tool);
    }
    if (!abandon.hasComplete()) {
      // A listing cut short by cancellation is not the toolset's tool list.
      toolset.tools = byName;
    }
    return byName;
  }

  /**
   * Runs the flow on an I/O thread, counting it as in flight on its toolset until it terminates
   * there, so the toolset is released only after its last background call has finished.
   */
  private <T> List<T> background(
      Toolset toolset, Supplier<Flowable<T>> flow, PublishProcessor<Object> abandon) {
    synchronized (this) {
      if (closed) {
        throw ApplicationFailure.newFailure(
            "MCP toolsets on this worker are closed because it is shutting down; retry on another"
                + " worker",
            FailureTypes.MCP_ERROR);
      }
      toolset.inFlight++;
    }
    return BackgroundCall.collect(flow, abandon, () -> exit(toolset));
  }

  /** Resolves the named toolset holder; the toolset itself is created on first use. */
  private Toolset resolve(String name) {
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
      return toolsets.computeIfAbsent(name, n -> new Toolset(n, factory));
    }
  }

  private void exit(Toolset toolset) {
    boolean release;
    synchronized (this) {
      toolset.inFlight--;
      release = closed && toolset.inFlight == 0 && toolsets.remove(toolset.name, toolset);
    }
    if (release) {
      release(toolset);
    }
  }

  /**
   * Marks the toolsets closed. Each is released now if it has no call in flight, otherwise when its
   * last in-flight call completes.
   */
  @Override
  public void close() {
    List<Toolset> idle = new ArrayList<>();
    synchronized (this) {
      closed = true;
      for (Iterator<Toolset> held = toolsets.values().iterator(); held.hasNext(); ) {
        Toolset toolset = held.next();
        if (toolset.inFlight == 0) {
          held.remove();
          idle.add(toolset);
        }
      }
    }
    for (Toolset toolset : idle) {
      release(toolset);
    }
  }

  /**
   * Accepts calls again after {@link #close()}. A toolset released by then is recreated on first
   * use; one still busy with a background call stays, and is reused.
   */
  public synchronized void reopen() {
    closed = false;
  }

  /** Closes the toolset, outside the monitor: closing may wait on the server. */
  private static void release(Toolset toolset) {
    try {
      toolset.close();
    } catch (Exception e) {
      log.warn("Closing MCP toolset '{}' failed", toolset.name, e);
    }
  }
}
