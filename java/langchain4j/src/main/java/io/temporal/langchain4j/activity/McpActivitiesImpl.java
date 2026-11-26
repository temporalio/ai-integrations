package io.temporal.langchain4j.activity;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.exception.AsyncNotSupportedException;
import dev.langchain4j.exception.ToolErrorVisibleToLlm;
import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.service.tool.ToolExecutionResult;
import io.temporal.activity.Activity;
import io.temporal.client.ActivityCompletionException;
import io.temporal.failure.ApplicationFailure;
import io.temporal.langchain4j.internal.Failures;
import io.temporal.langchain4j.internal.Heartbeats;
import io.temporal.langchain4j.internal.McpCodec;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Lists and calls the tools of the MCP clients registered with {@code LangChain4jPlugin}. Each
 * client is created at most once per worker, on first use. {@link #close()} marks the clients
 * closed and closes every idle one at once; a client with a call in flight is closed when that call
 * completes, and later calls fail with a retryable error so that another worker picks them up.
 */
public final class McpActivitiesImpl implements McpActivities, AutoCloseable {

  /** The failure type when a workflow names an MCP client that the worker doesn't serve. */
  public static final String UNKNOWN_CLIENT = "LangChain4jUnknownMcpClient";

  private static final Logger log = LoggerFactory.getLogger(McpActivitiesImpl.class);

  private final Map<String, Supplier<McpClient>> registrations;
  private final Map<String, Client> clients = new LinkedHashMap<>();
  private boolean closed;

  /** A registered client, created on first use, and the calls it has in flight. */
  private static final class Client {
    private final String name;
    private final Supplier<McpClient> registration;
    private volatile McpClient created;

    /** Calls on this client that have not finished; guarded by the activities' monitor. */
    int inFlight;

    Client(String name, Supplier<McpClient> registration) {
      this.name = name;
      this.registration = registration;
    }

    /** Returns the client, creating it on first use without holding the activities' monitor. */
    McpClient get() {
      McpClient existing = created;
      if (existing != null) {
        return existing;
      }
      synchronized (this) {
        if (created == null) {
          created = Objects.requireNonNull(registration.get(), "MCP client supplier returned null");
        }
        return created;
      }
    }

    void close() throws Exception {
      McpClient existing = created;
      if (existing != null) {
        existing.close();
      }
    }
  }

  /** Creates the activities for the given registrations; each client is created on first use. */
  public McpActivitiesImpl(Map<String, Supplier<McpClient>> registrations) {
    this.registrations = Map.copyOf(Objects.requireNonNull(registrations, "registrations"));
  }

  @Override
  public ListMcpToolsOutput listMcpTools(ListMcpToolsInput input) {
    Client client = enter(input.client());
    try (Heartbeats ignored = Heartbeats.start(Activity.getExecutionContext())) {
      List<ToolSpecification> tools = client.get().listTools();
      return new ListMcpToolsOutput(tools.stream().map(ToolSpecification::toJson).toList());
    } catch (RuntimeException e) {
      throw Failures.mcpFailure("Listing the tools of MCP client '" + input.client() + "'", e);
    } finally {
      exit(client);
    }
  }

  @Override
  public CallMcpToolOutput callMcpTool(CallMcpToolInput input) {
    Client client = enter(input.client());
    String subject = "MCP tool '" + input.tool() + "' of client '" + input.client() + "'";
    AtomicReference<CompletableFuture<ToolExecutionResult>> call = new AtomicReference<>();
    try (Heartbeats heartbeats =
        Heartbeats.start(
            Activity.getExecutionContext(),
            () -> {
              CompletableFuture<ToolExecutionResult> running = call.get();
              if (running != null) {
                // DefaultMcpClient also tells the server that the call is cancelled.
                running.cancel(true);
              }
            })) {
      try {
        ToolExecutionResult result = execute(client.get(), input, call, heartbeats);
        return McpCodec.encode(input.tool(), result);
      } catch (CancellationException e) {
        ActivityCompletionException completion = heartbeats.completion();
        throw completion != null ? completion : e;
      } catch (RuntimeException e) {
        ToolErrorVisibleToLlm forLlm = forLlm(e);
        if (forLlm != null) {
          // Text the tool wrote for the model; LangChain4j sends it to the model as the result.
          log.debug("{} failed with an error for the model", subject, e);
          return McpCodec.errorForLlm(forLlm.messageForLlm());
        }
        throw e;
      }
    } catch (ActivityCompletionException e) {
      throw e;
    } catch (RuntimeException e) {
      throw Failures.mcpFailure(subject, e);
    } finally {
      exit(client);
    }
  }

  /**
   * Runs the call through the client's cancellable asynchronous API, or its blocking API when the
   * client doesn't offer one. Waiting on the future lets a cancellation reported by a heartbeat
   * stop the call and release the activity thread.
   */
  private static ToolExecutionResult execute(
      McpClient client,
      CallMcpToolInput input,
      AtomicReference<CompletableFuture<ToolExecutionResult>> call,
      Heartbeats heartbeats) {
    CompletableFuture<ToolExecutionResult> future =
        client.executeToolAsync(McpCodec.decode(input), null);
    call.set(future);
    if (heartbeats.completion() != null) {
      // Cancelled before the future was published to the heartbeat callback.
      future.cancel(true);
    }
    try {
      return future.get();
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      if (cause instanceof AsyncNotSupportedException) {
        return client.executeTool(McpCodec.decode(input));
      }
      throw cause instanceof RuntimeException runtime ? runtime : new CompletionException(cause);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new CancellationException("Interrupted while waiting for the MCP tool");
    }
  }

  /** Returns the error LangChain4j shows the model, as thrown by the client or wrapped once. */
  @Nullable
  private static ToolErrorVisibleToLlm forLlm(Throwable error) {
    for (Throwable candidate : new Throwable[] {error, error.getCause()}) {
      if (candidate instanceof ToolErrorVisibleToLlm visible) {
        String text = visible.messageForLlm();
        if (text != null && !text.isBlank()) {
          return visible;
        }
      }
    }
    return null;
  }

  /** Resolves the named client and counts a call in flight on it. */
  private Client enter(String name) {
    Supplier<McpClient> registration = registrations.get(name);
    if (registration == null) {
      throw ApplicationFailure.newNonRetryableFailure(
          "No MCP client is registered under '"
              + name
              + "'. Register it with LangChain4jPlugin.newBuilder().mcpClient(...); registered: "
              + registrations.keySet(),
          UNKNOWN_CLIENT);
    }
    synchronized (this) {
      if (closed) {
        throw ApplicationFailure.newFailure(
            "MCP clients on this worker are closed because it is shutting down; retry on another"
                + " worker",
            Failures.MCP_ERROR);
      }
      Client client = clients.computeIfAbsent(name, key -> new Client(key, registration));
      client.inFlight++;
      return client;
    }
  }

  private void exit(Client client) {
    boolean release;
    synchronized (this) {
      client.inFlight--;
      release = closed && client.inFlight == 0 && clients.remove(client.name, client);
    }
    if (release) {
      release(client);
    }
  }

  /**
   * Marks the clients closed. Each is closed now if it has no call in flight, otherwise when its
   * last call completes.
   */
  @Override
  public void close() {
    List<Client> idle = new ArrayList<>();
    synchronized (this) {
      closed = true;
      for (Iterator<Client> held = clients.values().iterator(); held.hasNext(); ) {
        Client client = held.next();
        if (client.inFlight == 0) {
          held.remove();
          idle.add(client);
        }
      }
    }
    idle.forEach(McpActivitiesImpl::release);
  }

  /**
   * Accepts calls again after {@link #close()}. A client closed by then is created again on first
   * use; one still busy with a call stays, and is reused.
   */
  public synchronized void reopen() {
    closed = false;
  }

  /** Closes the client outside the monitor: closing may wait on the server. */
  private static void release(Client client) {
    try {
      client.close();
    } catch (Exception e) {
      log.warn("Closing MCP client '{}' failed", client.name, e);
    }
  }
}
