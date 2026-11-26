package io.temporal.langchain4j.activity;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.CompleteToolCall;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.PartialThinkingContext;
import dev.langchain4j.model.chat.response.PartialToolCall;
import dev.langchain4j.model.chat.response.PartialToolCallContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.chat.response.StreamingHandle;
import io.temporal.activity.Activity;
import io.temporal.activity.ActivityExecutionContext;
import io.temporal.client.ActivityCompletionException;
import io.temporal.failure.ApplicationFailure;
import io.temporal.langchain4j.internal.ChatCodec;
import io.temporal.langchain4j.internal.Failures;
import io.temporal.langchain4j.internal.Heartbeats;
import io.temporal.langchain4j.internal.ModelStreamPublisher;
import io.temporal.langchain4j.model.ModelStreamEvent;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Runs chat requests against the streaming chat models registered with {@code LangChain4jPlugin},
 * collecting the partial output for the workflow and, when the request names a topic, publishing it
 * to a Workflow Stream while the model runs.
 */
public final class StreamingChatModelActivityImpl implements StreamingChatModelActivity {

  /** The failure type when live delivery of partial output to a Workflow Stream fails. */
  public static final String STREAM_ERROR = "LangChain4jStreamError";

  private static final Logger log = LoggerFactory.getLogger(StreamingChatModelActivityImpl.class);

  private final Map<String, Supplier<StreamingChatModel>> registrations;
  private final Map<String, StreamingChatModel> models = new ConcurrentHashMap<>();

  /** Creates the activity for the given registrations; each model is created on first use. */
  public StreamingChatModelActivityImpl(Map<String, Supplier<StreamingChatModel>> registrations) {
    this.registrations = Map.copyOf(Objects.requireNonNull(registrations, "registrations"));
  }

  @Override
  public StreamingChatOutput streamingChat(StreamingChatInput input) {
    String name = input.request().model();
    StreamingChatModel model = model(name);
    ActivityExecutionContext context = Activity.getExecutionContext();
    Collector collector = new Collector();
    // A cancellation reported by a heartbeat cancels the provider's stream, when it supports
    // that, and releases the activity thread at once.
    try (Heartbeats heartbeats = Heartbeats.start(context, collector::abandon);
        ModelStreamPublisher publisher = publisher(context, input)) {
      collector.publishTo(publisher);
      try {
        model.chat(ChatCodec.decode(input.request()), collector);
        ChatResponse response = collector.await();
        if (publisher != null) {
          publisher.terminal(ModelStreamEvent.Kind.COMPLETE);
        }
        return new StreamingChatOutput(ChatCodec.encode(response), collector.parts());
      } catch (CancellationException e) {
        ActivityCompletionException completion = heartbeats.completion();
        RuntimeException failure = completion != null ? completion : e;
        if (publisher != null) {
          publisher.failed(ModelStreamEvent.Kind.CANCELED, failure);
        }
        throw failure;
      } catch (RuntimeException | Error e) {
        if (publisher != null) {
          publisher.failed(ModelStreamEvent.Kind.FAILED, e);
        }
        throw e;
      }
    } catch (ApplicationFailure | ActivityCompletionException e) {
      throw e;
    } catch (ModelStreamPublisher.PublishFailure e) {
      log.warn("Live delivery of chat model {} output failed", name, e);
      String message =
          "Live delivery of chat model "
              + name
              + " output to topic '"
              + input.streamingTopic()
              + "' failed with "
              + e.getCause().getClass().getName();
      throw e.isNonRetryable()
          ? ApplicationFailure.newNonRetryableFailure(message, STREAM_ERROR)
          : ApplicationFailure.newFailure(message, STREAM_ERROR);
    } catch (RuntimeException e) {
      throw Failures.modelFailure(name, e);
    }
  }

  @Nullable
  private static ModelStreamPublisher publisher(
      ActivityExecutionContext context, StreamingChatInput input) {
    if (input.streamingTopic() == null) {
      return null;
    }
    if (input.streamingTopic().isBlank()) {
      throw ApplicationFailure.newNonRetryableFailure(
          "streamingTopic must not be blank", STREAM_ERROR);
    }
    try {
      return new ModelStreamPublisher(context, input.request().model(), input.streamingTopic());
    } catch (LinkageError e) {
      throw ApplicationFailure.newNonRetryableFailure(
          "Live streaming requires matching io.temporal:temporal-sdk and temporal-workflowstreams"
              + " artifacts on the worker; use the unshaded SDK for live streaming",
          STREAM_ERROR);
    } catch (RuntimeException e) {
      log.warn("Live delivery to topic '{}' could not start", input.streamingTopic(), e);
      throw ApplicationFailure.newFailure(
          "Live delivery of chat model "
              + input.request().model()
              + " output to topic '"
              + input.streamingTopic()
              + "' could not start: "
              + e.getClass().getName(),
          STREAM_ERROR);
    }
  }

  private StreamingChatModel model(String name) {
    Supplier<StreamingChatModel> registration = registrations.get(name);
    if (registration == null) {
      throw ApplicationFailure.newNonRetryableFailure(
          "No streaming chat model is registered under '"
              + name
              + "'. Register it with LangChain4jPlugin.newBuilder().streamingChatModel(...);"
              + " registered: "
              + registrations.keySet(),
          ChatModelActivityImpl.UNKNOWN_MODEL);
    }
    return models.computeIfAbsent(name, key -> Objects.requireNonNull(registration.get(), key));
  }

  /**
   * Collects one streamed response on the provider's threads. After the attempt is abandoned, later
   * output of the provider is dropped.
   */
  private static final class Collector implements StreamingChatResponseHandler {
    private final CompletableFuture<ChatResponse> response = new CompletableFuture<>();
    private final List<ChatStreamPart> parts = new ArrayList<>();
    @Nullable private ModelStreamPublisher publisher;
    @Nullable private volatile StreamingHandle handle;

    synchronized void publishTo(@Nullable ModelStreamPublisher publisher) {
      this.publisher = publisher;
    }

    @Override
    public void onPartialResponse(String partialResponse) {
      add(
          new ChatStreamPart(
              ChatStreamPart.Type.PARTIAL_RESPONSE, partialResponse, null, null, null));
    }

    @Override
    public void onPartialResponse(PartialResponse partialResponse, PartialResponseContext context) {
      handle = context.streamingHandle();
      onPartialResponse(partialResponse.text());
    }

    @Override
    public void onPartialThinking(PartialThinking partialThinking) {
      add(
          new ChatStreamPart(
              ChatStreamPart.Type.PARTIAL_THINKING, partialThinking.text(), null, null, null));
    }

    @Override
    public void onPartialThinking(PartialThinking partialThinking, PartialThinkingContext context) {
      handle = context.streamingHandle();
      onPartialThinking(partialThinking);
    }

    @Override
    public void onPartialToolCall(PartialToolCall partialToolCall) {
      add(
          new ChatStreamPart(
              ChatStreamPart.Type.PARTIAL_TOOL_CALL,
              partialToolCall.partialArguments(),
              partialToolCall.index(),
              partialToolCall.id(),
              partialToolCall.name()));
    }

    @Override
    public void onPartialToolCall(PartialToolCall partialToolCall, PartialToolCallContext context) {
      handle = context.streamingHandle();
      onPartialToolCall(partialToolCall);
    }

    @Override
    public void onCompleteToolCall(CompleteToolCall completeToolCall) {
      ToolExecutionRequest request = completeToolCall.toolExecutionRequest();
      add(
          new ChatStreamPart(
              ChatStreamPart.Type.COMPLETE_TOOL_CALL,
              request.arguments(),
              completeToolCall.index(),
              request.id(),
              request.name()));
    }

    @Override
    public void onCompleteResponse(ChatResponse completeResponse) {
      response.complete(completeResponse);
    }

    @Override
    public void onError(Throwable error) {
      response.completeExceptionally(error);
    }

    private synchronized void add(ChatStreamPart part) {
      if (response.isDone()) {
        return;
      }
      parts.add(part);
      if (publisher != null) {
        try {
          publisher.accept(part);
        } catch (RuntimeException e) {
          // Fail the attempt on the activity thread and stop the provider.
          response.completeExceptionally(e);
          cancelStream();
        }
      }
    }

    /** Stops the provider's stream, if it supports that, and releases the activity thread. */
    void abandon() {
      response.cancel(false);
      cancelStream();
    }

    private void cancelStream() {
      StreamingHandle current = handle;
      if (current != null) {
        try {
          current.cancel();
        } catch (RuntimeException e) {
          // UnsupportedFeatureException: the provider finishes on its own; its output is dropped.
          log.debug("Cancelling the model's stream failed", e);
        }
      }
    }

    ChatResponse await() {
      try {
        return response.get();
      } catch (ExecutionException e) {
        Throwable cause = e.getCause();
        if (cause instanceof RuntimeException runtime) {
          throw runtime;
        }
        if (cause instanceof Error error) {
          throw error;
        }
        throw new IllegalStateException(cause);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new CancellationException("Interrupted while waiting for the model");
      }
    }

    synchronized List<ChatStreamPart> parts() {
      return List.copyOf(parts);
    }
  }
}
