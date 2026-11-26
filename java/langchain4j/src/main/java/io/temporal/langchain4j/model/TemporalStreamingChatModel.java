package io.temporal.langchain4j.model;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.CompleteToolCall;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.PartialThinkingContext;
import dev.langchain4j.model.chat.response.PartialToolCall;
import dev.langchain4j.model.chat.response.PartialToolCallContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.chat.response.StreamingHandle;
import io.temporal.activity.ActivityOptions;
import io.temporal.langchain4j.activity.ChatStreamPart;
import io.temporal.langchain4j.activity.StreamingChatInput;
import io.temporal.langchain4j.activity.StreamingChatModelActivity;
import io.temporal.langchain4j.activity.StreamingChatOutput;
import io.temporal.langchain4j.internal.ChatCodec;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.unsafe.WorkflowUnsafe;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * A LangChain4j {@link StreamingChatModel} whose calls run as the {@code langchain4j.StreamingChat}
 * activity.
 *
 * <p>The worker streams the request from the {@code StreamingChatModel} registered under the same
 * name with {@code LangChain4jPlugin} and returns the partial output with the complete response.
 * When the activity completes, the workflow passes each partial response, thinking and tool call to
 * the handler in the order the model produced them, then the complete response, so an AI Service
 * that returns a {@code TokenStream} works unchanged and replays deterministically:
 *
 * <pre>{@code
 * StreamingAssistant assistant =
 *     AiServices.builder(StreamingAssistant.class)
 *         .streamingChatModel(TemporalStreamingChatModel.create("openai"))
 *         .build();
 * }</pre>
 *
 * <p>To show the output while the model is still running, set {@link Builder#streamingTopic} and
 * host a Workflow Stream in the workflow: the activity then also publishes each part to that topic
 * as a {@link ModelStreamEvent}.
 */
public final class TemporalStreamingChatModel implements StreamingChatModel {

  /** The activity options used when none are configured: a two-minute start-to-close timeout. */
  public static final ActivityOptions DEFAULT_ACTIVITY_OPTIONS =
      TemporalChatModel.DEFAULT_ACTIVITY_OPTIONS;

  private final String model;
  private final ActivityOptions activityOptions;
  private final Set<Capability> capabilities;
  private final Function<ChatRequest, String> summary;
  @Nullable private final String streamingTopic;

  private TemporalStreamingChatModel(Builder builder) {
    this.model = builder.model;
    this.activityOptions = builder.activityOptions;
    this.capabilities = Set.copyOf(builder.capabilities);
    this.summary = builder.summary;
    this.streamingTopic = builder.streamingTopic;
  }

  /** Creates a model for the given registration name with the default activity options. */
  public static TemporalStreamingChatModel create(String model) {
    return newBuilder(model).build();
  }

  /** Starts building a model for the given registration name. */
  public static Builder newBuilder(String model) {
    return new Builder(model);
  }

  /** Returns the name the worker's streaming chat model is registered under. */
  public String model() {
    return model;
  }

  /**
   * Runs the request in the activity, then passes its partial output and complete response to the
   * handler on the workflow thread. A failed activity is passed to {@link
   * StreamingChatResponseHandler#onError}. Cancelling the {@link StreamingHandle} given with a
   * partial output stops the remaining callbacks, including the complete response.
   */
  @Override
  public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
    if (!WorkflowUnsafe.isWorkflowThread()) {
      throw new IllegalStateException(
          "TemporalStreamingChatModel '"
              + model
              + "' runs model calls as Temporal activities and can only be used from a workflow;"
              + " use the underlying StreamingChatModel directly outside workflows");
    }
    ActivityOptions options =
        ActivityOptions.newBuilder(activityOptions).setSummary(summary.apply(request)).build();
    StreamingChatModelActivity activity =
        Workflow.newActivityStub(StreamingChatModelActivity.class, options);
    StreamingChatOutput output;
    try {
      output =
          activity.streamingChat(
              new StreamingChatInput(ChatCodec.encode(model, request), streamingTopic));
    } catch (RuntimeException e) {
      handler.onError(e);
      return;
    }
    ReplayHandle handle = new ReplayHandle();
    for (ChatStreamPart part : output.parts()) {
      deliver(part, handler, handle);
      if (handle.isCancelled()) {
        return;
      }
    }
    handler.onCompleteResponse(ChatCodec.decode(output.response()));
  }

  private static void deliver(
      ChatStreamPart part, StreamingChatResponseHandler handler, StreamingHandle handle) {
    switch (part.type()) {
      case PARTIAL_RESPONSE ->
          handler.onPartialResponse(
              new PartialResponse(part.text()), new PartialResponseContext(handle));
      case PARTIAL_THINKING ->
          handler.onPartialThinking(
              new PartialThinking(part.text()), new PartialThinkingContext(handle));
      case PARTIAL_TOOL_CALL ->
          handler.onPartialToolCall(
              PartialToolCall.builder()
                  .index(part.index())
                  .id(part.id())
                  .name(part.name())
                  .partialArguments(part.text())
                  .build(),
              new PartialToolCallContext(handle));
      case COMPLETE_TOOL_CALL ->
          handler.onCompleteToolCall(
              new CompleteToolCall(
                  part.index(),
                  ToolExecutionRequest.builder()
                      .id(part.id())
                      .name(part.name())
                      .arguments(part.text())
                      .build()));
    }
  }

  @Override
  public Set<Capability> supportedCapabilities() {
    return capabilities;
  }

  /**
   * The handle given with replayed partial output. The model has already finished, so cancelling
   * only stops the remaining callbacks.
   */
  private static final class ReplayHandle implements StreamingHandle {
    private boolean cancelled;

    @Override
    public void cancel() {
      cancelled = true;
    }

    @Override
    public boolean isCancelled() {
      return cancelled;
    }
  }

  /** Builder for {@link TemporalStreamingChatModel}. */
  public static final class Builder {
    private final String model;
    private ActivityOptions activityOptions = DEFAULT_ACTIVITY_OPTIONS;
    private final Set<Capability> capabilities = EnumSet.noneOf(Capability.class);
    private Function<ChatRequest, String> summary;
    @Nullable private String streamingTopic;

    private Builder(String model) {
      this.model = Objects.requireNonNull(model, "model");
      if (model.isEmpty()) {
        throw new IllegalArgumentException("model must not be empty");
      }
      this.summary = request -> "StreamingChat: " + model;
    }

    /** Sets the activity options for each model call; a summary set here is overridden. */
    public Builder activityOptions(ActivityOptions activityOptions) {
      this.activityOptions = Objects.requireNonNull(activityOptions, "activityOptions");
      return this;
    }

    /**
     * Declares capabilities of the worker's streaming chat model, so AI Services use them; see
     * {@link TemporalChatModel.Builder#supportedCapabilities}.
     */
    public Builder supportedCapabilities(Capability... capabilities) {
      for (Capability capability : capabilities) {
        this.capabilities.add(Objects.requireNonNull(capability, "capability"));
      }
      return this;
    }

    /** Sets the function that produces the activity summary shown in the Temporal UI. */
    public Builder summary(Function<ChatRequest, String> summary) {
      this.summary = Objects.requireNonNull(summary, "summary");
      return this;
    }

    /**
     * Publishes each partial output and one attempt-terminal record to this Workflow Streams topic
     * while the model activity runs. The workflow must host a {@code WorkflowStream}, and the
     * worker running the activity needs the {@code temporal-workflowstreams} artifact matching its
     * SDK, with the unshaded SDK. A retried activity publishes its output again under the next
     * attempt.
     */
    public Builder streamingTopic(String streamingTopic) {
      Objects.requireNonNull(streamingTopic, "streamingTopic");
      if (streamingTopic.isBlank()) {
        throw new IllegalArgumentException("streamingTopic must not be blank");
      }
      this.streamingTopic = streamingTopic;
      return this;
    }

    /** Builds the model. */
    public TemporalStreamingChatModel build() {
      return new TemporalStreamingChatModel(this);
    }
  }
}
