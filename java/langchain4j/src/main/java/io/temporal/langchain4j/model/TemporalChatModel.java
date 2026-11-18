package io.temporal.langchain4j.model;

import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import io.temporal.activity.ActivityOptions;
import io.temporal.langchain4j.activity.ChatModelActivity;
import io.temporal.langchain4j.activity.ChatOutput;
import io.temporal.langchain4j.internal.ChatCodec;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.unsafe.WorkflowUnsafe;
import java.time.Duration;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

/**
 * A LangChain4j {@link ChatModel} whose calls run as the {@code langchain4j.Chat} activity.
 *
 * <p>Build an AI Service inside a workflow with a {@code TemporalChatModel} and every model call it
 * makes becomes a durable activity. The workflow only carries the registration name; the worker
 * runs the request against the {@code ChatModel} registered under that name with {@code
 * LangChain4jPlugin}, so credentials and provider clients stay on the worker.
 *
 * <pre>{@code
 * Assistant assistant =
 *     AiServices.builder(Assistant.class)
 *         .chatModel(TemporalChatModel.create("openai"))
 *         .tools(new WorkflowTools())
 *         .chatMemory(MessageWindowChatMemory.withMaxMessages(20))
 *         .build();
 * }</pre>
 */
public final class TemporalChatModel implements ChatModel {

  /** The activity options used when none are configured: a two-minute start-to-close timeout. */
  public static final ActivityOptions DEFAULT_ACTIVITY_OPTIONS =
      ActivityOptions.newBuilder().setStartToCloseTimeout(Duration.ofMinutes(2)).build();

  private final String model;
  private final ActivityOptions activityOptions;
  private final Set<Capability> capabilities;
  private final Function<ChatRequest, String> summary;

  private TemporalChatModel(Builder builder) {
    this.model = builder.model;
    this.activityOptions = builder.activityOptions;
    this.capabilities = Set.copyOf(builder.capabilities);
    this.summary = builder.summary;
  }

  /** Creates a model for the given registration name with the default activity options. */
  public static TemporalChatModel create(String model) {
    return newBuilder(model).build();
  }

  /** Starts building a model for the given registration name. */
  public static Builder newBuilder(String model) {
    return new Builder(model);
  }

  /** Returns the name the worker's chat model is registered under. */
  public String model() {
    return model;
  }

  @Override
  public ChatResponse doChat(ChatRequest request) {
    if (!WorkflowUnsafe.isWorkflowThread()) {
      throw new IllegalStateException(
          "TemporalChatModel '"
              + model
              + "' runs model calls as Temporal activities and can only be used from a workflow;"
              + " use the underlying ChatModel directly outside workflows");
    }
    ActivityOptions options =
        ActivityOptions.newBuilder(activityOptions).setSummary(summary.apply(request)).build();
    ChatModelActivity activity = Workflow.newActivityStub(ChatModelActivity.class, options);
    ChatOutput output = activity.chat(ChatCodec.encode(model, request));
    return ChatCodec.decode(output);
  }

  @Override
  public Set<Capability> supportedCapabilities() {
    return capabilities;
  }

  /** Builder for {@link TemporalChatModel}. */
  public static final class Builder {
    private final String model;
    private ActivityOptions activityOptions = DEFAULT_ACTIVITY_OPTIONS;
    private final Set<Capability> capabilities = EnumSet.noneOf(Capability.class);
    private Function<ChatRequest, String> summary;

    private Builder(String model) {
      this.model = Objects.requireNonNull(model, "model");
      if (model.isEmpty()) {
        throw new IllegalArgumentException("model must not be empty");
      }
      this.summary = request -> "Chat: " + model;
    }

    /** Sets the activity options for each model call; a summary set here is overridden. */
    public Builder activityOptions(ActivityOptions activityOptions) {
      this.activityOptions = Objects.requireNonNull(activityOptions, "activityOptions");
      return this;
    }

    /**
     * Declares capabilities of the worker's chat model, so AI Services use them. Declare {@link
     * Capability#RESPONSE_FORMAT_JSON_SCHEMA} when the registered model supports JSON schema
     * responses; an AI Service then requests structured output with a schema.
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

    /** Builds the model. */
    public TemporalChatModel build() {
      return new TemporalChatModel(this);
    }
  }
}
