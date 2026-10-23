package io.temporal.googleadk.model;

import com.google.adk.models.BaseLlm;
import com.google.adk.models.BaseLlmConnection;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import io.reactivex.rxjava3.core.Flowable;
import io.temporal.activity.ActivityOptions;
import io.temporal.googleadk.activity.InvokeModelInput;
import io.temporal.googleadk.activity.InvokeModelOutput;
import io.temporal.googleadk.activity.ModelActivity;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.unsafe.WorkflowUnsafe;
import java.time.Duration;
import java.util.Objects;
import java.util.function.Function;

/**
 * An ADK model whose calls run as the {@code googleadk.InvokeModel} activity.
 *
 * <p>Give an {@code LlmAgent} a {@code TemporalModel} instead of a model name and every model call
 * the agent makes inside a workflow becomes a durable activity. The workflow side only carries the
 * model name; the worker resolves it through the factory configured on {@code GoogleAdkPlugin}, so
 * credentials stay on the worker.
 *
 * <pre>{@code
 * LlmAgent agent =
 *     LlmAgent.builder()
 *         .name("assistant")
 *         .model(TemporalModel.create("gemini-2.5-flash"))
 *         .instruction("Answer concisely.")
 *         .build();
 * }</pre>
 *
 * <p>Streamed calls ({@code RunConfig.StreamingMode.SSE}) are served by the same activity: the
 * worker collects every chunk the model emits and the workflow replays them in order, so ADK's
 * partial events keep their shape. Chunks reach the workflow only when the activity completes. Live
 * (bidirectional) connections are not supported inside workflows.
 */
public final class TemporalModel extends BaseLlm {

  /** The activity options used when none are configured: a two-minute start-to-close timeout. */
  public static final ActivityOptions DEFAULT_ACTIVITY_OPTIONS =
      ActivityOptions.newBuilder().setStartToCloseTimeout(Duration.ofMinutes(2)).build();

  private final ActivityOptions activityOptions;
  private final Function<LlmRequest, String> summary;

  private TemporalModel(Builder builder) {
    super(builder.modelName);
    this.activityOptions = builder.activityOptions;
    this.summary = builder.summary;
  }

  /** Creates a model with the default activity options. */
  public static TemporalModel create(String modelName) {
    return newBuilder(modelName).build();
  }

  /** Starts building a model for the given name. */
  public static Builder newBuilder(String modelName) {
    return new Builder(modelName);
  }

  @Override
  public Flowable<LlmResponse> generateContent(LlmRequest llmRequest, boolean stream) {
    return Flowable.defer(
        () -> {
          if (!WorkflowUnsafe.isWorkflowThread()) {
            throw new IllegalStateException(
                "TemporalModel '"
                    + model()
                    + "' runs model calls as Temporal activities and can only be used from a"
                    + " workflow; use the underlying model directly outside workflows");
          }
          ActivityOptions options =
              ActivityOptions.newBuilder(activityOptions)
                  .setSummary(summary.apply(llmRequest))
                  .build();
          ModelActivity activity = Workflow.newActivityStub(ModelActivity.class, options);
          InvokeModelOutput output =
              activity.invokeModel(new InvokeModelInput(model(), llmRequest, stream));
          return Flowable.fromIterable(output.responses());
        });
  }

  @Override
  public BaseLlmConnection connect(LlmRequest llmRequest) {
    throw new UnsupportedOperationException(
        "Live connections are not supported inside Temporal workflows");
  }

  /** Builder for {@link TemporalModel}. */
  public static final class Builder {
    private final String modelName;
    private ActivityOptions activityOptions = DEFAULT_ACTIVITY_OPTIONS;
    private Function<LlmRequest, String> summary;

    private Builder(String modelName) {
      this.modelName = Objects.requireNonNull(modelName, "modelName");
      if (modelName.isEmpty()) {
        throw new IllegalArgumentException("modelName must not be empty");
      }
      this.summary = request -> "InvokeModel: " + modelName;
    }

    /** Sets the activity options for each model call; a summary set here is overridden. */
    public Builder activityOptions(ActivityOptions activityOptions) {
      this.activityOptions = Objects.requireNonNull(activityOptions, "activityOptions");
      return this;
    }

    /** Sets the function that produces the activity summary shown in the Temporal UI. */
    public Builder summary(Function<LlmRequest, String> summary) {
      this.summary = Objects.requireNonNull(summary, "summary");
      return this;
    }

    /** Builds the model. */
    public TemporalModel build() {
      return new TemporalModel(this);
    }
  }
}
