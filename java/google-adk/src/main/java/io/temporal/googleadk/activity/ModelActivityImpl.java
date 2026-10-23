package io.temporal.googleadk.activity;

import com.google.adk.models.BaseLlm;
import com.google.adk.models.LlmResponse;
import io.reactivex.rxjava3.processors.PublishProcessor;
import io.reactivex.rxjava3.schedulers.Schedulers;
import io.temporal.activity.Activity;
import io.temporal.activity.ActivityExecutionContext;
import io.temporal.client.ActivityCompletionException;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.FailureTypes;
import io.temporal.googleadk.internal.Failures;
import io.temporal.googleadk.internal.Heartbeats;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * Worker-side implementation of {@link ModelActivity}. Models are created once per name by the
 * configured factory and cached for the life of the worker; credentials never cross the activity
 * boundary because the factory runs on the worker.
 */
public final class ModelActivityImpl implements ModelActivity {

  private final Function<String, BaseLlm> modelFactory;
  private final Map<String, BaseLlm> models = new ConcurrentHashMap<>();

  /**
   * Creates the activity implementation.
   *
   * @param modelFactory resolves a model name to the model that serves it; invoked at most once per
   *     name
   */
  public ModelActivityImpl(Function<String, BaseLlm> modelFactory) {
    this.modelFactory = Objects.requireNonNull(modelFactory, "modelFactory");
  }

  @Override
  public InvokeModelOutput invokeModel(InvokeModelInput input) {
    if (input == null || input.request() == null) {
      throw ApplicationFailure.newNonRetryableFailure(
          "InvokeModel requires a request", FailureTypes.MODEL_ERROR);
    }
    BaseLlm model = resolve(input.model());
    ActivityExecutionContext context = Activity.getExecutionContext();
    // Completing this processor abandons the model call once the server cancels the activity. The
    // call runs on an I/O thread so that abandoning it releases the activity thread at once; the
    // underlying request still runs to completion there.
    PublishProcessor<Object> abandon = PublishProcessor.create();
    try (Heartbeats heartbeats = Heartbeats.start(context, abandon::onComplete)) {
      List<LlmResponse> responses =
          model
              .generateContent(input.request(), input.stream())
              .subscribeOn(Schedulers.io())
              .takeUntil(abandon)
              .toList()
              .blockingGet();
      ActivityCompletionException completion = heartbeats.completion();
      if (completion != null) {
        throw completion;
      }
      if (responses.isEmpty()) {
        throw ApplicationFailure.newFailure(
            "Model " + input.model() + " returned no response", FailureTypes.MODEL_ERROR);
      }
      return new InvokeModelOutput(responses);
    } catch (ApplicationFailure | ActivityCompletionException e) {
      throw e;
    } catch (RuntimeException e) {
      throw Failures.modelFailure(input.model(), e);
    }
  }

  private BaseLlm resolve(String modelName) {
    if (modelName == null || modelName.isEmpty()) {
      throw ApplicationFailure.newNonRetryableFailure(
          "InvokeModel requires a model name", FailureTypes.MODEL_ERROR);
    }
    try {
      BaseLlm model = models.computeIfAbsent(modelName, modelFactory);
      if (model == null) {
        throw ApplicationFailure.newNonRetryableFailure(
            "No model is registered for '" + modelName + "'", FailureTypes.MODEL_ERROR);
      }
      return model;
    } catch (ApplicationFailure e) {
      throw e;
    } catch (RuntimeException e) {
      throw ApplicationFailure.newNonRetryableFailureWithCause(
          "Could not create model '" + modelName + "': " + e.getMessage(),
          FailureTypes.MODEL_ERROR,
          e);
    }
  }
}
