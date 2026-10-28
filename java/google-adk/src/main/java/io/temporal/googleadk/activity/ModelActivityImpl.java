package io.temporal.googleadk.activity;

import com.google.adk.models.BaseLlm;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.tools.BaseTool;
import com.google.adk.tools.ToolContext;
import com.google.genai.types.FunctionDeclaration;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.Tool;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.processors.PublishProcessor;
import io.temporal.activity.Activity;
import io.temporal.activity.ActivityExecutionContext;
import io.temporal.client.ActivityCompletionException;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.FailureTypes;
import io.temporal.googleadk.internal.BackgroundCall;
import io.temporal.googleadk.internal.Failures;
import io.temporal.googleadk.internal.Heartbeats;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
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
    LlmRequest request = withDeclaredTools(input.request());
    ActivityExecutionContext context = Activity.getExecutionContext();
    // Completing this processor abandons the model call once the server cancels the activity. The
    // whole call runs on an I/O thread, including any blocking work the model does before it
    // returns a flow, so that abandoning it releases the activity thread at once; the call itself
    // finishes on its own there, and its result is dropped.
    PublishProcessor<Object> abandon = PublishProcessor.create();
    try (Heartbeats heartbeats = Heartbeats.start(context, abandon::onComplete)) {
      List<LlmResponse> responses =
          BackgroundCall.collect(
              () -> model.generateContent(request, input.stream()), abandon, () -> {});
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

  /**
   * Restores the request's tool map from the function declarations in its config. The live tool
   * instances stay in the workflow and are not part of the request's JSON, so the map arrives
   * empty; Gemini reads the declarations from the config, but ADK's LangChain4j and Spring AI
   * models build their tool lists from the map, and its Claude model sends tools and a tool choice
   * only when the map is not empty. A request that still carries tools, or declares none, is
   * returned unchanged.
   */
  static LlmRequest withDeclaredTools(LlmRequest request) {
    if (!request.tools().isEmpty()) {
      return request;
    }
    Map<String, BaseTool> tools = new LinkedHashMap<>();
    for (Tool tool : request.config().flatMap(GenerateContentConfig::tools).orElse(List.of())) {
      for (FunctionDeclaration declaration : tool.functionDeclarations().orElse(List.of())) {
        declaration
            .name()
            .ifPresent(name -> tools.putIfAbsent(name, new DeclaredTool(name, declaration)));
      }
    }
    if (tools.isEmpty()) {
      return request;
    }
    return request.toBuilder().tools(Collections.unmodifiableMap(tools)).build();
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

  /**
   * A tool the worker knows only by its declaration, with or without a parameter schema. It
   * describes itself to the model; the call itself runs in the workflow, never here.
   */
  private static final class DeclaredTool extends BaseTool {
    private final FunctionDeclaration declaration;

    DeclaredTool(String name, FunctionDeclaration declaration) {
      super(name, declaration.description().orElse(""));
      this.declaration = declaration;
    }

    @Override
    public Optional<FunctionDeclaration> declaration() {
      return Optional.of(declaration);
    }

    @Override
    public Single<Map<String, Object>> runAsync(Map<String, Object> args, ToolContext toolContext) {
      throw new UnsupportedOperationException(
          "Tool '" + name() + "' runs in the workflow, not on the worker that invokes the model");
    }
  }
}
