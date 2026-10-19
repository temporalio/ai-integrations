package io.temporal.googleadk.testing;

import com.google.adk.models.BaseLlm;
import com.google.adk.models.BaseLlmConnection;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionCall;
import com.google.genai.types.Part;
import io.reactivex.rxjava3.core.Flowable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

/**
 * A deterministic ADK model for tests. Register it on the worker with {@code
 * GoogleAdkPlugin.newBuilder().model(name, scripted)} and give the workflow's agents a {@code
 * TemporalModel} of the same name.
 *
 * <p>{@link #of} replays a fixed sequence of responses, one per call, repeating the last one;
 * {@link #responding} computes each response from the request. Every request is recorded.
 */
public final class ScriptedModel extends BaseLlm {

  private final Function<LlmRequest, List<LlmResponse>> script;
  private final List<LlmRequest> requests = new CopyOnWriteArrayList<>();

  private ScriptedModel(String modelName, Function<LlmRequest, List<LlmResponse>> script) {
    super(modelName);
    this.script = script;
  }

  /** Creates a model that returns the given responses in order, repeating the last one. */
  public static ScriptedModel of(String modelName, LlmResponse... responses) {
    if (responses.length == 0) {
      throw new IllegalArgumentException("at least one response is required");
    }
    List<LlmResponse> sequence = List.of(responses);
    AtomicInteger next = new AtomicInteger();
    return new ScriptedModel(
        modelName,
        request -> {
          int index = Math.min(next.getAndIncrement(), sequence.size() - 1);
          return List.of(sequence.get(index));
        });
  }

  /** Creates a model that computes one response per request. */
  public static ScriptedModel responding(
      String modelName, Function<LlmRequest, LlmResponse> responder) {
    Objects.requireNonNull(responder, "responder");
    return new ScriptedModel(modelName, request -> List.of(responder.apply(request)));
  }

  /** Creates a model that computes a list of responses per request, for streamed replies. */
  public static ScriptedModel streaming(
      String modelName, Function<LlmRequest, List<LlmResponse>> responder) {
    Objects.requireNonNull(responder, "responder");
    return new ScriptedModel(modelName, responder);
  }

  /** Returns a final text response. */
  public static LlmResponse text(String text) {
    return LlmResponse.builder()
        .content(Content.builder().role("model").parts(List.of(Part.fromText(text))).build())
        .turnComplete(true)
        .build();
  }

  /** Returns a response that calls the named function with the given arguments. */
  public static LlmResponse functionCall(String name, Map<String, Object> args) {
    return LlmResponse.builder()
        .content(
            Content.builder()
                .role("model")
                .parts(
                    List.of(
                        Part.builder()
                            .functionCall(FunctionCall.builder().name(name).args(args).build())
                            .build()))
                .build())
        .build();
  }

  /** Returns a partial (streamed) text chunk. */
  public static LlmResponse partialText(String text) {
    return LlmResponse.builder()
        .content(Content.builder().role("model").parts(List.of(Part.fromText(text))).build())
        .partial(true)
        .build();
  }

  /** Returns the requests received so far, in order. */
  public List<LlmRequest> requests() {
    return Collections.unmodifiableList(new ArrayList<>(requests));
  }

  @Override
  public Flowable<LlmResponse> generateContent(LlmRequest llmRequest, boolean stream) {
    requests.add(llmRequest);
    return Flowable.fromIterable(script.apply(llmRequest));
  }

  @Override
  public BaseLlmConnection connect(LlmRequest llmRequest) {
    throw new UnsupportedOperationException("ScriptedModel does not support live connections");
  }
}
