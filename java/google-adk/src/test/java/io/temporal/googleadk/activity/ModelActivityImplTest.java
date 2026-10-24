package io.temporal.googleadk.activity;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.models.BaseLlm;
import com.google.adk.models.BaseLlmConnection;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.tools.BaseTool;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionDeclaration;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.Part;
import com.google.genai.types.Schema;
import com.google.genai.types.Tool;
import com.google.genai.types.Type;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.functions.Consumer;
import io.reactivex.rxjava3.plugins.RxJavaPlugins;
import io.temporal.activity.ActivityOptions;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.failure.ActivityFailure;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.converter.GoogleAdkDataConverter;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.testing.TestActivityEnvironment;
import io.temporal.testing.TestEnvironmentOptions;
import java.io.IOException;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** The model activity resolves models once per name and fails with typed, classified errors. */
class ModelActivityImplTest {

  private static final LlmRequest REQUEST =
      LlmRequest.builder()
          .model("scripted")
          .contents(List.of(Content.fromParts(Part.fromText("hi"))))
          .build();

  /** Declarations as ADK's tools build them: one with a schema, one with a JSON schema. */
  private static final FunctionDeclaration LOOKUP =
      FunctionDeclaration.builder()
          .name("lookup")
          .description("Looks up a key.")
          .parameters(
              Schema.builder()
                  .type(Type.Known.OBJECT)
                  .properties(Map.of("key", Schema.builder().type(Type.Known.STRING).build()))
                  .required(List.of("key"))
                  .build())
          .build();

  private static final FunctionDeclaration SEARCH =
      FunctionDeclaration.builder()
          .name("search")
          .description("Searches the index.")
          .parametersJsonSchema(
              Map.of("type", "object", "properties", Map.of("query", Map.of("type", "string"))))
          .build();

  private static final LlmRequest REQUEST_WITH_TOOLS =
      REQUEST.toBuilder()
          .config(
              GenerateContentConfig.builder()
                  .tools(
                      List.of(Tool.builder().functionDeclarations(List.of(LOOKUP, SEARCH)).build()))
                  .build())
          .build();

  private TestActivityEnvironment env;
  private final AtomicInteger factoryCalls = new AtomicInteger();
  private final CountDownLatch release = new CountDownLatch(1);
  private final CountDownLatch blocked = new CountDownLatch(1);
  private static final AtomicBoolean interrupted = new AtomicBoolean();
  private static final AtomicBoolean finished = new AtomicBoolean();
  private final ScriptedModel recording = ScriptedModel.of("recording", ScriptedModel.text("ok"));

  @BeforeEach
  void setUp() {
    interrupted.set(false);
    finished.set(false);
    env =
        TestActivityEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder()
                        .setDataConverter(GoogleAdkDataConverter.newDefaultInstance())
                        .build())
                .build());
    env.registerActivitiesImplementations(
        new ModelActivityImpl(
            name -> {
              factoryCalls.incrementAndGet();
              switch (name) {
                case "broken":
                  throw new IllegalArgumentException("no credentials for " + name);
                case "hanging":
                  return new HangingModel(blocked, release);
                case "blocking":
                  return new BlockingModel(blocked, release);
                case "failing":
                  return new FailingModel();
                case "recording":
                  return recording;
                default:
                  return ScriptedModel.of(name, ScriptedModel.text("reply from " + name));
              }
            }));
  }

  @AfterEach
  void tearDown() {
    release.countDown();
    env.close();
  }

  @Test
  void invokeModel_returnsTheModelResponsesAndCachesTheModel() {
    ModelActivity activity = env.newActivityStub(ModelActivity.class);

    InvokeModelOutput first =
        activity.invokeModel(new InvokeModelInput("scripted", REQUEST, false));
    InvokeModelOutput second =
        activity.invokeModel(new InvokeModelInput("scripted", REQUEST, false));

    assertEquals(1, first.responses().size());
    assertEquals("reply from scripted", first.responses().get(0).content().get().text());
    assertEquals(first.responses(), second.responses());
    assertEquals(1, factoryCalls.get(), "the factory runs once per model name");
  }

  @Test
  void invokeModel_restoresTheToolsFromTheRequestDeclarations() {
    ModelActivity activity = env.newActivityStub(ModelActivity.class);

    activity.invokeModel(new InvokeModelInput("recording", REQUEST_WITH_TOOLS, false));

    LlmRequest received = recording.requests().get(0);
    assertEquals(
        List.of("lookup", "search"),
        List.copyOf(received.tools().keySet()),
        "the tool map, which the request's JSON omits, is rebuilt from the declarations");
    BaseTool lookup = received.tools().get("lookup");
    BaseTool search = received.tools().get("search");
    assertEquals(LOOKUP, lookup.declaration().orElseThrow());
    assertEquals(SEARCH, search.declaration().orElseThrow());
    assertEquals("Looks up a key.", lookup.description());
    Map<String, Object> args = Map.of("key", "k1");
    assertThrows(
        UnsupportedOperationException.class,
        () -> lookup.runAsync(args, null),
        "a restored tool runs in the workflow, never on the worker");
  }

  @Test
  void withDeclaredTools_keepsRequestsThatCarryToolsOrDeclareNone() {
    LlmRequest withLiveTool =
        REQUEST_WITH_TOOLS.toBuilder()
            .tools(Map.of("lookup", new BaseTool("lookup", "A live tool.") {}))
            .build();

    assertSame(withLiveTool, ModelActivityImpl.withDeclaredTools(withLiveTool));
    assertSame(REQUEST, ModelActivityImpl.withDeclaredTools(REQUEST));
  }

  @Test
  void invokeModel_withoutModelName_failsNonRetryably() {
    ModelActivity activity = env.newActivityStub(ModelActivity.class);

    ApplicationFailure failure =
        applicationFailure(() -> activity.invokeModel(new InvokeModelInput("", REQUEST, false)));

    assertEquals("googleadk.ModelError", failure.getType());
    assertTrue(failure.isNonRetryable());
  }

  @Test
  void invokeModel_whenTheFactoryFails_failsNonRetryably() {
    ModelActivity activity = env.newActivityStub(ModelActivity.class);

    ApplicationFailure failure =
        applicationFailure(
            () -> activity.invokeModel(new InvokeModelInput("broken", REQUEST, false)));

    assertEquals("googleadk.ModelError", failure.getType());
    assertTrue(failure.isNonRetryable());
    assertTrue(failure.getOriginalMessage().contains("no credentials for broken"));
  }

  /**
   * A failure a model delivers from a future, as ADK's Gemini model does, keeps its own message.
   */
  @Test
  void invokeModel_reportsACheckedFailureByItsOwnMessage() {
    ModelActivity activity = env.newActivityStub(ModelActivity.class);

    ApplicationFailure failure =
        applicationFailure(
            () -> activity.invokeModel(new InvokeModelInput("failing", REQUEST, false)));

    assertEquals("googleadk.ModelError", failure.getType());
    assertFalse(failure.isNonRetryable());
    assertTrue(
        failure.getOriginalMessage().endsWith("failed: reset"), failure.getOriginalMessage());
  }

  /**
   * Whether the model blocks inside its flow or before it even returns one, cancelling the activity
   * releases the activity thread while the model call is still blocked; the call is left to finish
   * on its own, and its late result reaches no error handler.
   */
  @ParameterizedTest(name = "{0}")
  @ValueSource(strings = {"hanging", "blocking"})
  void invokeModel_abandonsTheCallWhenTheActivityIsCancelled(String model) throws Exception {
    ModelActivity activity =
        env.newActivityStub(
            ModelActivity.class,
            ActivityOptions.newBuilder()
                .setStartToCloseTimeout(Duration.ofSeconds(30))
                .setHeartbeatTimeout(Duration.ofMillis(300))
                .build());
    AtomicReference<Throwable> outcome = new AtomicReference<>();
    Thread call =
        new Thread(
            () -> {
              try {
                activity.invokeModel(new InvokeModelInput(model, REQUEST, false));
              } catch (Throwable t) {
                outcome.set(t);
              }
            });
    call.start();
    assertTrue(blocked.await(10, TimeUnit.SECONDS), "the model must be blocked before the cancel");
    List<Throwable> uncaught = new CopyOnWriteArrayList<>();
    Thread.UncaughtExceptionHandler previousHandler = Thread.getDefaultUncaughtExceptionHandler();
    Consumer<? super Throwable> previousRxHandler = RxJavaPlugins.getErrorHandler();
    Thread.setDefaultUncaughtExceptionHandler((thread, error) -> uncaught.add(error));
    RxJavaPlugins.setErrorHandler(uncaught::add);
    try {
      env.requestCancelActivity();
      call.join(10_000);

      assertFalse(call.isAlive(), "the activity returns although the model call is still blocked");
      assertNotNull(outcome.get(), "cancellation surfaces as a failure");
      assertFalse(finished.get(), "the model call is still blocked when the activity returns");
      release.countDown();
      for (int i = 0; i < 100 && !finished.get(); i++) {
        Thread.sleep(50);
      }
      assertTrue(finished.get(), "the abandoned model call finishes on its own");
      assertFalse(interrupted.get(), "the abandoned model call is not interrupted");
      Thread.sleep(200);
      assertTrue(uncaught.isEmpty(), "no error escapes to the uncaught handler: " + uncaught);
    } finally {
      Thread.setDefaultUncaughtExceptionHandler(previousHandler);
      RxJavaPlugins.setErrorHandler(previousRxHandler);
    }
  }

  /** Blocks the calling thread until the test releases it or interrupts it. */
  private static void block(CountDownLatch release) {
    try {
      if (!release.await(60, TimeUnit.SECONDS)) {
        throw new IllegalStateException("the model was never released");
      }
      finished.set(true);
    } catch (InterruptedException e) {
      interrupted.set(true);
      Thread.currentThread().interrupt();
      throw new IllegalStateException(e);
    }
  }

  /** A model whose flow blocks the subscribing thread, as a streaming HTTP client does. */
  static final class HangingModel extends BaseLlm {
    private final CountDownLatch blocked;
    private final CountDownLatch release;

    HangingModel(CountDownLatch blocked, CountDownLatch release) {
      super("hanging");
      this.blocked = blocked;
      this.release = release;
    }

    @Override
    public Flowable<LlmResponse> generateContent(LlmRequest llmRequest, boolean stream) {
      return Flowable.fromCallable(
          () -> {
            blocked.countDown();
            block(release);
            return ScriptedModel.text("too late");
          });
    }

    @Override
    public BaseLlmConnection connect(LlmRequest llmRequest) {
      throw new UnsupportedOperationException();
    }
  }

  /** A model that makes its blocking call before returning a flow, as ADK's Claude model does. */
  static final class BlockingModel extends BaseLlm {
    private final CountDownLatch blocked;
    private final CountDownLatch release;

    BlockingModel(CountDownLatch blocked, CountDownLatch release) {
      super("blocking");
      this.blocked = blocked;
      this.release = release;
    }

    @Override
    public Flowable<LlmResponse> generateContent(LlmRequest llmRequest, boolean stream) {
      blocked.countDown();
      block(release);
      return Flowable.just(ScriptedModel.text("too late"));
    }

    @Override
    public BaseLlmConnection connect(LlmRequest llmRequest) {
      throw new UnsupportedOperationException();
    }
  }

  /** A model whose call fails from a future, the way ADK's Gemini model delivers its failures. */
  static final class FailingModel extends BaseLlm {
    FailingModel() {
      super("failing");
    }

    @Override
    public Flowable<LlmResponse> generateContent(LlmRequest llmRequest, boolean stream) {
      return Flowable.fromFuture(CompletableFuture.failedFuture(new IOException("reset")));
    }

    @Override
    public BaseLlmConnection connect(LlmRequest llmRequest) {
      throw new UnsupportedOperationException();
    }
  }

  private static ApplicationFailure applicationFailure(Runnable call) {
    ActivityFailure failure = assertThrows(ActivityFailure.class, call::run);
    assertInstanceOf(ApplicationFailure.class, failure.getCause());
    return (ApplicationFailure) failure.getCause();
  }
}
