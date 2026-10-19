package io.temporal.googleadk.activity;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.models.BaseLlm;
import com.google.adk.models.BaseLlmConnection;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.reactivex.rxjava3.core.Flowable;
import io.temporal.activity.ActivityOptions;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.failure.ActivityFailure;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.converter.GoogleAdkDataConverter;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.testing.TestActivityEnvironment;
import io.temporal.testing.TestEnvironmentOptions;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** The model activity resolves models once per name and fails with typed, classified errors. */
class ModelActivityImplTest {

  private static final LlmRequest REQUEST =
      LlmRequest.builder()
          .model("scripted")
          .contents(List.of(Content.fromParts(Part.fromText("hi"))))
          .build();

  private TestActivityEnvironment env;
  private final AtomicInteger factoryCalls = new AtomicInteger();

  @BeforeEach
  void setUp() {
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
              if (name.equals("broken")) {
                throw new IllegalArgumentException("no credentials for " + name);
              }
              if (name.equals("hanging")) {
                return new HangingModel();
              }
              return ScriptedModel.of(name, ScriptedModel.text("reply from " + name));
            }));
  }

  @AfterEach
  void tearDown() {
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

  @Test
  void invokeModel_abandonsTheCallWhenTheActivityIsCancelled() throws Exception {
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
                activity.invokeModel(new InvokeModelInput("hanging", REQUEST, false));
              } catch (Throwable t) {
                outcome.set(t);
              }
            });
    call.start();
    Thread.sleep(500);

    env.requestCancelActivity();
    call.join(10_000);

    assertFalse(call.isAlive(), "the activity returns although the model call is still blocked");
    assertTrue(outcome.get() != null, "cancellation surfaces as a failure");
  }

  /** A model whose call blocks the subscribing thread, as a real HTTP client does. */
  static final class HangingModel extends BaseLlm {
    HangingModel() {
      super("hanging");
    }

    @Override
    public Flowable<LlmResponse> generateContent(LlmRequest llmRequest, boolean stream) {
      return Flowable.fromCallable(
          () -> {
            Thread.sleep(60_000);
            return ScriptedModel.text("too late");
          });
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
