package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.*;

import com.google.adk.agents.LlmAgent;
import com.google.adk.agents.RunConfig;
import com.google.adk.models.BaseLlm;
import com.google.adk.models.BaseLlmConnection;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import com.google.protobuf.ByteString;
import io.reactivex.rxjava3.core.BackpressureStrategy;
import io.reactivex.rxjava3.core.Flowable;
import io.temporal.activity.ActivityCancellationType;
import io.temporal.activity.ActivityOptions;
import io.temporal.api.common.v1.Payload;
import io.temporal.api.history.v1.HistoryEvent;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowOptions;
import io.temporal.client.WorkflowStub;
import io.temporal.common.RetryOptions;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.common.converter.CodecDataConverter;
import io.temporal.common.interceptors.WorkflowClientCallsInterceptor;
import io.temporal.common.interceptors.WorkflowClientCallsInterceptorBase;
import io.temporal.common.interceptors.WorkflowClientInterceptor;
import io.temporal.common.interceptors.WorkflowClientInterceptorBase;
import io.temporal.failure.ActivityFailure;
import io.temporal.failure.ApplicationFailure;
import io.temporal.failure.CanceledFailure;
import io.temporal.googleadk.activity.InvokeModelInput;
import io.temporal.googleadk.converter.GoogleAdkDataConverter;
import io.temporal.googleadk.converter.GoogleAdkPayloadConverter;
import io.temporal.googleadk.model.ModelStreamEvent;
import io.temporal.googleadk.model.TemporalModel;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.payload.codec.PayloadCodec;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.SignalMethod;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.WorkflowInit;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import io.temporal.workflowstreams.SubscribeOptions;
import io.temporal.workflowstreams.WorkflowStream;
import io.temporal.workflowstreams.WorkflowStreamClient;
import io.temporal.workflowstreams.WorkflowStreamConstants;
import io.temporal.workflowstreams.WorkflowStreamListener;
import io.temporal.workflowstreams.WorkflowStreamSubscriptionHandle;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

@Timeout(45)
class LiveStreamingWorkflowTest {
  private static final String MODEL = "scripted";
  private static final String TOPIC = "model-output";

  @Test
  void deliversBeforeTheModelActivityCompletesAndReplaysWithoutPublishing() throws Exception {
    try (Fixture fixture = new Fixture(LiveWorkflowImpl.class)) {
      fixture.start(true, true);
      try (Subscriber subscriber = fixture.subscribe()) {
        ModelStreamEvent first = subscriber.next();
        assertEquals(ModelStreamEvent.Kind.RESPONSE, first.kind());
        assertEquals("Hel", first.response().content().orElseThrow().text());
        assertTrue(first.response().partial().orElse(false));
        assertEquals(0, first.sequence());
        assertEquals(1, first.attempt());
        assertEquals(fixture.id, first.workflowId());
        assertFalse(first.runId().isEmpty());
        assertFalse(first.activityId().isEmpty());
        assertEquals(1, fixture.model.finished.getCount(), "provider remains blocked");
        assertFalse(fixture.result().isDone(), "Activity result has not been delivered");

        fixture.model.release.countDown();
        ModelStreamEvent last = subscriber.next();
        ModelStreamEvent complete = subscriber.next();
        assertEquals("Hello", last.response().content().orElseThrow().text());
        assertEquals(ModelStreamEvent.Kind.COMPLETE, complete.kind());
        assertNull(complete.response());
        assertEquals(
            List.of(0L, 1L, 2L), List.of(first.sequence(), last.sequence(), complete.sequence()));
        assertEquals(first.activityId(), complete.activityId());
        assertEquals(first.runId(), complete.runId());
        subscriber.awaitTurnComplete();
        fixture.workflow.finish();
        assertEquals("Hello", fixture.result().get(10, TimeUnit.SECONDS));
        fixture.replay();
        assertEquals(1, fixture.model.calls.get());
      }
    }
  }

  @Test
  void completesWithoutASubscriber() throws Exception {
    try (Fixture fixture = new Fixture(LiveWorkflowImpl.class)) {
      fixture.model.release.countDown();
      fixture.start(true, true);
      assertEquals("Hello", fixture.result().get(10, TimeUnit.SECONDS));
      fixture.replay();
    }
  }

  @Test
  void preservesTheActivityClientsCodecOnSignalEnvelopes() throws Exception {
    StrictCodec codec = new StrictCodec();
    try (Fixture fixture = new Fixture(LiveWorkflowImpl.class, codec)) {
      fixture.start(true, true);
      try (Subscriber subscriber = fixture.subscribe()) {
        assertEquals(0, subscriber.next().sequence());
        fixture.model.release.countDown();
        assertEquals(1, subscriber.next().sequence());
        assertEquals(ModelStreamEvent.Kind.COMPLETE, subscriber.next().kind());
        fixture.workflow.finish();
        assertEquals("Hello", fixture.result().get(10, TimeUnit.SECONDS));
        assertTrue(codec.encoded.get() > 0);
        assertTrue(codec.decoded.get() > 0);
        fixture.replay();
      }
    }
  }

  @Test
  void disposingTheSubscriberDoesNotCancelOrBlockTheModel() throws Exception {
    try (Fixture fixture = new Fixture(LiveWorkflowImpl.class)) {
      fixture.start(true, true);
      try (Subscriber subscriber = fixture.subscribe()) {
        assertEquals(0, subscriber.next().sequence());
      }
      fixture.model.release.countDown();
      assertEquals("Hello", fixture.result().get(10, TimeUnit.SECONDS));
    }
  }

  @Test
  void retriesIdentifyEachAttemptAndRestartTheSequence() throws Exception {
    try (Fixture fixture = new Fixture(LiveWorkflowImpl.class)) {
      fixture.model.failFirst = true;
      fixture.start(true, true);
      try (Subscriber subscriber = fixture.subscribe()) {
        ModelStreamEvent first = subscriber.next();
        fixture.model.release.countDown();
        ModelStreamEvent failed = subscriber.next();
        ModelStreamEvent retry = subscriber.next();
        ModelStreamEvent last = subscriber.next();
        ModelStreamEvent complete = subscriber.next();
        assertEquals(ModelStreamEvent.Kind.FAILED, failed.kind());
        assertEquals(List.of(0L, 1L), List.of(first.sequence(), failed.sequence()));
        assertEquals(List.of(1, 1), List.of(first.attempt(), failed.attempt()));
        assertEquals(
            List.of(0L, 1L, 2L), List.of(retry.sequence(), last.sequence(), complete.sequence()));
        assertEquals(
            List.of(2, 2, 2), List.of(retry.attempt(), last.attempt(), complete.attempt()));
        assertEquals(first.activityId(), retry.activityId());
        assertEquals(ModelStreamEvent.Kind.COMPLETE, complete.kind());
        fixture.workflow.finish();
        assertEquals("Hello", fixture.result().get(10, TimeUnit.SECONDS));
        assertEquals(2, fixture.model.calls.get());
        fixture.replay();
      }
    }
  }

  @Test
  void reportsANonRetryableFailure() throws Exception {
    try (Fixture fixture = new Fixture(LiveWorkflowImpl.class)) {
      fixture.model.failPermanently = true;
      fixture.start(true, true);
      try (Subscriber subscriber = fixture.subscribe()) {
        subscriber.next();
        fixture.model.release.countDown();
        assertEquals(ModelStreamEvent.Kind.FAILED, subscriber.next().kind());
        fixture.workflow.finish();
        assertEquals("failed:test", fixture.result().get(10, TimeUnit.SECONDS));
        assertEquals(1, fixture.model.calls.get());
        fixture.replay();
      }
    }
  }

  @Test
  void cancellationStopsLiveOutputEvenWhenTheProviderFinishesLater() throws Exception {
    try (Fixture fixture = new Fixture(LiveWorkflowImpl.class)) {
      fixture.start(true, true);
      try (Subscriber subscriber = fixture.subscribe()) {
        subscriber.next();
        WorkflowStub.fromTyped(fixture.workflow).cancel();
        assertEquals(ModelStreamEvent.Kind.CANCELED, subscriber.next().kind());
        assertEquals(1, fixture.model.finished.getCount());
        fixture.model.release.countDown();
        assertTrue(fixture.model.finished.await(5, TimeUnit.SECONDS));
        assertNull(subscriber.items.poll(300, TimeUnit.MILLISECONDS), "late output is dropped");
        fixture.workflow.finish();
        var failure =
            assertThrows(
                java.util.concurrent.ExecutionException.class,
                () -> fixture.result().get(10, TimeUnit.SECONDS));
        assertInstanceOf(CanceledFailure.class, failure.getCause().getCause());
        fixture.replay();
      }
    }
  }

  @Test
  void enablingLiveDeliveryReplaysHistoriesRecordedWithoutATopic() throws Exception {
    try (Fixture fixture = new Fixture(LegacyWorkflowImpl.class)) {
      fixture.model.release.countDown();
      fixture.start(true, false);
      assertEquals("Hello", fixture.result().get(10, TimeUnit.SECONDS));
      fixture.replay();
    }
  }

  @Test
  void defaultInputEncodingIsUnchangedAndOldPayloadsRemainReadable() {
    var converter = GoogleAdkDataConverter.newDefaultInstance();
    LlmRequest request = LlmRequest.builder().model(MODEL).build();
    var legacy = converter.toPayload(new LegacyInput(MODEL, request, true)).orElseThrow();
    var input = new InvokeModelInput(MODEL, request, true);
    assertEquals(legacy, converter.toPayload(input).orElseThrow());
    assertEquals(
        input, converter.fromPayload(legacy, InvokeModelInput.class, InvokeModelInput.class));
    var live = new InvokeModelInput(MODEL, request, true, TOPIC);
    assertEquals(
        live,
        converter.fromPayload(
            converter.toPayload(live).orElseThrow(),
            InvokeModelInput.class,
            InvokeModelInput.class));
  }

  @Test
  void streamingTopicRejectsBlankNames() {
    assertThrows(
        IllegalArgumentException.class, () -> TemporalModel.newBuilder(MODEL).streamingTopic(" "));
  }

  @Test
  void publishingFailuresFailTheActivityAsStreamErrors() throws Exception {
    RejectingPublishInterceptor publishes = new RejectingPublishInterceptor();
    try (Fixture fixture = new Fixture(LiveWorkflowImpl.class, null, publishes)) {
      fixture.model.release.countDown();
      fixture.start(true, false);
      assertEquals(
          "failed:" + FailureTypes.STREAM_ERROR, fixture.result().get(30, TimeUnit.SECONDS));
      assertEquals(2, fixture.model.calls.get(), "a publishing failure is retried");
      assertTrue(publishes.rejected.get() > 0);
      HistoryEvent failed =
          fixture.client.fetchHistory(fixture.id).getEvents().stream()
              .filter(HistoryEvent::hasActivityTaskFailedEventAttributes)
              .findFirst()
              .orElseThrow();
      String message = failed.getActivityTaskFailedEventAttributes().getFailure().getMessage();
      assertTrue(message.contains("topic '" + TOPIC + "'"), message);
      fixture.replay();
    }
  }

  private record LegacyInput(String model, LlmRequest request, boolean stream) {}

  @WorkflowInterface
  public interface LiveWorkflow {
    @WorkflowMethod
    String run(boolean live, boolean holdOpen);

    @SignalMethod
    void finish();
  }

  public static class LiveWorkflowImpl implements LiveWorkflow {
    private boolean finished;
    private final WorkflowStream output;

    @WorkflowInit
    public LiveWorkflowImpl(boolean live, boolean holdOpen) {
      output = WorkflowStream.newInstance();
    }

    protected boolean live(boolean requested) {
      return requested;
    }

    @Override
    public String run(boolean live, boolean holdOpen) {
      TemporalModel.Builder model =
          TemporalModel.newBuilder(MODEL)
              .activityOptions(
                  ActivityOptions.newBuilder()
                      .setStartToCloseTimeout(Duration.ofSeconds(20))
                      .setHeartbeatTimeout(Duration.ofSeconds(2))
                      .setCancellationType(ActivityCancellationType.WAIT_CANCELLATION_COMPLETED)
                      .setRetryOptions(RetryOptions.newBuilder().setMaximumAttempts(2).build())
                      .build());
      if (live(live)) model.streamingTopic(TOPIC);
      LlmAgent agent =
          LlmAgent.builder().name("streamer").model(model.build()).instruction("Reply.").build();
      String result;
      try {
        result =
            TestSupport.lastText(
                TemporalAdk.run(
                    TemporalAdk.runnerBuilder().appName("live-stream").agent(agent).build(),
                    "user",
                    "session",
                    Content.fromParts(Part.fromText("Hi")),
                    RunConfig.builder()
                        .autoCreateSession(true)
                        .streamingMode(RunConfig.StreamingMode.SSE)
                        .build()));
      } catch (CanceledFailure e) {
        result = "cancelled";
      } catch (ActivityFailure e) {
        result =
            e.getCause() instanceof ApplicationFailure failure
                ? "failed:" + failure.getType()
                : "failed";
      }
      if (holdOpen) {
        output.topic("turn-status").publish("complete");
        Workflow.newDetachedCancellationScope(
                () -> Workflow.await(Duration.ofSeconds(20), () -> finished))
            .run();
      }
      return result;
    }

    @Override
    public void finish() {
      finished = true;
    }
  }

  public static class LegacyWorkflowImpl extends LiveWorkflowImpl {
    @WorkflowInit
    public LegacyWorkflowImpl(boolean live, boolean holdOpen) {
      super(live, holdOpen);
    }

    @Override
    protected boolean live(boolean requested) {
      return false;
    }
  }

  private static final class Fixture implements AutoCloseable {
    final GateModel model = new GateModel();
    final String id = "stream-" + UUID.randomUUID();
    final String queue = "stream-" + UUID.randomUUID();
    final TestWorkflowEnvironment environment;
    final WorkflowClient client;
    final LiveWorkflow workflow;

    Fixture(Class<?> implementation) {
      this(implementation, null);
    }

    Fixture(
        Class<?> implementation, PayloadCodec codec, WorkflowClientInterceptor... interceptors) {
      GoogleAdkPlugin plugin =
          GoogleAdkPlugin.newBuilder()
              .model(MODEL, model)
              .customizeDataConverter(codec == null)
              .build();
      var options =
          WorkflowClientOptions.newBuilder().setPlugins(plugin).setInterceptors(interceptors);
      if (codec != null) {
        options.setDataConverter(
            new CodecDataConverter(GoogleAdkDataConverter.newDefaultInstance(), List.of(codec)));
      }
      environment =
          TestWorkflowEnvironment.newInstance(
              TestEnvironmentOptions.newBuilder()
                  .setWorkflowClientOptions(options.build())
                  .build());
      client = environment.getWorkflowClient();
      Worker worker = environment.newWorker(queue);
      worker.registerWorkflowImplementationTypes(implementation);
      environment.start();
      workflow =
          client.newWorkflowStub(
              LiveWorkflow.class,
              WorkflowOptions.newBuilder().setTaskQueue(queue).setWorkflowId(id).build());
    }

    void start(boolean live, boolean holdOpen) {
      WorkflowClient.start(workflow::run, live, holdOpen);
    }

    CompletableFuture<String> result() {
      return WorkflowStub.fromTyped(workflow).getResultAsync(String.class);
    }

    Subscriber subscribe() {
      return new Subscriber(client, id);
    }

    void replay() throws Exception {
      WorkflowExecutionHistory history = client.fetchHistory(id);
      int calls = model.calls.get();
      try (TestWorkflowEnvironment replay =
          TestWorkflowEnvironment.newInstance(
              TestEnvironmentOptions.newBuilder()
                  .setWorkflowClientOptions(
                      WorkflowClientOptions.newBuilder()
                          .setDataConverter(client.getOptions().getDataConverter())
                          .build())
                  .build())) {
        Worker worker = replay.newWorker("replay");
        worker.registerWorkflowImplementationTypes(LiveWorkflowImpl.class);
        worker.replayWorkflowExecution(history);
      }
      assertEquals(calls, model.calls.get());
    }

    @Override
    public void close() {
      model.release.countDown();
      environment.close();
    }
  }

  /** Rejects every Workflow Streams publish Signal, as an unreachable service would. */
  private static final class RejectingPublishInterceptor extends WorkflowClientInterceptorBase {
    final AtomicInteger rejected = new AtomicInteger();

    @Override
    public WorkflowClientCallsInterceptor workflowClientCallsInterceptor(
        WorkflowClientCallsInterceptor next) {
      return new WorkflowClientCallsInterceptorBase(next) {
        @Override
        public WorkflowSignalOutput signal(WorkflowSignalInput input) {
          if (input.getSignalName().equals(WorkflowStreamConstants.PUBLISH_SIGNAL_NAME)) {
            rejected.incrementAndGet();
            throw new IllegalStateException("publish rejected");
          }
          return super.signal(input);
        }
      };
    }
  }

  private static final class StrictCodec implements PayloadCodec {
    private static final ByteString ENCODING = ByteString.copyFromUtf8("binary/test-codec");
    final AtomicInteger encoded = new AtomicInteger();
    final AtomicInteger decoded = new AtomicInteger();

    @Override
    public List<Payload> encode(List<Payload> payloads) {
      return payloads.stream()
          .map(
              payload -> {
                encoded.incrementAndGet();
                return Payload.newBuilder()
                    .putMetadata("encoding", ENCODING)
                    .setData(payload.toByteString())
                    .build();
              })
          .toList();
    }

    @Override
    public List<Payload> decode(List<Payload> payloads) {
      return payloads.stream()
          .map(
              payload -> {
                if (!ENCODING.equals(payload.getMetadataMap().get("encoding"))) {
                  throw new IllegalArgumentException(
                      "codec was applied to a codec-free inner item");
                }
                decoded.incrementAndGet();
                try {
                  return Payload.parseFrom(payload.getData());
                } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                  throw new IllegalArgumentException(e);
                }
              })
          .toList();
    }
  }

  private static final class Subscriber implements AutoCloseable, WorkflowStreamListener {
    final BlockingQueue<ModelStreamEvent> items = new LinkedBlockingQueue<>();
    final AtomicReference<Throwable> error = new AtomicReference<>();
    final CountDownLatch turnComplete = new CountDownLatch(1);
    final WorkflowStreamClient client;
    final WorkflowStreamSubscriptionHandle subscription;

    Subscriber(WorkflowClient workflowClient, String id) {
      client = WorkflowStreamClient.newInstance(workflowClient, id);
      subscription =
          client.subscribe(
              SubscribeOptions.newBuilder()
                  .setTopics(TOPIC, "turn-status")
                  .setFromOffset(0)
                  .build(),
              this);
    }

    ModelStreamEvent next() throws Exception {
      ModelStreamEvent next = items.poll(15, TimeUnit.SECONDS);
      assertNull(error.get(), () -> "subscription failed: " + error.get());
      assertNotNull(next, "subscriber timed out");
      return next;
    }

    @Override
    public CompletionStage<Void> onNext(io.temporal.workflowstreams.WorkflowStreamItem item) {
      if (item.getTopic().equals("turn-status")) {
        turnComplete.countDown();
        return null;
      }
      items.add(
          new GoogleAdkPayloadConverter()
              .fromData(item.getPayload(), ModelStreamEvent.class, ModelStreamEvent.class));
      return null;
    }

    void awaitTurnComplete() throws InterruptedException {
      assertTrue(
          turnComplete.await(5, TimeUnit.SECONDS),
          "the durable model Activity must complete before acknowledging the turn");
    }

    @Override
    public void onError(Throwable error) {
      this.error.set(error);
    }

    @Override
    public void close() {
      subscription.close();
      client.close();
    }
  }

  private static final class GateModel extends BaseLlm {
    final CountDownLatch release = new CountDownLatch(1);
    final CountDownLatch finished = new CountDownLatch(1);
    final AtomicInteger calls = new AtomicInteger();
    volatile boolean failFirst;
    volatile boolean failPermanently;

    GateModel() {
      super(MODEL);
    }

    @Override
    public Flowable<LlmResponse> generateContent(LlmRequest request, boolean stream) {
      int attempt = calls.incrementAndGet();
      return Flowable.create(
          emitter -> {
            emitter.onNext(ScriptedModel.partialText("Hel"));
            if (!release.await(30, TimeUnit.SECONDS))
              throw new IllegalStateException("model was not released");
            if (failPermanently) {
              emitter.onError(
                  ApplicationFailure.newNonRetryableFailure("provider rejected request", "test"));
            } else if (failFirst && attempt == 1) {
              emitter.onError(new IllegalStateException("transient provider error"));
            } else {
              emitter.onNext(ScriptedModel.text("Hello"));
              emitter.onComplete();
            }
            finished.countDown();
          },
          BackpressureStrategy.BUFFER);
    }

    @Override
    public BaseLlmConnection connect(LlmRequest request) {
      throw new UnsupportedOperationException();
    }
  }
}
