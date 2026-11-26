package io.temporal.langchain4j;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.StreamingHandle;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.TokenStream;
import io.temporal.activity.ActivityOptions;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowOptions;
import io.temporal.client.WorkflowStub;
import io.temporal.common.RetryOptions;
import io.temporal.common.converter.DefaultDataConverter;
import io.temporal.langchain4j.activity.ChatStreamPart;
import io.temporal.langchain4j.model.ModelStreamEvent;
import io.temporal.langchain4j.model.TemporalStreamingChatModel;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.testing.WorkflowReplayer;
import io.temporal.worker.Worker;
import io.temporal.workflow.SignalMethod;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.WorkflowInit;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import io.temporal.workflowstreams.SubscribeOptions;
import io.temporal.workflowstreams.WorkflowStream;
import io.temporal.workflowstreams.WorkflowStreamClient;
import io.temporal.workflowstreams.WorkflowStreamItem;
import io.temporal.workflowstreams.WorkflowStreamListener;
import io.temporal.workflowstreams.WorkflowStreamSubscriptionHandle;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/** Publishes a streaming model's partial output to a Workflow Stream while the activity runs. */
@Timeout(45)
class LiveStreamingWorkflowTest {

  private static final String MODEL = "streaming";
  private static final String TOPIC = "model-output";
  private static final String STATUS = "turn-status";

  private final CountDownLatch release = new CountDownLatch(1);
  private final String id = "live-" + UUID.randomUUID();
  private final String queue = "live-" + UUID.randomUUID();
  private TestWorkflowEnvironment environment;
  private WorkflowClient client;
  private LiveWorkflow workflow;

  @BeforeEach
  void setUp() {
    ScriptedStreamingChatModel model =
        new ScriptedStreamingChatModel(
            (request, handler) -> {
              StreamingHandle handle = new IgnoredHandle();
              handler.onPartialResponse(
                  new PartialResponse("Hel"), new PartialResponseContext(handle));
              try {
                release.await(20, TimeUnit.SECONDS);
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
              }
              handler.onPartialResponse(
                  new PartialResponse("lo"), new PartialResponseContext(handle));
              handler.onCompleteResponse(
                  ChatResponse.builder().aiMessage(AiMessage.from("Hello")).build());
            });
    environment =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder()
                        .setPlugins(
                            LangChain4jPlugin.newBuilder().streamingChatModel(MODEL, model).build())
                        .build())
                .build());
    Worker worker = environment.newWorker(queue);
    worker.registerWorkflowImplementationTypes(LiveWorkflowImpl.class);
    environment.start();
    client = environment.getWorkflowClient();
    workflow =
        client.newWorkflowStub(
            LiveWorkflow.class,
            WorkflowOptions.newBuilder().setTaskQueue(queue).setWorkflowId(id).build());
  }

  @AfterEach
  void tearDown() {
    release.countDown();
    environment.close();
  }

  public interface StreamingAssistant {
    TokenStream chat(String message);
  }

  @WorkflowInterface
  public interface LiveWorkflow {
    @WorkflowMethod
    String run(String message);

    @SignalMethod
    void streamConsumed();
  }

  public static final class LiveWorkflowImpl implements LiveWorkflow {
    private final WorkflowStream output;
    private boolean consumed;

    @WorkflowInit
    public LiveWorkflowImpl(String message) {
      output = WorkflowStream.newInstance();
    }

    @Override
    public String run(String message) {
      StreamingAssistant assistant =
          AiServices.create(
              StreamingAssistant.class,
              TemporalStreamingChatModel.newBuilder(MODEL)
                  .streamingTopic(TOPIC)
                  .activityOptions(
                      ActivityOptions.newBuilder()
                          .setStartToCloseTimeout(Duration.ofSeconds(30))
                          .setRetryOptions(RetryOptions.newBuilder().setMaximumAttempts(1).build())
                          .build())
                  .build());
      List<String> events = new ArrayList<>();
      assistant
          .chat(message)
          .onPartialResponse(partial -> events.add(partial))
          .onCompleteResponse(response -> events.add(response.aiMessage().text()))
          .onError(error -> events.add("error"))
          .start();
      // Subscribers read through an Update, which a closed workflow can't serve.
      output.topic(STATUS).publish("complete");
      Workflow.await(Duration.ofSeconds(10), () -> consumed);
      return String.join("|", events);
    }

    @Override
    public void streamConsumed() {
      consumed = true;
    }
  }

  @Test
  void deliversPartialOutputBeforeTheActivityCompletesAndReplaysWithoutPublishing()
      throws Exception {
    WorkflowClient.start(workflow::run, "Say hello");
    try (Subscriber subscriber = new Subscriber()) {
      ModelStreamEvent first = subscriber.next();
      assertEquals(ModelStreamEvent.Kind.PART, first.kind());
      assertEquals(
          new ChatStreamPart(ChatStreamPart.Type.PARTIAL_RESPONSE, "Hel", null, null, null),
          first.part());
      assertEquals(0, first.sequence());
      assertEquals(1, first.attempt());
      assertEquals(id, first.workflowId());
      assertEquals(MODEL, first.model());
      assertFalse(first.runId().isEmpty());
      assertFalse(subscriber.turnComplete.await(0, TimeUnit.SECONDS), "the model is still running");

      release.countDown();
      ModelStreamEvent second = subscriber.next();
      ModelStreamEvent complete = subscriber.next();
      assertEquals("lo", second.part().text());
      assertEquals(ModelStreamEvent.Kind.COMPLETE, complete.kind());
      assertNull(complete.part());
      assertEquals(List.of(1L, 2L), List.of(second.sequence(), complete.sequence()));
      assertEquals(first.activityId(), complete.activityId());
      assertTrue(subscriber.turnComplete.await(10, TimeUnit.SECONDS));
    }
    workflow.streamConsumed();
    CompletableFuture<String> result =
        WorkflowStub.fromTyped(workflow).getResultAsync(String.class);
    // The workflow gets the same output from the activity result, after it completes.
    assertEquals("Hel|lo|Hello", result.get(10, TimeUnit.SECONDS));
    assertDoesNotThrow(
        () ->
            WorkflowReplayer.replayWorkflowExecution(
                client.fetchHistory(id), LiveWorkflowImpl.class));
  }

  @Test
  void completesWithoutASubscriber() throws Exception {
    release.countDown();
    WorkflowClient.start(workflow::run, "Say hello");
    workflow.streamConsumed();
    assertEquals(
        "Hel|lo|Hello",
        WorkflowStub.fromTyped(workflow).getResultAsync(String.class).get(10, TimeUnit.SECONDS));
  }

  /** Reads the model topic from the start; items are codec-free JSON of the standard converter. */
  private final class Subscriber implements AutoCloseable, WorkflowStreamListener {
    final BlockingQueue<ModelStreamEvent> items = new LinkedBlockingQueue<>();
    final AtomicReference<Throwable> error = new AtomicReference<>();
    final CountDownLatch turnComplete = new CountDownLatch(1);
    final WorkflowStreamClient streams = WorkflowStreamClient.newInstance(client, id);
    final WorkflowStreamSubscriptionHandle subscription =
        streams.subscribe(
            SubscribeOptions.newBuilder().setTopics(TOPIC, STATUS).setFromOffset(0).build(), this);

    ModelStreamEvent next() throws InterruptedException {
      ModelStreamEvent next = items.poll(15, TimeUnit.SECONDS);
      assertNull(error.get(), () -> "subscription failed: " + error.get());
      assertNotNull(next, "subscriber timed out");
      return next;
    }

    @Override
    public CompletionStage<Void> onNext(WorkflowStreamItem item) {
      if (item.getTopic().equals(STATUS)) {
        turnComplete.countDown();
      } else {
        items.add(
            DefaultDataConverter.STANDARD_INSTANCE.fromPayload(
                item.getPayload(), ModelStreamEvent.class, ModelStreamEvent.class));
      }
      return null;
    }

    @Override
    public void onError(Throwable error) {
      this.error.set(error);
    }

    @Override
    public void close() {
      subscription.close();
      streams.close();
    }
  }

  /** The handle of a provider that ignores cancellation. */
  private static final class IgnoredHandle implements StreamingHandle {
    @Override
    public void cancel() {}

    @Override
    public boolean isCancelled() {
      return false;
    }
  }
}
