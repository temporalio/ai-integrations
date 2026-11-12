package io.temporal.googleadk.internal;

import com.google.adk.models.LlmResponse;
import io.grpc.Context;
import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.health.v1.HealthCheckResponse;
import io.temporal.activity.ActivityExecutionContext;
import io.temporal.activity.ActivityInfo;
import io.temporal.api.workflowservice.v1.GetSystemInfoResponse;
import io.temporal.api.workflowservice.v1.WorkflowServiceGrpc;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.common.interceptors.WorkflowClientCallsInterceptor;
import io.temporal.common.interceptors.WorkflowClientCallsInterceptorBase;
import io.temporal.common.interceptors.WorkflowClientInterceptor;
import io.temporal.common.interceptors.WorkflowClientInterceptorBase;
import io.temporal.googleadk.converter.GoogleAdkPayloadConverter;
import io.temporal.googleadk.model.ModelStreamEvent;
import io.temporal.serviceclient.WorkflowServiceStubs;
import io.temporal.serviceclient.WorkflowServiceStubsOptions;
import io.temporal.workflowstreams.FlushTimeoutException;
import io.temporal.workflowstreams.PublishInput;
import io.temporal.workflowstreams.TopicHandle;
import io.temporal.workflowstreams.WorkflowStreamClient;
import io.temporal.workflowstreams.WorkflowStreamClientOptions;
import java.time.Duration;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Publishes one model Activity attempt's responses to a Workflow Stream topic. Responses are
 * accepted on the provider's thread and batched; the terminal record is delivered synchronously on
 * the Activity thread. Transport and encoding failures surface as {@link PublishFailure}.
 */
public final class ModelStreamPublisher implements AutoCloseable {
  private static final Duration PUBLISH_TIMEOUT = Duration.ofSeconds(5);
  private static final ScheduledExecutorService DEADLINES =
      Executors.newSingleThreadScheduledExecutor(
          runnable -> {
            Thread thread = new Thread(runnable, "googleadk-stream-deadline");
            thread.setDaemon(true);
            return thread;
          });
  private final PublishFailures publishFailures = new PublishFailures();
  private final WorkflowStreamClient client;
  private final TopicHandle topic;
  private final ActivityInfo info;
  private final String model;
  private long sequence;
  private boolean closed;

  /** Creates a publisher for the current Activity's Workflow. */
  public ModelStreamPublisher(ActivityExecutionContext context, String model, String topic) {
    this(context, model, topic, PUBLISH_TIMEOUT);
  }

  ModelStreamPublisher(
      ActivityExecutionContext context, String model, String topic, Duration publishTimeout) {
    this.info = context.getInfo();
    this.model = model;
    WorkflowStreamClientOptions options =
        WorkflowStreamClientOptions.newBuilder()
            .setBatchInterval(Duration.ofMillis(100))
            .setMaxRetryDuration(publishTimeout)
            .setPayloadConverters(new GoogleAdkPayloadConverter())
            .build();
    this.client =
        WorkflowStreamClient.newInstance(
            boundedClient(context.getWorkflowClient(), publishTimeout),
            info.getWorkflowId(),
            options);
    this.topic = client.topic(topic);
  }

  /**
   * Clones the Worker's client with a gRPC deadline on every publish Signal. The SDK retries a
   * Signal for up to a minute unless the calling context carries a deadline, and it applies a
   * service stub's plugins again to any client built from it, so the clone borrows the stubs
   * without their plugins.
   */
  private WorkflowClient boundedClient(WorkflowClient source, Duration timeout) {
    Objects.requireNonNull(
        source.getWorkflowServiceStubs().getRawChannel(), "Workflow Service channel");
    WorkflowClientInterceptor[] interceptors = source.getOptions().getInterceptors();
    WorkflowClientInterceptor[] extended = Arrays.copyOf(interceptors, interceptors.length + 1);
    extended[interceptors.length] =
        new WorkflowClientInterceptorBase() {
          @Override
          public WorkflowClientCallsInterceptor workflowClientCallsInterceptor(
              WorkflowClientCallsInterceptor next) {
            return new WorkflowClientCallsInterceptorBase(next) {
              @Override
              public WorkflowSignalOutput signal(WorkflowSignalInput input) {
                PublishInput batch = (PublishInput) input.getArguments()[0];
                try (Context.CancellableContext deadline =
                    Context.current()
                        .withDeadlineAfter(timeout.toNanos(), TimeUnit.NANOSECONDS, DEADLINES)) {
                  WorkflowSignalOutput output = deadline.call(() -> super.signal(input));
                  publishFailures.record(batch.sequence, null);
                  return output;
                } catch (Exception e) {
                  RuntimeException failure =
                      e instanceof RuntimeException
                          ? (RuntimeException) e
                          : new IllegalStateException(e);
                  publishFailures.record(batch.sequence, failure);
                  throw failure;
                }
              }
            };
          }
        };
    return WorkflowClient.newInstance(
        new BorrowedService(source.getWorkflowServiceStubs()),
        WorkflowClientOptions.newBuilder(source.getOptions())
            .setPlugins()
            .setInterceptors(extended)
            .build());
  }

  /** Queues a response for the next batch while the Activity is still collecting its result. */
  public synchronized void accept(LlmResponse response) {
    if (!closed) {
      publish(ModelStreamEvent.Kind.RESPONSE, response);
    }
  }

  /**
   * Publishes one terminal record behind the buffered responses and waits for delivery. Later
   * output of an abandoned provider call is dropped.
   */
  public void terminal(ModelStreamEvent.Kind kind) {
    // Deliver the responses before claiming completion, so that a failed delivery leaves no
    // COMPLETE record queued for a later drain. The provider has finished, so nothing races accept.
    if (kind == ModelStreamEvent.Kind.COMPLETE) {
      transport(client::flush);
    }
    synchronized (this) {
      if (closed) {
        return;
      }
      closed = true;
      publish(kind, null);
    }
    transport(client::flush);
  }

  /** Publishes a terminal record for a failed attempt without displacing the primary failure. */
  public void failed(ModelStreamEvent.Kind kind, Throwable failure) {
    try {
      terminal(kind);
    } catch (RuntimeException e) {
      failure.addSuppressed(e);
    }
  }

  private void publish(ModelStreamEvent.Kind kind, @Nullable LlmResponse response) {
    ModelStreamEvent event =
        new ModelStreamEvent(
            info.getWorkflowId(),
            info.getWorkflowRunId(),
            info.getActivityId(),
            model,
            info.getAttempt(),
            sequence++,
            kind,
            response);
    transport(() -> topic.publish(event, false));
  }

  /** Stops publishing and delivers anything still buffered. */
  @Override
  public void close() {
    synchronized (this) {
      closed = true;
    }
    transport(client::close);
  }

  private void transport(Runnable action) {
    try {
      action.run();
    } catch (RuntimeException e) {
      throw new PublishFailure(
          e, e instanceof FlushTimeoutException ? publishFailures.forDroppedBatch() : e);
    }
  }

  /** Tracks the cause that Workflow Streams omits from a dropped batch's timeout. */
  private static final class PublishFailures {
    private long sequence;
    private RuntimeException pending;
    private RuntimeException dropped;

    // Workflow Streams serializes sends, but a foreground flush can observe a background failure.
    synchronized void record(long batchSequence, @Nullable RuntimeException failure) {
      if (batchSequence != sequence) {
        // A new sequence without a successful send means the previous batch was dropped.
        // Keep its failure even if close() drains a later batch before surfacing the timeout.
        if (pending != null && dropped == null) dropped = pending;
        sequence = batchSequence;
      }
      pending = failure;
    }

    @Nullable
    synchronized RuntimeException forDroppedBatch() {
      return dropped != null ? dropped : pending;
    }
  }

  /** A Workflow Streams transport or encoding failure, as opposed to a model failure. */
  public static final class PublishFailure extends RuntimeException {
    private final boolean nonRetryable;

    PublishFailure(RuntimeException cause, @Nullable RuntimeException publishCause) {
      super(cause.getMessage() != null ? cause.getMessage() : cause.toString(), cause);
      Status status = publishCause == null ? Status.UNKNOWN : Status.fromThrowable(publishCause);
      nonRetryable =
          status.getCode() == Status.Code.INVALID_ARGUMENT
              || (status.getCode() == Status.Code.RESOURCE_EXHAUSTED
                  && status.getDescription() != null
                  && status.getDescription().contains("larger than max"));
    }

    /**
     * Returns whether the server refused the failed batch, including one dropped after the stream
     * retry window. Outages and rate limiting remain retryable after that window expires.
     */
    public boolean isNonRetryable() {
      return nonRetryable;
    }
  }

  private static final class BorrowedService implements WorkflowServiceStubs {
    private final WorkflowServiceStubs source;
    private final WorkflowServiceStubsOptions options;

    private BorrowedService(WorkflowServiceStubs source) {
      this.source = source;
      options = WorkflowServiceStubsOptions.newBuilder(source.getOptions()).setPlugins().build();
    }

    @Override
    public WorkflowServiceStubsOptions getOptions() {
      return options;
    }

    @Override
    public WorkflowServiceGrpc.WorkflowServiceBlockingStub blockingStub() {
      return source.blockingStub();
    }

    @Override
    public WorkflowServiceGrpc.WorkflowServiceFutureStub futureStub() {
      return source.futureStub();
    }

    @Override
    public ManagedChannel getRawChannel() {
      return source.getRawChannel();
    }

    @Override
    public void shutdown() {
      source.shutdown();
    }

    @Override
    public void shutdownNow() {
      source.shutdownNow();
    }

    @Override
    public boolean isShutdown() {
      return source.isShutdown();
    }

    @Override
    public boolean isTerminated() {
      return source.isTerminated();
    }

    @Override
    public boolean awaitTermination(long timeout, TimeUnit unit) {
      return source.awaitTermination(timeout, unit);
    }

    @Override
    public void connect(Duration timeout) {
      source.connect(timeout);
    }

    @Override
    public HealthCheckResponse healthCheck() {
      return source.healthCheck();
    }

    @Override
    public Supplier<GetSystemInfoResponse.Capabilities> getServerCapabilities() {
      return source.getServerCapabilities();
    }
  }
}
