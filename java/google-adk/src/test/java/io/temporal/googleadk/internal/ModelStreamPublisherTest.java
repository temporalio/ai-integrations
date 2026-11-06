package io.temporal.googleadk.internal;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.temporal.activity.ActivityExecutionContext;
import io.temporal.activity.ActivityInfo;
import io.temporal.api.workflowservice.v1.GetSystemInfoRequest;
import io.temporal.api.workflowservice.v1.GetSystemInfoResponse;
import io.temporal.api.workflowservice.v1.SignalWorkflowExecutionRequest;
import io.temporal.api.workflowservice.v1.SignalWorkflowExecutionResponse;
import io.temporal.api.workflowservice.v1.WorkflowServiceGrpc;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.common.SimplePlugin;
import io.temporal.common.interceptors.WorkflowClientCallsInterceptor;
import io.temporal.common.interceptors.WorkflowClientCallsInterceptorBase;
import io.temporal.common.interceptors.WorkflowClientInterceptorBase;
import io.temporal.googleadk.model.ModelStreamEvent;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.serviceclient.WorkflowServiceStubs;
import io.temporal.serviceclient.WorkflowServiceStubsOptions;
import io.temporal.workflowstreams.FlushTimeoutException;
import io.temporal.workflowstreams.PublishInput;
import java.time.Duration;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class ModelStreamPublisherTest {
  @ParameterizedTest
  @MethodSource("droppedFailures")
  @Timeout(10)
  void closeKeepsDroppedBatchFailureAfterDrainingAnotherBatch(Status status, boolean nonRetryable)
      throws Exception {
    AtomicReference<Status> failure = new AtomicReference<>(status);
    CountDownLatch rejected = new CountDownLatch(1);
    AtomicInteger delivered = new AtomicInteger();
    try (PublisherFixture fixture =
        new PublisherFixture(
            input -> {
              Status current = failure.get();
              if (current != null) {
                rejected.countDown();
                return current.asRuntimeException();
              }
              delivered.incrementAndGet();
              return null;
            })) {
      fixture.publisher.accept(ScriptedModel.partialText("first"));
      assertTrue(rejected.await(3, TimeUnit.SECONDS));
      TimeUnit.MILLISECONDS.sleep(1500);
      failure.set(null);
      fixture.publisher.accept(ScriptedModel.text("last"));
      var error = assertThrows(ModelStreamPublisher.PublishFailure.class, fixture.publisher::close);
      assertInstanceOf(FlushTimeoutException.class, error.getCause());
      assertEquals(1, delivered.get(), "close drains the later batch before surfacing the timeout");
      assertEquals(nonRetryable, error.isNonRetryable());
    }
  }

  static Stream<Arguments> droppedFailures() {
    return Stream.of(
        Arguments.of(Status.INVALID_ARGUMENT, true),
        Arguments.of(Status.RESOURCE_EXHAUSTED.withDescription("larger than max"), true),
        Arguments.of(
            Status.RESOURCE_EXHAUSTED.withDescription("namespace rate limit exceeded"), false),
        Arguments.of(Status.UNAVAILABLE, false));
  }

  @Test
  @Timeout(10)
  void successfulBatchClearsItsEarlierRefusal() throws Exception {
    AtomicReference<Status> failure = new AtomicReference<>(Status.INVALID_ARGUMENT);
    CountDownLatch refused = new CountDownLatch(1);
    CountDownLatch accepted = new CountDownLatch(1);
    CountDownLatch unavailable = new CountDownLatch(1);
    try (PublisherFixture fixture =
        new PublisherFixture(
            input -> {
              Status current = failure.get();
              if (current == null) {
                accepted.countDown();
                return null;
              }
              (input.sequence == 1 ? refused : unavailable).countDown();
              return current.asRuntimeException();
            })) {
      fixture.publisher.accept(ScriptedModel.partialText("first"));
      assertTrue(refused.await(3, TimeUnit.SECONDS));
      failure.set(null);
      assertTrue(accepted.await(3, TimeUnit.SECONDS));
      failure.set(Status.UNAVAILABLE);
      fixture.publisher.accept(ScriptedModel.text("last"));
      assertTrue(unavailable.await(3, TimeUnit.SECONDS));
      TimeUnit.MILLISECONDS.sleep(1500);
      var error = assertThrows(ModelStreamPublisher.PublishFailure.class, fixture.publisher::close);
      assertInstanceOf(FlushTimeoutException.class, error.getCause());
      assertFalse(error.isNonRetryable(), "the refused batch was delivered before the outage");
    }
  }

  private static final class PublisherFixture implements AutoCloseable {
    final ManagedChannel channel =
        InProcessChannelBuilder.forName(UUID.randomUUID().toString()).directExecutor().build();
    final WorkflowServiceStubs service =
        WorkflowServiceStubs.newServiceStubs(
            WorkflowServiceStubsOptions.newBuilder().setChannel(channel).build());
    final ModelStreamPublisher publisher;

    PublisherFixture(Function<PublishInput, RuntimeException> signal) {
      var interceptor =
          new WorkflowClientInterceptorBase() {
            @Override
            public WorkflowClientCallsInterceptor workflowClientCallsInterceptor(
                WorkflowClientCallsInterceptor next) {
              return new WorkflowClientCallsInterceptorBase(next) {
                @Override
                public WorkflowSignalOutput signal(WorkflowSignalInput input) {
                  RuntimeException failure = signal.apply((PublishInput) input.getArguments()[0]);
                  if (failure != null) throw failure;
                  return new WorkflowSignalOutput();
                }
              };
            }
          };
      WorkflowClient client =
          WorkflowClient.newInstance(
              service, WorkflowClientOptions.newBuilder().setInterceptors(interceptor).build());
      ActivityInfo info = mock(ActivityInfo.class);
      when(info.getWorkflowId()).thenReturn("workflow");
      when(info.getWorkflowRunId()).thenReturn("run");
      when(info.getActivityId()).thenReturn("activity");
      when(info.getAttempt()).thenReturn(1);
      ActivityExecutionContext context = mock(ActivityExecutionContext.class);
      when(context.getInfo()).thenReturn(info);
      when(context.getWorkflowClient()).thenReturn(client);
      publisher = new ModelStreamPublisher(context, "model", "topic", Duration.ofSeconds(1));
    }

    @Override
    public void close() {
      try {
        publisher.close();
      } catch (ModelStreamPublisher.PublishFailure ignored) {
        // A test may leave a refused batch pending.
      } finally {
        service.shutdownNow();
        channel.shutdownNow();
      }
    }
  }

  @Test
  @Timeout(10)
  void boundsSignalRpcRetriesWhenTheServiceNeverResponds() throws Exception {
    String name = UUID.randomUUID().toString();
    CountDownLatch received = new CountDownLatch(1);
    Server server =
        InProcessServerBuilder.forName(name)
            .directExecutor()
            .addService(
                new WorkflowServiceGrpc.WorkflowServiceImplBase() {
                  @Override
                  public void getSystemInfo(
                      GetSystemInfoRequest request,
                      StreamObserver<GetSystemInfoResponse> observer) {
                    observer.onNext(GetSystemInfoResponse.getDefaultInstance());
                    observer.onCompleted();
                  }

                  @Override
                  public void signalWorkflowExecution(
                      SignalWorkflowExecutionRequest request,
                      StreamObserver<SignalWorkflowExecutionResponse> observer) {
                    received.countDown();
                  }
                })
            .build()
            .start();
    ManagedChannel channel = InProcessChannelBuilder.forName(name).directExecutor().build();
    AtomicInteger customizations = new AtomicInteger();
    var plugin =
        SimplePlugin.newBuilder("one-shot-converter")
            .customizeDataConverter(
                converter -> {
                  if (customizations.incrementAndGet() > 1)
                    throw new IllegalStateException("plugin reapplied");
                  return converter == null
                      ? io.temporal.common.converter.DefaultDataConverter.newDefaultInstance()
                      : converter;
                })
            .build();
    WorkflowServiceStubs service =
        WorkflowServiceStubs.newServiceStubs(
            WorkflowServiceStubsOptions.newBuilder()
                .setChannel(channel)
                .setPlugins(plugin)
                .build());
    try {
      WorkflowClient source = WorkflowClient.newInstance(service);
      ActivityInfo info = mock(ActivityInfo.class);
      when(info.getWorkflowId()).thenReturn("workflow");
      when(info.getWorkflowRunId()).thenReturn("run");
      when(info.getActivityId()).thenReturn("activity");
      when(info.getAttempt()).thenReturn(1);
      ActivityExecutionContext context = mock(ActivityExecutionContext.class);
      when(context.getInfo()).thenReturn(info);
      when(context.getWorkflowClient()).thenReturn(source);
      ModelStreamPublisher publisher =
          new ModelStreamPublisher(context, "model", "topic", Duration.ofMillis(200));
      publisher.accept(ScriptedModel.partialText("first"));
      assertTrue(received.await(3, TimeUnit.SECONDS), "the real signal RPC must reach the server");
      long start = System.nanoTime();
      assertThrows(
          ModelStreamPublisher.PublishFailure.class,
          () -> publisher.terminal(ModelStreamEvent.Kind.COMPLETE));
      assertTrue(
          Duration.ofNanos(System.nanoTime() - start).compareTo(Duration.ofSeconds(2)) < 0,
          "a blackholed signal RPC must not use the client's much longer default retry timeout");
      long closing = System.nanoTime();
      try {
        publisher.close();
      } catch (ModelStreamPublisher.PublishFailure repeated) {
        // Nothing is left to deliver; a repeated transport error is acceptable.
      }
      assertTrue(
          Duration.ofNanos(System.nanoTime() - closing).compareTo(Duration.ofSeconds(2)) < 0,
          "closing after a failed delivery must not block");
      assertEquals(0, source.getOptions().getInterceptors().length);
      assertEquals(
          1,
          customizations.get(),
          "service-stub plugins are not applied to the cloned client again");
    } finally {
      service.shutdownNow();
      channel.shutdownNow();
      server.shutdownNow();
      assertTrue(channel.awaitTermination(3, TimeUnit.SECONDS));
      assertTrue(server.awaitTermination(3, TimeUnit.SECONDS));
    }
  }
}
