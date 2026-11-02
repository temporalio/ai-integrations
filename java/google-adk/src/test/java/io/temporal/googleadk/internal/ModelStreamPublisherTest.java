package io.temporal.googleadk.internal;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import io.grpc.ManagedChannel;
import io.grpc.Server;
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
import io.temporal.common.SimplePlugin;
import io.temporal.googleadk.model.ModelStreamEvent;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.serviceclient.WorkflowServiceStubs;
import io.temporal.serviceclient.WorkflowServiceStubsOptions;
import java.time.Duration;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class ModelStreamPublisherTest {
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
