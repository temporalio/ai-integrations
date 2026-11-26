package io.temporal.langchain4j;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.exception.AuthenticationException;
import dev.langchain4j.exception.RateLimitException;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.CompleteToolCall;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.PartialToolCall;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.chat.response.StreamingHandle;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.TokenStream;
import io.temporal.activity.ActivityOptions;
import io.temporal.api.history.v1.HistoryEvent;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowOptions;
import io.temporal.client.WorkflowStub;
import io.temporal.common.RetryOptions;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.failure.ActivityFailure;
import io.temporal.failure.ApplicationFailure;
import io.temporal.langchain4j.activity.StreamingChatModelActivity;
import io.temporal.langchain4j.internal.Failures;
import io.temporal.langchain4j.model.TemporalStreamingChatModel;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.testing.WorkflowReplayer;
import io.temporal.worker.Worker;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/** Runs LangChain4j AI Services that return a {@code TokenStream} in workflows. */
@Timeout(30)
class StreamingWorkflowTest {

  private static final String TASK_QUEUE = "langchain4j-streaming";
  private static final String SECRET = "secret-token-1234";
  private static final ActivityOptions FAST_RETRIES =
      ActivityOptions.newBuilder()
          .setStartToCloseTimeout(Duration.ofSeconds(10))
          .setRetryOptions(
              RetryOptions.newBuilder()
                  .setInitialInterval(Duration.ofMillis(10))
                  .setMaximumAttempts(3)
                  .build())
          .build();

  private TestWorkflowEnvironment testEnv;
  private WorkflowClient client;

  @AfterEach
  void tearDown() {
    if (testEnv != null) {
      testEnv.close();
    }
  }

  private StreamingWorkflow start(ScriptedStreamingChatModel model) {
    testEnv =
        TestWorkflowEnvironment.newInstance(
            TestEnvironmentOptions.newBuilder()
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder()
                        .setPlugins(
                            LangChain4jPlugin.newBuilder()
                                .streamingChatModel("streaming", model)
                                .build())
                        .build())
                .build());
    Worker worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(StreamingWorkflowImpl.class);
    testEnv.start();
    client = testEnv.getWorkflowClient();
    return client.newWorkflowStub(
        StreamingWorkflow.class, WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());
  }

  private WorkflowExecutionHistory history(StreamingWorkflow workflow) {
    return client.fetchHistory(WorkflowStub.fromTyped(workflow).getExecution().getWorkflowId());
  }

  private static List<Integer> attempts(WorkflowExecutionHistory history) {
    return history.getEvents().stream()
        .filter(HistoryEvent::hasActivityTaskStartedEventAttributes)
        .map(e -> e.getActivityTaskStartedEventAttributes().getAttempt())
        .toList();
  }

  public interface StreamingAssistant {
    TokenStream chat(String message);
  }

  @WorkflowInterface
  public interface StreamingWorkflow {
    @WorkflowMethod
    String ask(String question, boolean cancelAfterFirstPartial);
  }

  public static final class StreamingWorkflowImpl implements StreamingWorkflow {
    @Override
    public String ask(String question, boolean cancelAfterFirstPartial) {
      StreamingAssistant assistant =
          AiServices.builder(StreamingAssistant.class)
              .streamingChatModel(
                  TemporalStreamingChatModel.newBuilder("streaming")
                      .activityOptions(FAST_RETRIES)
                      .build())
              .tools(new ChatWorkflowTest.Calculator())
              .build();
      List<String> events = new ArrayList<>();
      // The model is synchronous in a workflow, so every callback has run when start() returns.
      assistant
          .chat(question)
          .onPartialThinking(thinking -> events.add("thinking:" + thinking.text()))
          .onPartialToolCall(call -> events.add("tool-call:" + call.partialArguments()))
          .onToolExecuted(tool -> events.add("tool:" + tool.result()))
          .onPartialResponseWithContext(
              (partial, context) -> {
                events.add("partial:" + partial.text());
                if (cancelAfterFirstPartial) {
                  context.streamingHandle().cancel();
                }
              })
          .onCompleteResponse(response -> events.add("complete:" + response.aiMessage().text()))
          .onError(error -> events.add("error:" + describe(error)))
          .start();
      return String.join("|", events);
    }

    private static String describe(Throwable error) {
      if (error instanceof ActivityFailure activity
          && activity.getCause() instanceof ApplicationFailure failure) {
        return failure.getType() + ":" + failure.getOriginalMessage();
      }
      return error.getClass().getName();
    }
  }

  private static void addCall(ChatRequest request, StreamingChatResponseHandler handler) {
    handler.onPartialThinking(new PartialThinking("adding"));
    handler.onPartialToolCall(
        PartialToolCall.builder()
            .index(0)
            .id("call-1")
            .name("add")
            .partialArguments("{\"a\":2,")
            .build());
    ToolExecutionRequest call =
        ToolExecutionRequest.builder()
            .id("call-1")
            .name("add")
            .arguments("{\"a\":2,\"b\":3}")
            .build();
    handler.onCompleteToolCall(new CompleteToolCall(0, call));
    handler.onCompleteResponse(ChatResponse.builder().aiMessage(AiMessage.from(call)).build());
  }

  private static void answer(ChatRequest request, StreamingChatResponseHandler handler) {
    StreamingHandle handle = new NoCancelHandle();
    handler.onPartialResponse(new PartialResponse("The sum "), new PartialResponseContext(handle));
    handler.onPartialResponse(new PartialResponse("is 5."), new PartialResponseContext(handle));
    handler.onCompleteResponse(
        ChatResponse.builder().aiMessage(AiMessage.from("The sum is 5.")).build());
  }

  @Test
  void tokenStreamGetsThePartialOutputInOrderAndToolsRunInTheWorkflow() {
    ScriptedStreamingChatModel model =
        new ScriptedStreamingChatModel(
            StreamingWorkflowTest::addCall, StreamingWorkflowTest::answer);
    StreamingWorkflow workflow = start(model);

    assertEquals(
        "thinking:adding|tool-call:{\"a\":2,|tool:5|partial:The sum |partial:is 5."
            + "|complete:The sum is 5.",
        workflow.ask("What is 2 + 3?", false));

    WorkflowExecutionHistory history = history(workflow);
    assertEquals(
        List.of(
            StreamingChatModelActivity.STREAMING_CHAT, StreamingChatModelActivity.STREAMING_CHAT),
        history.getEvents().stream()
            .filter(HistoryEvent::hasActivityTaskScheduledEventAttributes)
            .map(e -> e.getActivityTaskScheduledEventAttributes().getActivityType().getName())
            .toList());
    // The first request carried the tool; the second, the tool's result from the workflow.
    assertEquals("add", model.requests.get(0).toolSpecifications().get(0).name());
    List<?> messages = model.requests.get(1).messages();
    assertEquals(
        "5",
        assertInstanceOf(ToolExecutionResultMessage.class, messages.get(messages.size() - 1))
            .text());

    // The token stream, its callbacks and the tool replay from the history.
    assertDoesNotThrow(
        () -> WorkflowReplayer.replayWorkflowExecution(history, StreamingWorkflowImpl.class));
  }

  @Test
  void cancellingTheHandleStopsTheRemainingCallbacks() {
    StreamingWorkflow workflow =
        start(new ScriptedStreamingChatModel(StreamingWorkflowTest::answer));
    // The model already finished in the activity, so cancelling stops only the replayed output,
    // including the complete response, as LangChain4j's StreamingHandle specifies.
    assertEquals("partial:The sum ", workflow.ask("What is 2 + 3?", true));
  }

  @Test
  void finalProviderErrorsReachTheHandlerWithoutRetriesOrTheProviderMessage() {
    StreamingWorkflow workflow =
        start(
            new ScriptedStreamingChatModel(
                (request, handler) ->
                    handler.onError(new AuthenticationException("invalid key " + SECRET))));

    assertEquals(
        "error:"
            + Failures.MODEL_ERROR
            + ":Chat model streaming failed with dev.langchain4j.exception.AuthenticationException",
        workflow.ask("Hi", false));
    WorkflowExecutionHistory history = history(workflow);
    assertEquals(List.of(1), attempts(history));
    assertFalse(history.toJson(false).contains(SECRET));
  }

  @Test
  void otherProviderErrorsFollowTheRetryPolicy() {
    StreamingWorkflow workflow =
        start(
            new ScriptedStreamingChatModel(
                (request, handler) -> handler.onError(new RateLimitException("slow down")),
                StreamingWorkflowTest::answer));

    assertEquals(
        "partial:The sum |partial:is 5.|complete:The sum is 5.", workflow.ask("Hi", false));
    assertEquals(List.of(2), attempts(history(workflow)));
  }

  /** The handle of a provider that can't cancel its stream. */
  private static final class NoCancelHandle implements StreamingHandle {
    @Override
    public void cancel() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean isCancelled() {
      return false;
    }
  }
}
