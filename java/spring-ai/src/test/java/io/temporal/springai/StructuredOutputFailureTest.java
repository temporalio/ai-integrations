package io.temporal.springai;

import static org.junit.jupiter.api.Assertions.*;

import io.temporal.client.WorkflowFailedException;
import io.temporal.client.WorkflowOptions;
import io.temporal.common.RetryOptions;
import io.temporal.failure.ApplicationFailure;
import io.temporal.springai.chat.TemporalChatClient;
import io.temporal.springai.model.ActivityChatModel;
import io.temporal.springai.plugin.SpringAiPlugin;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.WorkerFactoryOptions;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import tools.jackson.core.JacksonException;
import tools.jackson.core.exc.StreamReadException;

class StructuredOutputFailureTest {
  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void conversionFailureCanBeCaughtOrRetried(boolean includeResponse) throws Exception {
    AtomicInteger calls = new AtomicInteger();
    ChatModel model =
        prompt -> {
          calls.incrementAndGet();
          return new ChatResponse(
              List.of(new Generation(new AssistantMessage("The answer is 42."))));
        };
    var environmentOptions =
        TestEnvironmentOptions.newBuilder()
            .setWorkerFactoryOptions(
                WorkerFactoryOptions.newBuilder().setPlugins(new SpringAiPlugin(model)).build())
            .build();
    try (TestWorkflowEnvironment env = TestWorkflowEnvironment.newInstance(environmentOptions)) {
      var worker = env.newWorker("structured-output");
      worker.registerWorkflowImplementationTypes(AnswerWorkflowImpl.class);
      env.start();
      var options =
          WorkflowOptions.newBuilder()
              .setTaskQueue(worker.getTaskQueue())
              .setWorkflowExecutionTimeout(Duration.ofSeconds(10))
              .setRetryOptions(RetryOptions.newBuilder().setMaximumAttempts(2).build())
              .build();

      var catching = env.getWorkflowClient().newWorkflowStub(AnswerWorkflow.class, options);
      assertEquals(new Answer("fallback"), catching.ask(true, includeResponse));

      var failing = env.getWorkflowClient().newWorkflowStub(AnswerWorkflow.class, options);
      var failure =
          assertThrows(WorkflowFailedException.class, () -> failing.ask(false, includeResponse));
      var cause = assertInstanceOf(ApplicationFailure.class, failure.getCause());
      assertEquals(StreamReadException.class.getName(), cause.getType());
      assertFalse(cause.isNonRetryable());
      var execution = failure.getExecution();
      worker.replayWorkflowExecution(
          env.getWorkflowClient().fetchHistory(execution.getWorkflowId(), execution.getRunId()));
      assertEquals(3, calls.get());
    }
  }

  public record Answer(String answer) {}

  @WorkflowInterface
  public interface AnswerWorkflow {
    @WorkflowMethod
    Answer ask(boolean catchError, boolean includeResponse);
  }

  public static class AnswerWorkflowImpl implements AnswerWorkflow {
    @Override
    public Answer ask(boolean catchError, boolean includeResponse) {
      try {
        var response =
            TemporalChatClient.builder(ActivityChatModel.forDefault())
                .build()
                .prompt()
                .user("What is the answer?")
                .call();
        return includeResponse
            ? response.responseEntity(Answer.class).entity()
            : response.entity(Answer.class);
      } catch (JacksonException e) {
        if (catchError) {
          return new Answer("fallback");
        }
        throw e;
      }
    }
  }
}
