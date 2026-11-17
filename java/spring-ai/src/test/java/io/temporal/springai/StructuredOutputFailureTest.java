package io.temporal.springai;

import static org.junit.jupiter.api.Assertions.*;

import io.temporal.client.WorkflowFailedException;
import io.temporal.client.WorkflowOptions;
import io.temporal.failure.ApplicationFailure;
import io.temporal.springai.activity.ChatModelActivityImpl;
import io.temporal.springai.chat.TemporalChatClient;
import io.temporal.springai.model.ActivityChatModel;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.WorkflowImplementationOptions;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import tools.jackson.core.JacksonException;
import tools.jackson.core.exc.StreamReadException;

class StructuredOutputFailureTest {
  @Test
  void configuredJacksonFailureCanBeCaughtOrFailTheWorkflow() {
    AtomicInteger calls = new AtomicInteger();
    ChatModel model =
        prompt -> {
          calls.incrementAndGet();
          return new ChatResponse(
              List.of(new Generation(new AssistantMessage("The answer is 42."))));
        };
    try (TestWorkflowEnvironment env = TestWorkflowEnvironment.newInstance()) {
      var worker = env.newWorker("structured-output");
      worker.registerWorkflowImplementationTypes(
          WorkflowImplementationOptions.newBuilder()
              .setFailWorkflowExceptionTypes(JacksonException.class)
              .build(),
          AnswerWorkflowImpl.class);
      worker.registerActivitiesImplementations(new ChatModelActivityImpl(model));
      env.start();
      var options =
          WorkflowOptions.newBuilder()
              .setTaskQueue(worker.getTaskQueue())
              .setWorkflowExecutionTimeout(Duration.ofSeconds(10))
              .build();

      var catching = env.getWorkflowClient().newWorkflowStub(AnswerWorkflow.class, options);
      assertEquals(new Answer("fallback"), catching.ask(true));

      var failing = env.getWorkflowClient().newWorkflowStub(AnswerWorkflow.class, options);
      var failure = assertThrows(WorkflowFailedException.class, () -> failing.ask(false));
      var cause = assertInstanceOf(ApplicationFailure.class, failure.getCause());
      assertEquals(StreamReadException.class.getName(), cause.getType());
      assertEquals(2, calls.get());
    }
  }

  public record Answer(String answer) {}

  @WorkflowInterface
  public interface AnswerWorkflow {
    @WorkflowMethod
    Answer ask(boolean catchError);
  }

  public static class AnswerWorkflowImpl implements AnswerWorkflow {
    @Override
    public Answer ask(boolean catchError) {
      try {
        return TemporalChatClient.builder(ActivityChatModel.forDefault())
            .build()
            .prompt()
            .user("What is the answer?")
            .call()
            .entity(Answer.class);
      } catch (JacksonException e) {
        if (catchError) {
          return new Answer("fallback");
        }
        throw e;
      }
    }
  }
}
