package io.temporal.springai;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowOptions;
import io.temporal.springai.activity.ChatModelActivityImpl;
import io.temporal.springai.model.ActivityChatModel;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;

/** Verifies the generation finish reason survives the activity boundary. */
class FinishReasonTest {

  private static final String TASK_QUEUE = "test-spring-ai-finish-reason";

  private TestWorkflowEnvironment testEnv;
  private WorkflowClient client;

  @BeforeEach
  void setUp() {
    testEnv = TestWorkflowEnvironment.newInstance();
    client = testEnv.getWorkflowClient();
  }

  @AfterEach
  void tearDown() {
    testEnv.close();
  }

  @Test
  void finishReason_isPreservedAcrossActivityBoundary() {
    Worker worker = testEnv.newWorker(TASK_QUEUE);
    worker.registerWorkflowImplementationTypes(FinishReasonWorkflowImpl.class);
    worker.registerActivitiesImplementations(new ChatModelActivityImpl(new LengthStubChatModel()));
    testEnv.start();

    FinishReasonWorkflow workflow =
        client.newWorkflowStub(
            FinishReasonWorkflow.class,
            WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build());

    assertEquals("length", workflow.finishReason("Hi"));
  }

  @WorkflowInterface
  public interface FinishReasonWorkflow {
    @WorkflowMethod
    String finishReason(String message);
  }

  public static class FinishReasonWorkflowImpl implements FinishReasonWorkflow {
    @Override
    public String finishReason(String message) {
      ChatResponse response = ActivityChatModel.forDefault().call(new Prompt(message));
      return response.getResult().getMetadata().getFinishReason();
    }
  }

  private static class LengthStubChatModel implements ChatModel {
    @Override
    public ChatResponse call(Prompt prompt) {
      return ChatResponse.builder()
          .generations(
              List.of(
                  new Generation(
                      new AssistantMessage("truncated"),
                      ChatGenerationMetadata.builder().finishReason("length").build())))
          .build();
    }

    @Override
    public reactor.core.publisher.Flux<ChatResponse> stream(Prompt prompt) {
      throw new UnsupportedOperationException();
    }
  }
}
