package io.temporal.springai;

import static org.junit.jupiter.api.Assertions.*;

import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowOptions;
import io.temporal.serviceclient.WorkflowServiceStubsOptions;
import io.temporal.spring.boot.TemporalOptionsCustomizer;
import io.temporal.springai.chat.TemporalChatClient;
import io.temporal.springai.model.ActivityChatModel;
import io.temporal.springai.plugin.SpringAiPlugin;
import io.temporal.testserver.TestServer;
import io.temporal.worker.Worker;
import io.temporal.worker.WorkerFactory;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.time.Duration;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/** Loads the real Boot 4 and Temporal starter auto-configurations with an offline model. */
class SpringBootAutoConfigurationTest {
  @Test
  @Timeout(30)
  void discoversPluginAndRegistersModelActivityWithStarterWorker() {
    new ApplicationContextRunner()
        .withUserConfiguration(Application.class)
        .withPropertyValues(
            "spring.temporal.connection.target=127.0.0.1:7233",
            "spring.temporal.start-workers=false",
            "spring.temporal.namespace=default",
            "spring.temporal.workers[0].task-queue=boot-ai-test")
        .run(
            context -> {
              assertNull(context.getStartupFailure());
              assertNotNull(context.getBean(SpringAiPlugin.class));
              WorkerFactory factory = context.getBean(WorkerFactory.class);
              Worker worker = factory.getWorker("boot-ai-test");
              worker.registerWorkflowImplementationTypes(ChatWorkflowImpl.class);
              factory.start();
              WorkflowClient client = context.getBean(WorkflowClient.class);
              ChatWorkflow workflow =
                  client.newWorkflowStub(
                      ChatWorkflow.class,
                      WorkflowOptions.newBuilder()
                          .setTaskQueue("boot-ai-test")
                          .setWorkflowExecutionTimeout(Duration.ofSeconds(10))
                          .build());
              assertEquals("boot-pong", workflow.chat());
            });
  }

  @Configuration(proxyBeanMethods = false)
  @EnableAutoConfiguration
  static class Application {
    // Use the starter's normal plugin propagation against a local in-process
    // server. Its separate test-server mode filters service-level plugins.
    @Bean(destroyMethod = "close")
    TestServer.InProcessTestServer testServer() {
      return TestServer.createServer();
    }

    @Bean
    TemporalOptionsCustomizer<WorkflowServiceStubsOptions.Builder> serviceOptions(
        TestServer.InProcessTestServer server) {
      return options -> options.setTarget(null).setChannel(server.getChannel());
    }

    @Bean
    ChatModel chatModel() {
      return prompt ->
          ChatResponse.builder()
              .generations(List.of(new Generation(new AssistantMessage("boot-pong"))))
              .build();
    }
  }

  @WorkflowInterface
  public interface ChatWorkflow {
    @WorkflowMethod
    String chat();
  }

  public static class ChatWorkflowImpl implements ChatWorkflow {
    @Override
    public String chat() {
      return TemporalChatClient.builder(ActivityChatModel.forDefault())
          .build()
          .prompt()
          .user("ping")
          .call()
          .content();
    }
  }
}
