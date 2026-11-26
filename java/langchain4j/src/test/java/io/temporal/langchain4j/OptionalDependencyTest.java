package io.temporal.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.StreamingHandle;
import io.temporal.activity.ActivityOptions;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.common.RetryOptions;
import io.temporal.failure.ActivityFailure;
import io.temporal.failure.ApplicationFailure;
import io.temporal.langchain4j.activity.ChatInput;
import io.temporal.langchain4j.activity.ChatModelActivity;
import io.temporal.langchain4j.activity.ChatModelActivityImpl;
import io.temporal.langchain4j.activity.StreamingChatInput;
import io.temporal.langchain4j.activity.StreamingChatModelActivity;
import io.temporal.langchain4j.activity.StreamingChatModelActivityImpl;
import io.temporal.langchain4j.internal.ChatCodec;
import io.temporal.testing.TestActivityEnvironment;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

/**
 * The plugin and its model activities work without the optional {@code langchain4j-mcp} and {@code
 * temporal-workflowstreams} artifacts on the classpath.
 */
class OptionalDependencyTest {

  @Test
  @Timeout(90)
  void runsTheModelActivitiesWithoutTheOptionalArtifacts(@TempDir Path temporary) throws Exception {
    Path arguments =
        TestClasspath.argumentFile(
            temporary,
            Probe.class,
            entry -> {
              String file = Path.of(entry).getFileName().toString();
              return !file.startsWith("temporal-workflowstreams-")
                  && !file.startsWith("langchain4j-mcp-");
            });
    Path output = temporary.resolve("probe.log");
    Process process =
        new ProcessBuilder(TestClasspath.java(), "@" + arguments)
            .redirectErrorStream(true)
            .redirectOutput(output.toFile())
            .start();
    try {
      assertTrue(process.waitFor(60, TimeUnit.SECONDS), "probe timed out: " + output);
      assertEquals(0, process.exitValue(), Files.readString(output, StandardCharsets.UTF_8));
    } finally {
      if (process.isAlive()) {
        process.destroyForcibly();
        process.waitFor(5, TimeUnit.SECONDS);
      }
    }
  }

  public static final class Probe {
    private static final ActivityOptions OPTIONS =
        ActivityOptions.newBuilder()
            .setStartToCloseTimeout(Duration.ofSeconds(5))
            .setRetryOptions(RetryOptions.newBuilder().setMaximumAttempts(1).build())
            .build();
    private static final ChatInput REQUEST =
        ChatCodec.encode("model", ChatRequest.builder().messages(UserMessage.from("Hi")).build());

    public static void main(String[] args) {
      try {
        run();
        System.exit(0);
      } catch (Throwable e) {
        e.printStackTrace();
        System.exit(1);
      }
    }

    private static void run() {
      for (String absent :
          new String[] {
            "io/temporal/workflowstreams/WorkflowStreamClient.class",
            "dev/langchain4j/mcp/client/McpClient.class"
          }) {
        if (Probe.class.getClassLoader().getResource(absent) != null) {
          throw new AssertionError(absent + " must be absent from the probe's classpath");
        }
      }
      ScriptedChatModel chat =
          new ScriptedChatModel(
              request -> ChatResponse.builder().aiMessage(AiMessage.from("chat works")).build());
      ScriptedStreamingChatModel streaming =
          new ScriptedStreamingChatModel(
              (request, handler) -> {
                handler.onPartialResponse(
                    new PartialResponse("stream"), new PartialResponseContext(new NoHandle()));
                handler.onCompleteResponse(
                    ChatResponse.builder().aiMessage(AiMessage.from("stream works")).build());
              });

      // A worker registers every activity of the plugin, including the MCP activities.
      LangChain4jPlugin plugin =
          LangChain4jPlugin.newBuilder()
              .chatModel("model", chat)
              .streamingChatModel("model", streaming)
              .build();
      TestWorkflowEnvironment workflows =
          TestWorkflowEnvironment.newInstance(
              TestEnvironmentOptions.newBuilder()
                  .setWorkflowClientOptions(
                      WorkflowClientOptions.newBuilder().setPlugins(plugin).build())
                  .build());
      try {
        workflows.newWorker("probe");
        workflows.start();
      } finally {
        workflows.close();
      }

      TestActivityEnvironment environment = TestActivityEnvironment.newInstance();
      try {
        environment.registerActivitiesImplementations(
            new ChatModelActivityImpl(Map.of("model", () -> chat)),
            new StreamingChatModelActivityImpl(Map.of("model", () -> streaming)));
        String text =
            environment.newActivityStub(ChatModelActivity.class, OPTIONS).chat(REQUEST).aiMessage();
        check(text.contains("chat works"), "unexpected chat response: " + text);
        StreamingChatModelActivity activity =
            environment.newActivityStub(StreamingChatModelActivity.class, OPTIONS);
        text = activity.streamingChat(new StreamingChatInput(REQUEST, null)).response().aiMessage();
        check(text.contains("stream works"), "unexpected streaming response: " + text);
        try {
          activity.streamingChat(new StreamingChatInput(REQUEST, "output"));
          throw new AssertionError(
              "live streaming must refuse a missing Workflow Streams artifact");
        } catch (ActivityFailure failure) {
          ApplicationFailure cause = (ApplicationFailure) failure.getCause();
          check(
              cause.isNonRetryable()
                  && StreamingChatModelActivityImpl.STREAM_ERROR.equals(cause.getType())
                  && cause.getOriginalMessage().contains("temporal-workflowstreams"),
              "a missing transport needs an actionable non-retryable failure: " + failure);
        }
      } finally {
        environment.close();
      }
    }

    private static void check(boolean condition, String message) {
      if (!condition) {
        throw new AssertionError(message);
      }
    }
  }

  private static final class NoHandle implements StreamingHandle {
    @Override
    public void cancel() {}

    @Override
    public boolean isCancelled() {
      return false;
    }
  }
}
