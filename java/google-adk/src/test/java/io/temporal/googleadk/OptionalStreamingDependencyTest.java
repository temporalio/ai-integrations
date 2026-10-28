package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.*;

import com.google.adk.models.LlmRequest;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.googleadk.activity.InvokeModelInput;
import io.temporal.googleadk.activity.ModelActivity;
import io.temporal.googleadk.activity.ModelActivityImpl;
import io.temporal.googleadk.converter.GoogleAdkDataConverter;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.testing.TestActivityEnvironment;
import io.temporal.testing.TestEnvironmentOptions;
import java.io.File;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class OptionalStreamingDependencyTest {
  @Test
  @Timeout(30)
  void executesTheDefaultModelActivityWithoutWorkflowStreamsAtRuntime() throws Exception {
    String original = System.getProperty("googleadk.test.classpath");
    assertNotNull(original);
    String isolated =
        Arrays.stream(original.split(Pattern.quote(File.pathSeparator)))
            .filter(
                entry ->
                    !Path.of(entry)
                        .getFileName()
                        .toString()
                        .startsWith("temporal-workflowstreams-"))
            .collect(Collectors.joining(File.pathSeparator));
    assertNotEquals(original, isolated);
    String executable = System.getProperty("os.name").startsWith("Windows") ? "java.exe" : "java";
    Process process =
        new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", executable).toString(),
                "-cp",
                isolated,
                DefaultActivityProbe.class.getName())
            .redirectErrorStream(true)
            .start();
    try {
      assertTrue(process.waitFor(20, TimeUnit.SECONDS));
      assertEquals(
          0,
          process.exitValue(),
          () -> {
            try {
              return new String(
                  process.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
            } catch (java.io.IOException e) {
              return e.toString();
            }
          });
    } finally {
      process.destroyForcibly();
    }
  }

  public static final class DefaultActivityProbe {
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
      if (DefaultActivityProbe.class
              .getClassLoader()
              .getResource("io/temporal/workflowstreams/WorkflowStreamClient.class")
          != null) {
        throw new AssertionError("Workflow Streams must be absent from the consumer classpath");
      }
      GoogleAdkPlugin.newDefaultInstance();
      TestActivityEnvironment environment =
          TestActivityEnvironment.newInstance(
              TestEnvironmentOptions.newBuilder()
                  .setWorkflowClientOptions(
                      WorkflowClientOptions.newBuilder()
                          .setDataConverter(GoogleAdkDataConverter.newDefaultInstance())
                          .build())
                  .build());
      try {
        environment.registerActivitiesImplementations(
            new ModelActivityImpl(
                name -> ScriptedModel.of(name, ScriptedModel.text("default activity works"))));
        var result =
            environment
                .newActivityStub(
                    ModelActivity.class,
                    io.temporal.activity.ActivityOptions.newBuilder()
                        .setStartToCloseTimeout(java.time.Duration.ofSeconds(3))
                        .setRetryOptions(
                            io.temporal.common.RetryOptions.newBuilder()
                                .setMaximumAttempts(1)
                                .build())
                        .build())
                .invokeModel(
                    new InvokeModelInput(
                        "model", LlmRequest.builder().model("model").build(), true));
        if (!"default activity works"
            .equals(result.responses().get(0).content().orElseThrow().text())) {
          throw new AssertionError("unexpected model response");
        }
        try {
          environment
              .newActivityStub(
                  ModelActivity.class,
                  io.temporal.activity.ActivityOptions.newBuilder()
                      .setStartToCloseTimeout(java.time.Duration.ofSeconds(3))
                      .setRetryOptions(
                          io.temporal.common.RetryOptions.newBuilder()
                              .setMaximumAttempts(1)
                              .build())
                      .build())
              .invokeModel(
                  new InvokeModelInput(
                      "model", LlmRequest.builder().model("model").build(), true, "output"));
          throw new AssertionError(
              "live streaming must reject a missing Workflow Streams artifact");
        } catch (io.temporal.failure.ActivityFailure failure) {
          var cause = (io.temporal.failure.ApplicationFailure) failure.getCause();
          if (!cause.isNonRetryable()
              || !cause.getOriginalMessage().contains("matching io.temporal:temporal-sdk")) {
            throw new AssertionError(
                "missing transport needs an actionable non-retryable failure", failure);
          }
        }
      } finally {
        environment.close();
      }
    }
  }
}
