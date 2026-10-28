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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

class OptionalStreamingDependencyTest {
  @Test
  @Timeout(30)
  void executesTheDefaultModelActivityWithoutWorkflowStreamsAtRuntime(@TempDir Path temporary)
      throws Exception {
    String classpathFile = System.getProperty("googleadk.test.classpath-file");
    assertNotNull(classpathFile);
    String original = Files.readString(Path.of(classpathFile), StandardCharsets.UTF_8);
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
    String repeatedEntry = isolated.split(Pattern.quote(File.pathSeparator))[0];
    while (isolated.length() < 40_000) {
      isolated += File.pathSeparator + repeatedEntry;
    }
    Path arguments = temporary.resolve("probe.args");
    Files.writeString(
        arguments,
        "-cp\n\""
            + isolated.replace("\\", "\\\\").replace("\"", "\\\"")
            + "\"\n"
            + DefaultActivityProbe.class.getName()
            + "\n",
        StandardCharsets.UTF_8);
    Path output = temporary.resolve("probe.log");
    String executable = System.getProperty("os.name").startsWith("Windows") ? "java.exe" : "java";
    Process process =
        new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", executable).toString(),
                "@" + arguments)
            .redirectErrorStream(true)
            .redirectOutput(output.toFile())
            .start();
    try {
      assertTrue(process.waitFor(20, TimeUnit.SECONDS), "consumer timed out: " + output);
      assertEquals(0, process.exitValue(), Files.readString(output, StandardCharsets.UTF_8));
    } finally {
      if (process.isAlive()) {
        process.destroyForcibly();
        process.waitFor(5, TimeUnit.SECONDS);
      }
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
