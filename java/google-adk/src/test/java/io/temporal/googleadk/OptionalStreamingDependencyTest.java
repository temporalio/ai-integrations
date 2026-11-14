package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.*;

import com.google.adk.models.LlmRequest;
import io.temporal.activity.ActivityOptions;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.common.RetryOptions;
import io.temporal.failure.ActivityFailure;
import io.temporal.failure.ApplicationFailure;
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
import java.time.Duration;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

/** The model Activity works without the optional Workflow Streams artifact on the classpath. */
class OptionalStreamingDependencyTest {
  @Test
  @Timeout(90)
  void executesTheDefaultModelActivityWithoutWorkflowStreamsAtRuntime(@TempDir Path temporary)
      throws Exception {
    String classpathFile = System.getProperty("googleadk.test.classpath-file");
    assertNotNull(classpathFile, "run through Gradle; see build.gradle");
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
    assertNotEquals(original, isolated, "the test classpath carries Workflow Streams");
    // Pad past the Windows command-line limit (32,767 characters) to keep the argument file
    // approach honest on every platform.
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
    Process process =
        new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(), "@" + arguments)
            .redirectErrorStream(true)
            .redirectOutput(output.toFile())
            .start();
    try {
      assertTrue(process.waitFor(60, TimeUnit.SECONDS), "consumer timed out: " + output);
      assertEquals(0, process.exitValue(), Files.readString(output, StandardCharsets.UTF_8));
    } finally {
      if (process.isAlive()) {
        process.destroyForcibly();
        process.waitFor(5, TimeUnit.SECONDS);
      }
    }
  }

  public static final class DefaultActivityProbe {
    private static final ActivityOptions OPTIONS =
        ActivityOptions.newBuilder()
            .setStartToCloseTimeout(Duration.ofSeconds(3))
            .setRetryOptions(RetryOptions.newBuilder().setMaximumAttempts(1).build())
            .build();

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
      // The plugin and the Activity implementation load without the optional artifact.
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
        ModelActivity activity = environment.newActivityStub(ModelActivity.class, OPTIONS);
        LlmRequest request = LlmRequest.builder().model("model").build();
        for (InvokeModelInput input :
            new InvokeModelInput[] {
              new InvokeModelInput("model", request, true),
              // A topic without SSE streaming never publishes, so it needs no artifact either.
              new InvokeModelInput("model", request, false, "output")
            }) {
          String text =
              activity.invokeModel(input).responses().get(0).content().orElseThrow().text();
          if (!"default activity works".equals(text)) {
            throw new AssertionError("unexpected model response: " + text);
          }
        }
        try {
          activity.invokeModel(new InvokeModelInput("model", request, true, "output"));
          throw new AssertionError(
              "live streaming must reject a missing Workflow Streams artifact");
        } catch (ActivityFailure failure) {
          ApplicationFailure cause = (ApplicationFailure) failure.getCause();
          if (!cause.isNonRetryable()
              || !FailureTypes.STREAM_ERROR.equals(cause.getType())
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
