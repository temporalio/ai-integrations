package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.agents.LlmAgent;
import com.google.adk.agents.ParallelAgent;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.runner.Runner;
import com.google.adk.tools.Annotations.Schema;
import com.google.adk.tools.FunctionTool;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanContext;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.TraceFlags;
import io.opentelemetry.api.trace.TraceState;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import io.temporal.activity.ActivityInterface;
import io.temporal.activity.ActivityMethod;
import io.temporal.activity.ActivityOptions;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.client.WorkflowFailedException;
import io.temporal.client.WorkflowOptions;
import io.temporal.client.WorkflowStub;
import io.temporal.common.WorkflowExecutionHistory;
import io.temporal.common.interceptors.WorkerInterceptorBase;
import io.temporal.common.interceptors.WorkflowInboundCallsInterceptor;
import io.temporal.common.interceptors.WorkflowInboundCallsInterceptorBase;
import io.temporal.failure.ApplicationFailure;
import io.temporal.googleadk.model.TemporalModel;
import io.temporal.googleadk.testing.ScriptedModel;
import io.temporal.googleadk.tool.ActivityTools;
import io.temporal.serviceclient.WorkflowServiceStubsOptions;
import io.temporal.testing.TestEnvironmentOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import io.temporal.worker.WorkerFactory;
import io.temporal.worker.WorkerFactoryOptions;
import io.temporal.workflow.Async;
import io.temporal.workflow.Promise;
import io.temporal.workflow.QueryMethod;
import io.temporal.workflow.SignalMethod;
import io.temporal.workflow.UpdateMethod;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class NativeTracingWorkflowTest {
  private static final String MODEL = "scripted";

  @ParameterizedTest
  @ValueSource(
      strings = {
        "tools",
        "eviction",
        "unfinished",
        "nested",
        "parallel",
        "overlap",
        "update",
        "signal",
        "retry",
        "failure",
        "cancel",
        "legacy",
        "multiple"
      })
  void nativeTracingUsesDocumentedBootstrapInAnIsolatedProcess(String scenario) throws Exception {
    Path output = Files.createTempFile("adk-native-tracing-", ".log");
    Path arguments = Files.createTempFile("adk-native-tracing-", ".args");
    String classpath =
        Files.readString(Path.of(System.getProperty("googleadk.test.classpath-file")));
    Files.writeString(
        arguments,
        "-cp\n"
            + argument(classpath)
            + "\n"
            + argument(Bootstrap.class.getName())
            + "\n"
            + argument(scenario)
            + "\n");
    Process process = null;
    try {
      process =
          new ProcessBuilder(
                  Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                  "@" + arguments)
              .redirectErrorStream(true)
              .redirectOutput(output.toFile())
              .start();
      assertTrue(process.waitFor(70, TimeUnit.SECONDS), () -> read(output));
      assertEquals(0, process.exitValue(), () -> read(output));
      assertTrue(read(output).contains("PASS " + scenario), () -> read(output));
    } finally {
      if (process != null) process.destroyForcibly().waitFor(10, TimeUnit.SECONDS);
      Files.deleteIfExists(arguments);
      Files.deleteIfExists(output);
    }
  }

  private static String argument(String value) {
    return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
  }

  private static String read(Path path) {
    try {
      return Files.readString(path);
    } catch (Exception failure) {
      return failure.toString();
    }
  }

  public static final class Bootstrap {
    public static void main(String[] args) throws Exception {
      String mode = args[0];
      InMemorySpanExporter nativeExporter = InMemorySpanExporter.create();
      InMemorySpanExporter appExporter = InMemorySpanExporter.create();
      try (OpenTelemetrySdk application =
              OpenTelemetrySdk.builder()
                  .setTracerProvider(
                      SdkTracerProvider.builder()
                          .addSpanProcessor(SimpleSpanProcessor.create(appExporter))
                          .build())
                  .build();
          GoogleAdkTracing tracing = GoogleAdkTracing.create(nativeExporter)) {
        GlobalOpenTelemetry.set(GoogleAdkTracing.wrap(application));
        if (mode.equals("multiple")) {
          run("tools", tracing, nativeExporter, appExporter);
          int originalSize = nativeExporter.getFinishedSpanItems().size();
          appExporter.reset();
          InMemorySpanExporter secondExporter = InMemorySpanExporter.create();
          try (GoogleAdkTracing second = GoogleAdkTracing.create(secondExporter)) {
            run("tools", second, secondExporter, appExporter);
            assertEquals(
                originalSize,
                nativeExporter.getFinishedSpanItems().size(),
                "Worker B must not use A's exporter");
          }
          appExporter.reset();
          try (Fixture unconfigured =
              new Fixture(null, ScriptedModel.of(MODEL, ScriptedModel.text("done")), "plain")) {
            assertEquals("done", unconfigured.workflow.run("plain"));
          }
          assertTrue(
              appExporter.getFinishedSpanItems().stream()
                  .anyMatch(s -> "invoke_agent".equals(operation(s))),
              "a Worker that did not opt in keeps the application's native ADK provider");
          flush(tracing);
          assertEquals(originalSize, nativeExporter.getFinishedSpanItems().size());
        } else run(mode, tracing, nativeExporter, appExporter);
      }
      System.out.println("PASS " + mode);
    }

    private static void run(
        String mode,
        GoogleAdkTracing tracing,
        InMemorySpanExporter nativeExporter,
        InMemorySpanExporter appExporter)
        throws Exception {
      boolean gated =
          List.of("eviction", "restart", "unfinished", "nested", "cancel").contains(mode);
      CountDownLatch entered = new CountDownLatch(1);
      CountDownLatch release = new CountDownLatch(gated ? 1 : 0);
      AtomicInteger requests = new AtomicInteger();
      LookupImpl.attempts.set(0);
      LookupImpl.retry = mode.equals("retry");
      ScriptedModel model =
          ScriptedModel.responding(
              MODEL,
              request -> {
                int call = requests.incrementAndGet();
                Span span =
                    GlobalOpenTelemetry.getTracer("application-model")
                        .spanBuilder("provider-call")
                        .startSpan();
                try (Scope scope = span.makeCurrent()) {
                  if (!mode.equals("nested") || call == 3) {
                    entered.countDown();
                    try {
                      if (!release.await(25, TimeUnit.SECONDS))
                        throw new IllegalStateException("provider gate timed out");
                    } catch (InterruptedException interrupted) {
                      Thread.currentThread().interrupt();
                      throw new IllegalStateException(interrupted);
                    }
                  }
                  if (mode.equals("failure"))
                    throw ApplicationFailure.newNonRetryableFailure(
                        "invalid model request", "InvalidRequest");
                  if (mode.equals("retry") && call == 1)
                    throw ApplicationFailure.newFailure("temporary provider failure", "Transient");
                  if (mode.equals("tools") || mode.equals("retry")) return response(request);
                  if (mode.equals("nested")
                      && request.tools().containsKey("nestedTurn")
                      && TestSupport.functionResponse(request, "nestedTurn").isEmpty())
                    return ScriptedModel.functionCall("nestedTurn", Map.of());
                  return ScriptedModel.text("done");
                } catch (RuntimeException failure) {
                  span.setStatus(StatusCode.ERROR, failure.getMessage());
                  throw failure;
                } finally {
                  span.end();
                }
              });
      try (Fixture fixture = new Fixture(mode.equals("legacy") ? null : tracing, model, mode)) {
        TracedWorkflow workflow = fixture.workflow;
        if (mode.equals("tools")) {
          SpanContext incoming =
              SpanContext.createFromRemoteParent(
                  "11111111111111111111111111111111",
                  "2222222222222222",
                  TraceFlags.getSampled(),
                  TraceState.builder().put("vendor", "test").build());
          try (Scope scope = Context.root().with(Span.wrap(incoming)).makeCurrent()) {
            WorkflowClient.start(workflow::run, mode);
          }
        } else WorkflowClient.start(workflow::run, mode);
        if (mode.equals("update")) assertEquals("done", workflow.ask());
        if (mode.equals("signal")) workflow.askSignal();
        assertTrue(entered.await(10, TimeUnit.SECONDS));
        if (gated) {
          if (mode.equals("eviction") || mode.equals("nested")) fixture.evict();
          if (mode.equals("restart")) fixture.restart();
          if (mode.equals("unfinished")) {
            WorkflowExecutionHistory partial =
                fixture
                    .environment
                    .getWorkflowClient()
                    .fetchHistory(WorkflowStub.fromTyped(workflow).getExecution().getWorkflowId());
            try (Fixture replay = new Fixture(tracing, model, "replay")) {
              replay.worker.replayWorkflowExecution(partial);
            }
          }
          flush(tracing);
          assertEquals(
              0,
              nativeExporter.getFinishedSpanItems().size(),
              "unfinished turns, nested calls and eviction must not export spans");
          if (mode.equals("cancel")) {
            WorkflowStub.fromTyped(workflow).cancel();
            assertThrows(
                WorkflowFailedException.class,
                () -> WorkflowStub.fromTyped(workflow).getResult(String.class));
          }
          release.countDown();
        }
        if (mode.equals("failure"))
          assertThrows(
              WorkflowFailedException.class,
              () -> WorkflowStub.fromTyped(workflow).getResult(String.class));
        else if (!mode.equals("cancel"))
          assertEquals("done", WorkflowStub.fromTyped(workflow).getResult(String.class));
        flush(tracing);
        List<SpanData> spans = nativeExporter.getFinishedSpanItems();
        if (mode.equals("tools")) {
          assertTrue(
              spans.stream()
                  .anyMatch(
                      span ->
                          "invoke_agent".equals(operation(span))
                              && "assistant"
                                  .equals(
                                      span.getAttributes()
                                          .get(AttributeKey.stringKey("gen_ai.agent.name")))));
          assertTrue(
              spans.stream()
                  .anyMatch(
                      span ->
                          "scripted"
                              .equals(
                                  span.getAttributes()
                                      .get(AttributeKey.stringKey("gen_ai.request.model")))));
          assertTrue(
              spans.stream()
                  .anyMatch(
                      span ->
                          "session"
                              .equals(
                                  span.getAttributes()
                                      .get(
                                          AttributeKey.stringKey("gcp.vertex.agent.session_id")))));
          assertTrue(
              spans.stream()
                  .anyMatch(
                      span -> {
                        String response =
                            span.getAttributes()
                                .get(AttributeKey.stringKey("gcp.vertex.agent.llm_response"));
                        return response != null && response.contains("done");
                      }));
          assertTrue(
              spans.stream().anyMatch(span -> span.getParentSpanId().equals("2222222222222222")));
          for (SpanData span : spans) {
            assertEquals("11111111111111111111111111111111", span.getTraceId());
            assertEquals("test", span.getSpanContext().getTraceState().get("vendor"));
            if ("call_llm".equals(operation(span)))
              assertTrue(
                  spans.stream()
                      .anyMatch(
                          parent ->
                              parent.getSpanId().equals(span.getParentSpanId())
                                  && "invoke_agent".equals(operation(parent))));
          }
          for (SpanData span : appExporter.getFinishedSpanItems())
            assertEquals("test", span.getSpanContext().getTraceState().get("vendor"));
        }
        if (!mode.equals("legacy")) {
          assertTrue(
              spans.stream().anyMatch(s -> "invoke_agent".equals(operation(s))), spans.toString());
          if (!mode.equals("cancel"))
            assertTrue(
                spans.stream().anyMatch(s -> "call_llm".equals(operation(s))), spans.toString());
          if (mode.equals("tools") || mode.equals("retry")) {
            for (String tool : List.of("lookup", "doubleIt"))
              assertTrue(
                  spans.stream()
                      .anyMatch(
                          s ->
                              "execute_tool".equals(operation(s))
                                  && tool.equals(
                                      s.getAttributes()
                                          .get(AttributeKey.stringKey("gen_ai.tool.name")))),
                  spans.toString());
          }
          assertEquals(
              spans.size(),
              spans.stream().map(SpanData::getSpanId).distinct().count(),
              "each native operation exports once");
          if (!mode.equals("overlap"))
            assertEquals(1, spans.stream().map(SpanData::getTraceId).distinct().count());
          for (SpanData span : appExporter.getFinishedSpanItems()) {
            if (span.getName().equals("provider-call") || span.getName().equals("lookup-activity"))
              assertTrue(
                  spans.stream()
                      .anyMatch(parent -> parent.getSpanId().equals(span.getParentSpanId())),
                  "Activity/RxJava context lost: " + span);
          }
          if (mode.equals("failure"))
            assertTrue(
                spans.stream().anyMatch(s -> s.getStatus().getStatusCode() == StatusCode.ERROR));
          if (mode.equals("retry")) {
            assertEquals(4, requests.get());
            assertEquals(2, LookupImpl.attempts.get());
            assertEquals(
                2,
                appExporter.getFinishedSpanItems().stream()
                    .filter(s -> s.getStatus().getStatusCode() == StatusCode.ERROR)
                    .count());
          }
        } else assertEquals(0, spans.size());
        if (mode.equals("eviction") || mode.equals("nested"))
          assertTrue(
              fixture.executions.get() >= 3,
              "a parked second Workflow must force the first to replay");
        int count = requests.get();
        WorkflowExecutionHistory history =
            fixture
                .environment
                .getWorkflowClient()
                .fetchHistory(WorkflowStub.fromTyped(workflow).getExecution().getWorkflowId());
        try (Fixture replay = new Fixture(tracing, model, "replay")) {
          replay.worker.replayWorkflowExecution(history);
        }
        flush(tracing);
        assertEquals(
            spans.size(),
            nativeExporter.getFinishedSpanItems().size(),
            "a fresh Worker replays completed and legacy history without exporting");
        assertEquals(count, requests.get(), "replay makes no provider calls");
        Span unrelated =
            GlobalOpenTelemetry.getTracer("application").spanBuilder("unrelated").startSpan();
        unrelated.end();
        Span outside =
            GlobalOpenTelemetry.getTracer("gcp.vertex.agent")
                .spanBuilder("outside-workflow")
                .startSpan();
        outside.end();
        assertTrue(
            appExporter.getFinishedSpanItems().stream()
                .anyMatch(s -> s.getName().equals("unrelated")));
        assertTrue(
            appExporter.getFinishedSpanItems().stream()
                .anyMatch(s -> s.getName().equals("outside-workflow")));
        flush(tracing);
        assertEquals(spans.size(), nativeExporter.getFinishedSpanItems().size());
      } finally {
        release.countDown();
      }
    }

    private static final class Fixture implements AutoCloseable {
      final TestWorkflowEnvironment environment;
      final Worker worker;
      final TracedWorkflow workflow;
      final AtomicInteger executions = new AtomicInteger();
      final List<WorkerFactory> extras = new ArrayList<>();

      Fixture(GoogleAdkTracing tracing, ScriptedModel model, String mode) {
        GoogleAdkPlugin.Builder plugin = GoogleAdkPlugin.newBuilder().model(MODEL, model);
        if (tracing != null) plugin.tracing(tracing);
        WorkerFactoryOptions options =
            WorkerFactoryOptions.newBuilder()
                .setWorkflowCacheSize(List.of("eviction", "nested").contains(mode) ? 1 : 600)
                .setWorkflowHostLocalTaskQueueScheduleToStartTimeout(Duration.ofMillis(100))
                .setWorkerInterceptors(
                    new WorkerInterceptorBase() {
                      @Override
                      public WorkflowInboundCallsInterceptor interceptWorkflow(
                          WorkflowInboundCallsInterceptor next) {
                        return new WorkflowInboundCallsInterceptorBase(next) {
                          @Override
                          public WorkflowOutput execute(WorkflowInput input) {
                            executions.incrementAndGet();
                            return super.execute(input);
                          }
                        };
                      }
                    })
                .build();
        TestEnvironmentOptions.Builder environmentOptions =
            TestEnvironmentOptions.newBuilder()
                .setWorkerFactoryOptions(options)
                .setWorkflowClientOptions(
                    WorkflowClientOptions.newBuilder().setPlugins(plugin.build()).build());
        String target = System.getProperty("googleadk.test.target");
        if (target != null)
          environmentOptions
              .setUseExternalService(true)
              .setTarget(target)
              .setWorkflowServiceStubsOptions(
                  WorkflowServiceStubsOptions.newBuilder().setTarget(target).build());
        environment = TestWorkflowEnvironment.newInstance(environmentOptions.build());
        worker = environment.newWorker("native-tracing");
        worker.registerWorkflowImplementationTypes(TracedWorkflowImpl.class);
        worker.registerActivitiesImplementations(new LookupImpl());
        if (mode.equals("restart")) {
          WorkerFactory activityFactory =
              WorkerFactory.newInstance(environment.getWorkflowClient());
          activityFactory.newWorker("native-models");
          activityFactory.start();
          extras.add(activityFactory);
        }
        environment.start();
        workflow =
            environment
                .getWorkflowClient()
                .newWorkflowStub(
                    TracedWorkflow.class,
                    WorkflowOptions.newBuilder().setTaskQueue("native-tracing").build());
      }

      void evict() {
        TracedWorkflow parked =
            environment
                .getWorkflowClient()
                .newWorkflowStub(
                    TracedWorkflow.class,
                    WorkflowOptions.newBuilder().setTaskQueue("native-tracing").build());
        WorkflowClient.start(parked::run, "park");
        assertTrue(parked.ready());
      }

      void restart() {
        environment.getWorkerFactory().shutdownNow();
        environment.getWorkerFactory().awaitTermination(10, TimeUnit.SECONDS);
        assertTrue(environment.getWorkerFactory().isTerminated());
        WorkerFactory resumed = WorkerFactory.newInstance(environment.getWorkflowClient());
        resumed
            .newWorker("native-tracing")
            .registerWorkflowImplementationTypes(TracedWorkflowImpl.class);
        resumed.start();
        extras.add(resumed);
      }

      @Override
      public void close() {
        extras.forEach(WorkerFactory::shutdownNow);
        extras.forEach(factory -> factory.awaitTermination(10, TimeUnit.SECONDS));
        environment.close();
      }
    }

    private static void flush(GoogleAdkTracing tracing) {
      assertTrue(tracing.forceFlush().join(10, TimeUnit.SECONDS).isSuccess());
    }

    private static String operation(SpanData span) {
      return span.getAttributes().get(AttributeKey.stringKey("gen_ai.operation.name"));
    }

    private static LlmResponse response(LlmRequest request) {
      if (TestSupport.functionResponse(request, "lookup").isEmpty())
        return ScriptedModel.functionCall("lookup", Map.of("key", "k1"));
      if (TestSupport.functionResponse(request, "doubleIt").isEmpty())
        return ScriptedModel.functionCall("doubleIt", Map.of("value", 2));
      return ScriptedModel.text("done");
    }
  }

  @WorkflowInterface
  public interface TracedWorkflow {
    @WorkflowMethod
    String run(String mode);

    @QueryMethod
    boolean ready();

    @UpdateMethod
    String ask();

    @SignalMethod
    void askSignal();
  }

  @ActivityInterface
  public interface Lookup {
    @ActivityMethod
    String lookup(String key);
  }

  public static final class LookupImpl implements Lookup {
    static final AtomicInteger attempts = new AtomicInteger();
    static boolean retry;

    @Override
    public String lookup(String key) {
      Span span =
          GlobalOpenTelemetry.getTracer("application-tools")
              .spanBuilder("lookup-activity")
              .startSpan();
      try (Scope scope = span.makeCurrent()) {
        if (retry && attempts.incrementAndGet() == 1) {
          span.setStatus(StatusCode.ERROR, "retry tool");
          throw ApplicationFailure.newFailure("retry tool", "Transient");
        }
        return "value-" + key;
      } finally {
        span.end();
      }
    }
  }

  public static final class PureTools {
    @Schema(description = "Double a number")
    public int doubleIt(@Schema(name = "value") int value) {
      return value * 2;
    }
  }

  public static final class NestedTool {
    @Schema(description = "Run another native ADK turn")
    public String nestedTurn() {
      return TracedWorkflowImpl.turn("inner", false);
    }
  }

  public static final class TracedWorkflowImpl implements TracedWorkflow {
    private boolean finished;

    @Override
    public boolean ready() {
      return true;
    }

    @Override
    public String ask() {
      String result = turn("update", false);
      finished = true;
      return result;
    }

    @Override
    public void askSignal() {
      turn("signal", false);
      finished = true;
    }

    @Override
    public String run(String mode) {
      if (mode.equals("park")) {
        Workflow.await(() -> false);
        return "parked";
      }
      if (mode.equals("update") || mode.equals("signal")) {
        Workflow.await(() -> finished);
        return "done";
      }
      if (mode.equals("restart")) {
        TemporalModel model =
            TemporalModel.newBuilder(MODEL)
                .activityOptions(
                    ActivityOptions.newBuilder()
                        .setTaskQueue("native-models")
                        .setStartToCloseTimeout(Duration.ofSeconds(40))
                        .build())
                .build();
        Runner runner =
            TemporalAdk.runnerBuilder()
                .appName("native-tracing")
                .agent(LlmAgent.builder().name("restart").model(model).build())
                .build();
        return TestSupport.lastText(
            TemporalAdk.run(runner, "user", "session", Content.fromParts(Part.fromText("hello"))));
      }
      if (mode.equals("nested")) {
        Runner runner =
            TemporalAdk.runnerBuilder()
                .appName("native-tracing")
                .agent(
                    LlmAgent.builder()
                        .name("outer")
                        .model(TemporalModel.create(MODEL))
                        .tools(FunctionTool.create(new NestedTool(), "nestedTurn"))
                        .build())
                .build();
        return TestSupport.lastText(
            TemporalAdk.run(runner, "user", "session", Content.fromParts(Part.fromText("hello"))));
      }
      if (mode.equals("overlap")) {
        Promise<String> first = Async.function(() -> turn("first", false));
        Promise<String> second = Async.function(() -> turn("second", false));
        assertEquals("done", first.get());
        return second.get();
      }
      if (mode.equals("parallel")) {
        LlmAgent first = agent("first", false);
        LlmAgent second = agent("second", false);
        ParallelAgent parallel =
            ParallelAgent.builder().name("parallel").subAgents(first, second).build();
        Runner runner =
            TemporalAdk.runnerBuilder().appName("native-tracing").agent(parallel).build();
        TemporalAdk.run(runner, "user", "session", Content.fromParts(Part.fromText("hello")));
        return "done";
      }
      return turn("assistant", mode.equals("tools") || mode.equals("retry"));
    }

    private static LlmAgent agent(String name, boolean tools) {
      LlmAgent.Builder builder = LlmAgent.builder().name(name).model(TemporalModel.create(MODEL));
      if (tools) {
        Lookup lookup =
            Workflow.newActivityStub(
                Lookup.class,
                ActivityOptions.newBuilder()
                    .setStartToCloseTimeout(Duration.ofSeconds(10))
                    .build());
        builder.tools(
            FunctionTool.create(new PureTools(), "doubleIt"),
            ActivityTools.create(lookup, "lookup"));
      }
      return builder.build();
    }

    private static String turn(String name, boolean tools) {
      Runner runner =
          TemporalAdk.runnerBuilder().appName("native-tracing").agent(agent(name, tools)).build();
      return TestSupport.lastText(
          TemporalAdk.run(runner, "user", "session", Content.fromParts(Part.fromText("hello"))));
    }
  }
}
