package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
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
import io.opentelemetry.sdk.resources.Resource;
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
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Each scenario runs the documented bootstrap and real ADK agents in its own JVM, because {@code
 * GlobalOpenTelemetry.set} is one-shot per process.
 */
class NativeTracingWorkflowTest {
  private static final String MODEL = "scripted";
  private static final String SERVICE = "native-tracing-test";
  private static final String TRACE_ID = "11111111111111111111111111111111";
  private static final String PARENT_SPAN_ID = "2222222222222222";

  /** The scenarios; the flags drive the shared harness in {@link Bootstrap#run}. */
  enum Scenario {
    /** Native attributes, parentage, incoming tracestate, resource, pure and Activity tools. */
    TOOLS(false, false, true),
    /** A parked second Workflow evicts the first mid-turn; it replays, then exports once. */
    EVICTION(true, true, false),
    /** A history that ends mid-turn replays on a fresh Worker without exporting. */
    UNFINISHED(true, false, false),
    /** A tool starts a nested turn, which shares the outer turn's buffer across an eviction. */
    NESTED(true, true, false),
    /** A ParallelAgent fans out on several Workflow threads. */
    PARALLEL(false, false, false),
    /** Two root turns overlap on separate Workflow threads and keep separate traces. */
    OVERLAP(false, false, false),
    /** The turn runs in an Update handler. */
    UPDATE(false, false, false),
    /** The turn runs in a Signal handler. */
    SIGNAL(false, false, false),
    /** The model and a tool Activity fail once and are retried. */
    RETRY(false, false, true),
    /** The model fails permanently; the failing span carries an error status. */
    FAILURE(false, false, false),
    /** The Workflow is canceled while the model Activity runs. */
    CANCEL(true, false, false),
    /** A Worker without tracing exports nothing natively. */
    LEGACY(false, false, false),
    /** Two tracing instances and an unconfigured Worker keep separate exporters. */
    MULTIPLE(false, false, false),
    /** The turn starts inside a timer callback, which enters through the SDK's callback thread. */
    CALLBACK(false, false, false),
    /**
     * The Worker factory is replaced while a model Activity is pending. Needs a real Temporal
     * service ({@code -Dgoogleadk.test.target=host:port}), so CI excludes it.
     */
    RESTART(true, false, false);

    /** The model blocks until the test releases it, so the Workflow can be disturbed mid-turn. */
    final boolean gated;

    /** The Worker caches one Workflow, so a parked second Workflow evicts the first. */
    final boolean evicts;

    /** The agent has a pure tool and an Activity tool. */
    final boolean tools;

    Scenario(boolean gated, boolean evicts, boolean tools) {
      this.gated = gated;
      this.evicts = evicts;
      this.tools = tools;
    }

    /** The Workflow input that selects this scenario's agent. */
    String mode() {
      return name().toLowerCase(Locale.ROOT);
    }
  }

  @ParameterizedTest
  @EnumSource(value = Scenario.class, mode = EnumSource.Mode.EXCLUDE, names = "RESTART")
  void nativeTracingUsesDocumentedBootstrapInAnIsolatedProcess(
      Scenario scenario, @TempDir Path temporary) throws Exception {
    String classpathFile = System.getProperty("googleadk.test.classpath-file");
    assertNotNull(classpathFile, "run through Gradle; see build.gradle");
    Path arguments = temporary.resolve("bootstrap.args");
    Files.writeString(
        arguments,
        "-cp\n"
            + argument(Files.readString(Path.of(classpathFile)))
            + "\n"
            + argument(Bootstrap.class.getName())
            + "\n"
            + argument(scenario.name())
            + "\n");
    Path output = temporary.resolve("bootstrap.log");
    Process process =
        new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(), "@" + arguments)
            .redirectErrorStream(true)
            .redirectOutput(output.toFile())
            .start();
    try {
      assertTrue(process.waitFor(70, TimeUnit.SECONDS), () -> read(output));
      assertEquals(0, process.exitValue(), () -> read(output));
      assertTrue(read(output).contains("PASS " + scenario), () -> read(output));
    } finally {
      if (process.isAlive()) process.destroyForcibly().waitFor(10, TimeUnit.SECONDS);
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
    public static void main(String[] args) {
      try {
        Scenario scenario = Scenario.valueOf(args[0]);
        InMemorySpanExporter nativeExporter = InMemorySpanExporter.create();
        InMemorySpanExporter appExporter = InMemorySpanExporter.create();
        try (OpenTelemetrySdk application =
                OpenTelemetrySdk.builder()
                    .setTracerProvider(
                        SdkTracerProvider.builder()
                            .addSpanProcessor(SimpleSpanProcessor.create(appExporter))
                            .build())
                    .build();
            GoogleAdkTracing tracing = GoogleAdkTracing.create(nativeExporter, resource())) {
          GlobalOpenTelemetry.set(GoogleAdkTracing.wrap(application));
          if (scenario == Scenario.MULTIPLE) {
            multiple(tracing, nativeExporter, appExporter);
          } else {
            run(scenario, tracing, nativeExporter, appExporter);
          }
        }
        System.out.println("PASS " + scenario);
        System.exit(0);
      } catch (Throwable failure) {
        failure.printStackTrace();
        System.exit(1);
      }
    }

    private static Resource resource() {
      return Resource.builder().put("service.name", SERVICE).build();
    }

    private static void multiple(
        GoogleAdkTracing tracing,
        InMemorySpanExporter nativeExporter,
        InMemorySpanExporter appExporter)
        throws Exception {
      run(Scenario.TOOLS, tracing, nativeExporter, appExporter);
      int exported = nativeExporter.getFinishedSpanItems().size();
      appExporter.reset();
      InMemorySpanExporter secondExporter = InMemorySpanExporter.create();
      try (GoogleAdkTracing second = GoogleAdkTracing.create(secondExporter, resource())) {
        run(Scenario.TOOLS, second, secondExporter, appExporter);
        assertEquals(
            exported,
            nativeExporter.getFinishedSpanItems().size(),
            "Worker B must not use A's exporter");
      }
      appExporter.reset();
      try (Fixture unconfigured =
          new Fixture(null, ScriptedModel.of(MODEL, ScriptedModel.text("done")), null)) {
        assertEquals("done", unconfigured.workflow.run("plain"));
      }
      assertTrue(
          appExporter.getFinishedSpanItems().stream()
              .anyMatch(span -> "invoke_agent".equals(operation(span))),
          "a Worker that did not opt in keeps the application's native ADK provider");
      flush(tracing);
      assertEquals(exported, nativeExporter.getFinishedSpanItems().size());
    }

    private static void run(
        Scenario scenario,
        GoogleAdkTracing tracing,
        InMemorySpanExporter nativeExporter,
        InMemorySpanExporter appExporter)
        throws Exception {
      String mode = scenario.mode();
      CountDownLatch entered = new CountDownLatch(1);
      CountDownLatch release = new CountDownLatch(scenario.gated ? 1 : 0);
      AtomicInteger requests = new AtomicInteger();
      LookupImpl.attempts.set(0);
      LookupImpl.retry = scenario == Scenario.RETRY;
      ScriptedModel model =
          ScriptedModel.responding(
              MODEL,
              request -> provide(scenario, request, requests.incrementAndGet(), entered, release));
      try (Fixture fixture =
          new Fixture(scenario == Scenario.LEGACY ? null : tracing, model, scenario)) {
        TracedWorkflow workflow = fixture.workflow;
        if (scenario == Scenario.TOOLS) {
          SpanContext incoming =
              SpanContext.createFromRemoteParent(
                  TRACE_ID,
                  PARENT_SPAN_ID,
                  TraceFlags.getSampled(),
                  TraceState.builder().put("vendor", "test").build());
          try (Scope scope = Context.root().with(Span.wrap(incoming)).makeCurrent()) {
            WorkflowClient.start(workflow::run, mode);
          }
        } else {
          WorkflowClient.start(workflow::run, mode);
        }
        if (scenario == Scenario.UPDATE) assertEquals("done", workflow.ask());
        if (scenario == Scenario.SIGNAL) workflow.askSignal();
        assertTrue(entered.await(10, TimeUnit.SECONDS));
        if (scenario.gated) {
          disturb(scenario, fixture, tracing, model);
          flush(tracing);
          assertEquals(
              0,
              nativeExporter.getFinishedSpanItems().size(),
              "unfinished turns, nested calls and eviction must not export spans");
          if (scenario == Scenario.CANCEL) {
            WorkflowStub.fromTyped(workflow).cancel();
            assertThrows(
                WorkflowFailedException.class,
                () -> WorkflowStub.fromTyped(workflow).getResult(String.class));
          }
          release.countDown();
        }
        if (scenario == Scenario.FAILURE) {
          assertThrows(
              WorkflowFailedException.class,
              () -> WorkflowStub.fromTyped(workflow).getResult(String.class));
        } else if (scenario != Scenario.CANCEL) {
          assertEquals("done", WorkflowStub.fromTyped(workflow).getResult(String.class));
        }
        flush(tracing);
        List<SpanData> spans = nativeExporter.getFinishedSpanItems();
        List<SpanData> appSpans = appExporter.getFinishedSpanItems();
        if (scenario == Scenario.TOOLS) assertNativeAttributesAndParentage(spans, appSpans);
        if (scenario == Scenario.LEGACY) {
          assertEquals(0, spans.size());
        } else {
          assertNativeOperations(scenario, spans, appSpans, requests.get());
        }
        if (scenario.evicts) {
          assertTrue(
              fixture.executions.get() >= 3,
              "a parked second Workflow must force the first to replay");
        }
        assertFreshWorkerReplayExportsNothing(
            fixture, tracing, model, nativeExporter, spans.size(), requests);
        assertOtherInstrumentationStaysWithTheApplication(appExporter);
        flush(tracing);
        assertEquals(spans.size(), nativeExporter.getFinishedSpanItems().size());
      } finally {
        release.countDown();
      }
    }

    /** The provider: a gate the test releases, then one scripted response per scenario. */
    private static LlmResponse provide(
        Scenario scenario,
        LlmRequest request,
        int call,
        CountDownLatch entered,
        CountDownLatch release) {
      Span span =
          GlobalOpenTelemetry.getTracer("application-model")
              .spanBuilder("provider-call")
              .startSpan();
      try (Scope scope = span.makeCurrent()) {
        if (scenario != Scenario.NESTED || call == 3) {
          entered.countDown();
          if (!release.await(25, TimeUnit.SECONDS)) {
            throw new IllegalStateException("provider gate timed out");
          }
        }
        if (scenario == Scenario.FAILURE) {
          throw ApplicationFailure.newNonRetryableFailure(
              "invalid model request", "InvalidRequest");
        }
        if (scenario == Scenario.RETRY && call == 1) {
          throw ApplicationFailure.newFailure("temporary provider failure", "Transient");
        }
        if (scenario.tools) return response(request);
        if (scenario == Scenario.NESTED
            && request.tools().containsKey("nestedTurn")
            && TestSupport.functionResponse(request, "nestedTurn").isEmpty()) {
          return ScriptedModel.functionCall("nestedTurn", Map.of());
        }
        return ScriptedModel.text("done");
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException(interrupted);
      } catch (RuntimeException failure) {
        span.setStatus(StatusCode.ERROR, failure.getMessage());
        throw failure;
      } finally {
        span.end();
      }
    }

    /** Disturbs the Workflow while the model is blocked, so the turn must replay. */
    private static void disturb(
        Scenario scenario, Fixture fixture, GoogleAdkTracing tracing, ScriptedModel model)
        throws Exception {
      switch (scenario) {
        case EVICTION, NESTED -> fixture.evict();
        case RESTART -> fixture.restart();
        case UNFINISHED -> {
          WorkflowExecutionHistory partial = fixture.history();
          try (Fixture replay = new Fixture(tracing, model, null)) {
            replay.worker.replayWorkflowExecution(partial);
          }
        }
        default -> {}
      }
    }

    private static void assertNativeAttributesAndParentage(
        List<SpanData> spans, List<SpanData> appSpans) {
      assertTrue(
          spans.stream()
              .anyMatch(
                  span ->
                      "invoke_agent".equals(operation(span))
                          && "assistant".equals(attribute(span, "gen_ai.agent.name"))));
      assertTrue(
          spans.stream().anyMatch(span -> MODEL.equals(attribute(span, "gen_ai.request.model"))));
      assertTrue(
          spans.stream()
              .anyMatch(span -> "session".equals(attribute(span, "gcp.vertex.agent.session_id"))));
      assertTrue(
          spans.stream()
              .anyMatch(
                  span -> {
                    String response = attribute(span, "gcp.vertex.agent.llm_response");
                    return response != null && response.contains("done");
                  }));
      assertTrue(spans.stream().anyMatch(span -> span.getParentSpanId().equals(PARENT_SPAN_ID)));
      for (SpanData span : spans) {
        assertEquals(TRACE_ID, span.getTraceId());
        assertEquals("test", span.getSpanContext().getTraceState().get("vendor"));
        assertEquals(
            SERVICE, span.getResource().getAttribute(AttributeKey.stringKey("service.name")));
        if ("call_llm".equals(operation(span))) {
          assertTrue(
              spans.stream()
                  .anyMatch(
                      parent ->
                          parent.getSpanId().equals(span.getParentSpanId())
                              && "invoke_agent".equals(operation(parent))));
        }
      }
      for (SpanData span : appSpans) {
        assertEquals("test", span.getSpanContext().getTraceState().get("vendor"));
      }
    }

    private static void assertNativeOperations(
        Scenario scenario, List<SpanData> spans, List<SpanData> appSpans, int requests) {
      assertTrue(
          spans.stream().anyMatch(s -> "invoke_agent".equals(operation(s))), spans.toString());
      if (scenario != Scenario.CANCEL) {
        assertTrue(spans.stream().anyMatch(s -> "call_llm".equals(operation(s))), spans.toString());
      }
      if (scenario.tools) {
        for (String tool : List.of("lookup", "doubleIt")) {
          assertTrue(
              spans.stream()
                  .anyMatch(
                      s ->
                          "execute_tool".equals(operation(s))
                              && tool.equals(attribute(s, "gen_ai.tool.name"))),
              spans.toString());
        }
      }
      assertEquals(
          spans.size(),
          spans.stream().map(SpanData::getSpanId).distinct().count(),
          "each native operation exports once");
      assertTrue(
          appSpans.stream()
              .noneMatch(s -> "gcp.vertex.agent".equals(s.getInstrumentationScopeInfo().getName())),
          "native spans of an opted-in Worker must not reach the application provider");
      if (scenario != Scenario.OVERLAP) {
        assertEquals(1, spans.stream().map(SpanData::getTraceId).distinct().count());
      }
      for (SpanData span : appSpans) {
        if (span.getName().equals("provider-call") || span.getName().equals("lookup-activity")) {
          assertTrue(
              spans.stream().anyMatch(parent -> parent.getSpanId().equals(span.getParentSpanId())),
              "Activity/RxJava context lost: " + span);
        }
      }
      if (scenario == Scenario.FAILURE) {
        assertTrue(spans.stream().anyMatch(s -> s.getStatus().getStatusCode() == StatusCode.ERROR));
      }
      if (scenario == Scenario.RETRY) {
        assertEquals(4, requests);
        assertEquals(2, LookupImpl.attempts.get());
        assertEquals(
            2,
            appSpans.stream()
                .filter(s -> s.getStatus().getStatusCode() == StatusCode.ERROR)
                .count());
      }
    }

    private static void assertFreshWorkerReplayExportsNothing(
        Fixture fixture,
        GoogleAdkTracing tracing,
        ScriptedModel model,
        InMemorySpanExporter nativeExporter,
        int exported,
        AtomicInteger requests)
        throws Exception {
      int calls = requests.get();
      WorkflowExecutionHistory history = fixture.history();
      try (Fixture replay = new Fixture(tracing, model, null)) {
        replay.worker.replayWorkflowExecution(history);
      }
      flush(tracing);
      assertEquals(
          exported,
          nativeExporter.getFinishedSpanItems().size(),
          "a fresh Worker replays completed and legacy history without exporting");
      assertEquals(calls, requests.get(), "replay makes no provider calls");
    }

    private static void assertOtherInstrumentationStaysWithTheApplication(
        InMemorySpanExporter appExporter) {
      GlobalOpenTelemetry.getTracer("application").spanBuilder("unrelated").startSpan().end();
      GlobalOpenTelemetry.getTracer("gcp.vertex.agent")
          .spanBuilder("outside-workflow")
          .startSpan()
          .end();
      for (String name : List.of("unrelated", "outside-workflow")) {
        assertTrue(
            appExporter.getFinishedSpanItems().stream().anyMatch(s -> s.getName().equals(name)),
            name);
      }
    }

    private static final class Fixture implements AutoCloseable {
      final TestWorkflowEnvironment environment;
      final Worker worker;
      final TracedWorkflow workflow;
      final AtomicInteger executions = new AtomicInteger();
      final List<WorkerFactory> extras = new ArrayList<>();

      /** A null scenario is a plain Worker for replay or for the opted-out comparison. */
      Fixture(
          @Nullable GoogleAdkTracing tracing, ScriptedModel model, @Nullable Scenario scenario) {
        GoogleAdkPlugin.Builder plugin = GoogleAdkPlugin.newBuilder().model(MODEL, model);
        if (tracing != null) plugin.tracing(tracing);
        WorkerFactoryOptions options =
            WorkerFactoryOptions.newBuilder()
                .setWorkflowCacheSize(scenario != null && scenario.evicts ? 1 : 600)
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
        if (target != null) {
          environmentOptions
              .setUseExternalService(true)
              .setTarget(target)
              .setWorkflowServiceStubsOptions(
                  WorkflowServiceStubsOptions.newBuilder().setTarget(target).build());
        }
        environment = TestWorkflowEnvironment.newInstance(environmentOptions.build());
        worker = environment.newWorker("native-tracing");
        worker.registerWorkflowImplementationTypes(TracedWorkflowImpl.class);
        worker.registerActivitiesImplementations(new LookupImpl());
        if (scenario == Scenario.RESTART) {
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

      WorkflowExecutionHistory history() {
        return environment
            .getWorkflowClient()
            .fetchHistory(WorkflowStub.fromTyped(workflow).getExecution().getWorkflowId());
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

    private static String attribute(SpanData span, String key) {
      return span.getAttributes().get(AttributeKey.stringKey(key));
    }

    /** ADK sets this on agent and tool spans; the plugin adds it to call_llm spans. */
    private static String operation(SpanData span) {
      return attribute(span, "gen_ai.operation.name");
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
      switch (mode) {
        case "park" -> {
          Workflow.await(() -> false);
          return "parked";
        }
        case "update", "signal" -> {
          Workflow.await(() -> finished);
          return "done";
        }
        case "restart" -> {
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
              TemporalAdk.run(
                  runner, "user", "session", Content.fromParts(Part.fromText("hello"))));
        }
        case "nested" -> {
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
              TemporalAdk.run(
                  runner, "user", "session", Content.fromParts(Part.fromText("hello"))));
        }
        case "overlap" -> {
          Promise<String> first = Async.function(() -> turn("first", false));
          Promise<String> second = Async.function(() -> turn("second", false));
          assertEquals("done", first.get());
          return second.get();
        }
        case "callback" -> {
          return Workflow.newTimer(Duration.ofMillis(10))
              .thenApply(ignored -> turn("callback", false))
              .get();
        }
        case "parallel" -> {
          ParallelAgent parallel =
              ParallelAgent.builder()
                  .name("parallel")
                  .subAgents(agent("first", false), agent("second", false))
                  .build();
          Runner runner =
              TemporalAdk.runnerBuilder().appName("native-tracing").agent(parallel).build();
          TemporalAdk.run(runner, "user", "session", Content.fromParts(Part.fromText("hello")));
          return "done";
        }
        default -> {
          return turn("assistant", mode.equals("tools") || mode.equals("retry"));
        }
      }
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
