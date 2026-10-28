package io.temporal.googleadk;

import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.logs.LoggerProvider;
import io.opentelemetry.api.metrics.MeterProvider;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.api.trace.TracerBuilder;
import io.opentelemetry.api.trace.TracerProvider;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.ContextKey;
import io.opentelemetry.context.Scope;
import io.opentelemetry.context.propagation.ContextPropagators;
import io.opentelemetry.sdk.common.Clock;
import io.opentelemetry.sdk.common.CompletableResultCode;
import io.opentelemetry.sdk.trace.IdGenerator;
import io.opentelemetry.sdk.trace.ReadWriteSpan;
import io.opentelemetry.sdk.trace.ReadableSpan;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.SpanProcessor;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import io.opentelemetry.sdk.trace.export.SpanExporter;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.WorkflowLocal;
import io.temporal.workflow.unsafe.WorkflowUnsafe;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

/** Native ADK tracing with deterministic Workflow identifiers and replay-suppressed export. */
public final class GoogleAdkTracing implements AutoCloseable {
  private static final String ADK_SCOPE = "gcp.vertex.agent";
  private static final WorkflowLocal<Long> SEQUENCE = WorkflowLocal.withInitial(() -> 0L);
  private static final ContextKey<TurnBuffer> TURN = ContextKey.named("temporal-adk-tracing-turn");
  static final ContextKey<GoogleAdkTracing> SELECTED =
      ContextKey.named("temporal-adk-tracing-instance");
  private final SdkTracerProvider provider;
  final NativeTracingPropagation propagation = new NativeTracingPropagation(this);

  private GoogleAdkTracing(SpanExporter exporter) {
    provider =
        SdkTracerProvider.builder()
            .setClock(new WorkflowClock())
            .setIdGenerator(new WorkflowIds())
            .addSpanProcessor(new ReplayProcessor(BatchSpanProcessor.builder(exporter).build()))
            .build();
  }

  /**
   * Creates native ADK tracing. The returned instance owns the exporter and closes it at shutdown.
   * Export runs asynchronously; the exporter must be dedicated to this instance.
   */
  public static GoogleAdkTracing create(SpanExporter exporter) {
    return new GoogleAdkTracing(Objects.requireNonNull(exporter, "exporter"));
  }

  /**
   * Wraps the application's telemetry before its one-time global registration and before ADK's
   * Tracing class initializes. Native ADK Workflow instrumentation selects the tracing instance
   * registered by its Worker's plugin. Other tracing, meters, logs, and propagation delegate to the
   * application unchanged. The wrapper does not own or close the application's telemetry.
   */
  public static OpenTelemetry wrap(OpenTelemetry application) {
    Objects.requireNonNull(application, "application");
    TracerProvider tracers =
        new TracerProvider() {
          @Override
          public Tracer get(String name) {
            return ADK_SCOPE.equals(name)
                ? routed(name, null, null, application.getTracerProvider().get(name))
                : application.getTracerProvider().get(name);
          }

          @Override
          public Tracer get(String name, String version) {
            return ADK_SCOPE.equals(name)
                ? routed(name, version, null, application.getTracerProvider().get(name, version))
                : application.getTracerProvider().get(name, version);
          }

          @Override
          public TracerBuilder tracerBuilder(String name) {
            if (!ADK_SCOPE.equals(name)) return application.getTracerProvider().tracerBuilder(name);
            return new RoutedTracerBuilder(
                name, application.getTracerProvider().tracerBuilder(name));
          }
        };
    return new OpenTelemetry() {
      @Override
      public TracerProvider getTracerProvider() {
        return tracers;
      }

      @Override
      public ContextPropagators getPropagators() {
        return application.getPropagators();
      }

      @Override
      public MeterProvider getMeterProvider() {
        return application.getMeterProvider();
      }

      @Override
      public LoggerProvider getLogsBridge() {
        return application.getLogsBridge();
      }
    };
  }

  private static Tracer routed(
      String scope, String version, String schema, Tracer applicationTracer) {
    return name -> {
      GoogleAdkTracing selected = Context.current().get(SELECTED);
      if (!workflow() || selected == null) return applicationTracer.spanBuilder(name);
      TracerBuilder builder = selected.provider.tracerBuilder(scope);
      if (version != null) builder.setInstrumentationVersion(version);
      if (schema != null) builder.setSchemaUrl(schema);
      return builder.build().spanBuilder(name);
    };
  }

  private static final class RoutedTracerBuilder implements TracerBuilder {
    private final String name;
    private final TracerBuilder application;
    private String version;
    private String schema;

    RoutedTracerBuilder(String name, TracerBuilder application) {
      this.name = name;
      this.application = application;
    }

    @Override
    public TracerBuilder setSchemaUrl(String schemaUrl) {
      schema = schemaUrl;
      application.setSchemaUrl(schemaUrl);
      return this;
    }

    @Override
    public TracerBuilder setInstrumentationVersion(String version) {
      this.version = version;
      application.setInstrumentationVersion(version);
      return this;
    }

    @Override
    public Tracer build() {
      return routed(name, version, schema, application.build());
    }
  }

  static Turn beginTurn() {
    if (!workflow() || Context.current().get(SELECTED) == null) {
      return new Turn(null, false, Scope.noop());
    }
    TurnBuffer buffer = Context.current().get(TURN);
    boolean owner = buffer == null;
    if (owner) buffer = new TurnBuffer();
    return new Turn(buffer, owner, Context.current().with(TURN, buffer).makeCurrent());
  }

  static final class Turn implements AutoCloseable {
    private final TurnBuffer buffer;
    private final boolean owner;
    private final Scope scope;

    Turn(TurnBuffer buffer, boolean owner, Scope scope) {
      this.buffer = buffer;
      this.owner = owner;
      this.scope = scope;
    }

    void discard() {
      if (buffer == null) return;
      buffer.discarded = true;
      buffer.completed.clear();
    }

    void fail(Throwable failure) {
      if (buffer == null) return;
      buffer.open.forEach(
          span -> {
            span.setStatus(
                StatusCode.ERROR,
                failure.getMessage() == null ? failure.getClass().getName() : failure.getMessage());
            span.recordException(failure);
          });
    }

    @Override
    public void close() {
      try {
        if (owner && !buffer.discarded && !WorkflowUnsafe.isReplaying()) {
          buffer.completed.forEach(completed -> completed.processor.onEnd(completed.span));
        }
      } finally {
        if (owner) {
          buffer.discarded = true;
          buffer.completed.clear();
          buffer.open.clear();
        }
        scope.close();
      }
    }
  }

  private static final class TurnBuffer {
    final List<CompletedSpan> completed = new ArrayList<>();
    final List<ReadWriteSpan> open = new ArrayList<>();
    boolean discarded;
  }

  private record CompletedSpan(SpanProcessor processor, ReadableSpan span) {}

  /** Flushes previously completed native ADK spans without closing the provider. */
  public CompletableResultCode forceFlush() {
    return provider.forceFlush();
  }

  /** Closes this instance and its dedicated exporter; application telemetry remains open. */
  @Override
  public void close() {
    provider.close();
  }

  private static boolean workflow() {
    return WorkflowUnsafe.isWorkflowThread();
  }

  private static String id(String kind, int characters) {
    long sequence = SEQUENCE.get();
    SEQUENCE.set(sequence + 1);
    String seed =
        "io.temporal.googleadk.native-tracing.v1|"
            + Workflow.getInfo().getOriginalExecutionRunId()
            + "|"
            + kind
            + "|"
            + sequence;
    try {
      return HexFormat.of()
          .formatHex(
              MessageDigest.getInstance("SHA-256").digest(seed.getBytes(StandardCharsets.UTF_8)))
          .substring(0, characters);
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException("SHA-256 is required by Java", impossible);
    }
  }

  private static final class WorkflowIds implements IdGenerator {
    private final IdGenerator random = IdGenerator.random();

    @Override
    public String generateSpanId() {
      return workflow() ? id("span", 16) : random.generateSpanId();
    }

    @Override
    public String generateTraceId() {
      return workflow() ? id("trace", 32) : random.generateTraceId();
    }
  }

  private static final class WorkflowClock implements Clock {
    @Override
    public long now() {
      return workflow()
          ? TimeUnit.MILLISECONDS.toNanos(Workflow.currentTimeMillis())
          : Clock.getDefault().now();
    }

    @Override
    public long nanoTime() {
      return workflow()
          ? TimeUnit.MILLISECONDS.toNanos(Workflow.currentTimeMillis())
          : Clock.getDefault().nanoTime();
    }
  }

  private static final class ReplayProcessor implements SpanProcessor {
    private final SpanProcessor delegate;

    ReplayProcessor(SpanProcessor delegate) {
      this.delegate = delegate;
    }

    @Override
    public void onStart(Context parent, ReadWriteSpan span) {
      if (workflow()) {
        if (span.getName().equals("call_llm"))
          span.setAttribute(AttributeKey.stringKey("gen_ai.operation.name"), "call_llm");
        TurnBuffer buffer = Context.current().get(TURN);
        if (buffer != null && !buffer.discarded) buffer.open.add(span);
      }
      if (!workflow() || !WorkflowUnsafe.isReplaying()) {
        delegate.onStart(parent, span);
      }
    }

    @Override
    public boolean isStartRequired() {
      return true;
    }

    @Override
    public void onEnd(ReadableSpan span) {
      if (!workflow()) {
        delegate.onEnd(span);
      } else {
        TurnBuffer buffer = Context.current().get(TURN);
        if (buffer != null && !buffer.discarded) {
          buffer.open.remove(span);
          buffer.completed.add(new CompletedSpan(delegate, span));
        }
      }
    }

    @Override
    public boolean isEndRequired() {
      return delegate.isEndRequired();
    }

    @Override
    public CompletableResultCode forceFlush() {
      return delegate.forceFlush();
    }

    @Override
    public CompletableResultCode shutdown() {
      return delegate.shutdown();
    }
  }
}
