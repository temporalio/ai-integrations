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
import io.opentelemetry.sdk.resources.Resource;
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

/**
 * Native ADK tracing for agents that run inside Workflows: Workflow time, deterministic span
 * identifiers, and export that replay never repeats. See the README for the bootstrap order.
 */
public final class GoogleAdkTracing implements AutoCloseable {
  private static final String ADK_SCOPE = "gcp.vertex.agent";
  private static final WorkflowLocal<Long> SEQUENCE = WorkflowLocal.withCachedInitial(() -> 0L);
  private static final ContextKey<TurnBuffer> TURN = ContextKey.named("temporal-adk-tracing-turn");
  static final ContextKey<GoogleAdkTracing> SELECTED =
      ContextKey.named("temporal-adk-tracing-instance");
  private final SdkTracerProvider provider;
  final NativeTracingPropagation propagation = new NativeTracingPropagation(this);

  private GoogleAdkTracing(SpanExporter exporter, Resource resource) {
    provider =
        SdkTracerProvider.builder()
            .addResource(resource)
            .setClock(new WorkflowClock())
            .setIdGenerator(new WorkflowIds())
            .addSpanProcessor(new ReplayProcessor(BatchSpanProcessor.builder(exporter).build()))
            .build();
  }

  /** Creates native ADK tracing with the SDK's default resource. */
  public static GoogleAdkTracing create(SpanExporter exporter) {
    return create(exporter, Resource.empty());
  }

  /**
   * Creates native ADK tracing that exports asynchronously through a dedicated batch processor. The
   * instance owns the exporter and closes it with {@link #close()}. {@code resource} is merged over
   * the SDK defaults; pass the application's resource so native spans carry its {@code
   * service.name}.
   */
  public static GoogleAdkTracing create(SpanExporter exporter, Resource resource) {
    return new GoogleAdkTracing(
        Objects.requireNonNull(exporter, "exporter"), Objects.requireNonNull(resource, "resource"));
  }

  /**
   * Wraps the application's telemetry for its one-time global registration, before any ADK code
   * runs. Native ADK spans created on the Workflow threads of a Worker whose plugin configured
   * tracing go to that tracing instance; everything else reaches the application's telemetry
   * unchanged. The wrapper neither owns nor closes the application's telemetry.
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

  /**
   * Starts a turn on the current Workflow thread, or joins the enclosing one. Spans completed in a
   * turn export when its outermost owner closes outside replay.
   */
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

    /** Drops the buffered spans; a later replay rebuilds and exports them. */
    void discard() {
      if (buffer == null) return;
      buffer.discarded = true;
      buffer.completed.clear();
    }

    /** Marks the spans still open with the failure that ends the turn. */
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

  /** Flushes native spans pending export without closing the provider; join the result to wait. */
  public CompletableResultCode forceFlush() {
    return provider.forceFlush();
  }

  /** Drains pending exports and closes the dedicated exporter; application telemetry stays open. */
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
    // Seeded by the Run ID: the new Run after a Reset must not reuse identifiers that the original
    // Run already exported.
    String seed =
        "io.temporal.googleadk.native-tracing.v1|"
            + Workflow.getInfo().getRunId()
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

  /** Buffers Workflow spans per turn and hands spans ended outside Workflows straight through. */
  private static final class ReplayProcessor implements SpanProcessor {
    private final SpanProcessor delegate;

    ReplayProcessor(SpanProcessor delegate) {
      this.delegate = delegate;
    }

    @Override
    public void onStart(Context parent, ReadWriteSpan span) {
      if (workflow()) {
        // ADK sets the operation name of a call_llm span only once a response arrives, so failed
        // and canceled calls would otherwise export without one.
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
        // A Workflow span ended outside a turn is dropped: nothing would suppress its replay.
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
