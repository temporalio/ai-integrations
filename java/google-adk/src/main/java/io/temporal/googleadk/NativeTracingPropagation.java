package io.temporal.googleadk;

import com.google.protobuf.ByteString;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;
import io.opentelemetry.context.propagation.TextMapGetter;
import io.temporal.api.common.v1.Payload;
import io.temporal.common.context.ContextPropagator;
import io.temporal.common.interceptors.ActivityInboundCallsInterceptor;
import io.temporal.common.interceptors.ActivityInboundCallsInterceptorBase;
import io.temporal.common.interceptors.WorkerInterceptorBase;
import io.temporal.common.interceptors.WorkflowInboundCallsInterceptor;
import io.temporal.common.interceptors.WorkflowInboundCallsInterceptorBase;
import io.temporal.common.interceptors.WorkflowOutboundCallsInterceptor;
import io.temporal.common.interceptors.WorkflowOutboundCallsInterceptorBase;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Carries W3C trace context through Temporal headers and selects the Worker's tracing instance on
 * Workflow and Activity threads.
 */
final class NativeTracingPropagation extends WorkerInterceptorBase implements ContextPropagator {
  private static final String HEADER = "io.temporal.googleadk.traceparent";
  private final ThreadLocal<Context> incoming = ThreadLocal.withInitial(Context::root);
  private final GoogleAdkTracing tracing;

  NativeTracingPropagation(GoogleAdkTracing tracing) {
    this.tracing = tracing;
  }

  @Override
  public String getName() {
    return "io.temporal.googleadk.tracing";
  }

  @Override
  public Map<String, Payload> serializeContext(Object value) {
    Map<String, Payload> headers = new LinkedHashMap<>();
    W3CTraceContextPropagator.getInstance()
        .inject(
            (Context) value,
            headers,
            (carrier, key, text) ->
                carrier.put(
                    HEADER + "." + key,
                    Payload.newBuilder().setData(ByteString.copyFromUtf8(text)).build()));
    return headers;
  }

  @Override
  public Object deserializeContext(Map<String, Payload> headers) {
    return W3CTraceContextPropagator.getInstance()
        .extract(
            Context.root(),
            headers,
            new TextMapGetter<Map<String, Payload>>() {
              @Override
              public Iterable<String> keys(Map<String, Payload> carrier) {
                return W3CTraceContextPropagator.getInstance().fields();
              }

              @Override
              public String get(Map<String, Payload> carrier, String key) {
                Payload payload = carrier.get(HEADER + "." + key);
                return payload == null ? null : payload.getData().toStringUtf8();
              }
            });
  }

  @Override
  public Object getCurrentContext() {
    Context current = Context.current();
    return Span.fromContext(current).getSpanContext().isValid() ? current : incoming.get();
  }

  @Override
  public void setCurrentContext(Object context) {
    incoming.set((Context) context);
  }

  @Override
  public ActivityInboundCallsInterceptor interceptActivity(ActivityInboundCallsInterceptor next) {
    return new ActivityInboundCallsInterceptorBase(next) {
      @Override
      public ActivityOutput execute(ActivityInput input) {
        try (Scope scope = incoming.get().with(GoogleAdkTracing.SELECTED, tracing).makeCurrent()) {
          return super.execute(input);
        } finally {
          incoming.remove();
        }
      }
    };
  }

  @Override
  public WorkflowInboundCallsInterceptor interceptWorkflow(WorkflowInboundCallsInterceptor next) {
    return new WorkflowInboundCallsInterceptorBase(next) {
      @Override
      public void init(WorkflowOutboundCallsInterceptor outbound) {
        super.init(
            new WorkflowOutboundCallsInterceptorBase(outbound) {
              @Override
              public Object newChildThread(Runnable runnable, boolean detached, String name) {
                Context context = Context.current();
                return super.newChildThread(
                    () -> {
                      try (Scope scope = context.makeCurrent()) {
                        runnable.run();
                      } finally {
                        incoming.remove();
                      }
                    },
                    detached,
                    name);
              }
            });
      }

      @Override
      public WorkflowOutput execute(WorkflowInput input) {
        try (Scope scope = incoming.get().with(GoogleAdkTracing.SELECTED, tracing).makeCurrent()) {
          return super.execute(input);
        } finally {
          incoming.remove();
        }
      }

      @Override
      public void handleSignal(SignalInput input) {
        try (Scope scope = incoming.get().with(GoogleAdkTracing.SELECTED, tracing).makeCurrent()) {
          super.handleSignal(input);
        } finally {
          incoming.remove();
        }
      }

      @Override
      public QueryOutput handleQuery(QueryInput input) {
        // Queries are not replayed, so a native span in a query handler must not consume the
        // deterministic identifier sequence; it uses the application's provider instead.
        try {
          return super.handleQuery(input);
        } finally {
          incoming.remove();
        }
      }

      @Override
      public UpdateOutput executeUpdate(UpdateInput input) {
        try (Scope scope = incoming.get().with(GoogleAdkTracing.SELECTED, tracing).makeCurrent()) {
          return super.executeUpdate(input);
        } finally {
          incoming.remove();
        }
      }
    };
  }
}
