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
import io.temporal.workflow.Functions;
import io.temporal.workflow.Promise;
import io.temporal.workflow.TimerOptions;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Supplier;

/**
 * Carries W3C trace context through Temporal headers, selects the Worker's tracing instance on
 * Workflow and Activity threads, and keeps the enclosing context across Promise callbacks.
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
        try (Scope scope = select()) {
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
        super.init(new ContextPreservingOutbound(outbound));
      }

      @Override
      public WorkflowOutput execute(WorkflowInput input) {
        try (Scope scope = select()) {
          return super.execute(input);
        } finally {
          incoming.remove();
        }
      }

      @Override
      public void handleSignal(SignalInput input) {
        try (Scope scope = select()) {
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
      public void validateUpdate(UpdateInput input) {
        // Validators are skipped on replay, so they must not consume the sequence either.
        try (Scope scope = deselect()) {
          super.validateUpdate(input);
        }
      }

      @Override
      public UpdateOutput executeUpdate(UpdateInput input) {
        try (Scope scope = select()) {
          return super.executeUpdate(input);
        } finally {
          incoming.remove();
        }
      }

      /** Timer and Activity completions run on SDK callback threads, which need the context too. */
      @Override
      public Object newCallbackThread(Runnable runnable, String name) {
        return super.newCallbackThread(
            () -> {
              try (Scope scope = select()) {
                runnable.run();
              } finally {
                incoming.remove();
              }
            },
            name);
      }
    };
  }

  /** Makes the incoming trace context current with this Worker's tracing instance selected. */
  private Scope select() {
    return incoming.get().with(GoogleAdkTracing.SELECTED, tracing).makeCurrent();
  }

  private static Scope deselect() {
    return Context.current().with(GoogleAdkTracing.SELECTED, null).makeCurrent();
  }

  /**
   * Child threads inherit the current context, and the Promises the SDK completes (timers,
   * Activities, child Workflows, Nexus operations, external Signals, cancellations) run their
   * callbacks in the context that registered them, so a turn started from a completion callback
   * keeps its enclosing span and turn.
   */
  private final class ContextPreservingOutbound extends WorkflowOutboundCallsInterceptorBase {
    ContextPreservingOutbound(WorkflowOutboundCallsInterceptor next) {
      super(next);
    }

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

    @Override
    public Promise<Void> newTimer(Duration duration) {
      return ContextPromise.of(super.newTimer(duration));
    }

    @Override
    public Promise<Void> newTimer(Duration duration, TimerOptions options) {
      return ContextPromise.of(super.newTimer(duration, options));
    }

    @Override
    public <R> ActivityOutput<R> executeActivity(ActivityInput<R> input) {
      ActivityOutput<R> output = super.executeActivity(input);
      return new ActivityOutput<>(output.getActivityId(), ContextPromise.of(output.getResult()));
    }

    @Override
    public <R> LocalActivityOutput<R> executeLocalActivity(LocalActivityInput<R> input) {
      return new LocalActivityOutput<>(
          ContextPromise.of(super.executeLocalActivity(input).getResult()));
    }

    @Override
    public <R> ChildWorkflowOutput<R> executeChildWorkflow(ChildWorkflowInput<R> input) {
      ChildWorkflowOutput<R> output = super.executeChildWorkflow(input);
      return new ChildWorkflowOutput<>(
          ContextPromise.of(output.getResult()), ContextPromise.of(output.getWorkflowExecution()));
    }

    @Override
    public <R> ExecuteNexusOperationOutput<R> executeNexusOperation(
        ExecuteNexusOperationInput<R> input) {
      ExecuteNexusOperationOutput<R> output = super.executeNexusOperation(input);
      return new ExecuteNexusOperationOutput<>(
          ContextPromise.of(output.getResult()), ContextPromise.of(output.getOperationExecution()));
    }

    @Override
    public SignalExternalOutput signalExternalWorkflow(SignalExternalInput input) {
      return new SignalExternalOutput(
          ContextPromise.of(super.signalExternalWorkflow(input).getResult()));
    }

    @Override
    public CancelWorkflowOutput cancelWorkflow(CancelWorkflowInput input) {
      return new CancelWorkflowOutput(ContextPromise.of(super.cancelWorkflow(input).getResult()));
    }
  }

  /** A Promise whose callbacks run in the trace context that was current when they were added. */
  private static final class ContextPromise<V> implements Promise<V> {
    private final Promise<V> delegate;

    private ContextPromise(Promise<V> delegate) {
      this.delegate = delegate;
    }

    static <V> Promise<V> of(Promise<V> promise) {
      return promise instanceof ContextPromise ? promise : new ContextPromise<>(promise);
    }

    private static <T> T call(Context context, Supplier<T> supplier) {
      try (Scope scope = context.makeCurrent()) {
        return supplier.get();
      }
    }

    @Override
    public boolean isCompleted() {
      return delegate.isCompleted();
    }

    @Override
    public V get() {
      return delegate.get();
    }

    @Override
    public V cancellableGet() {
      return delegate.cancellableGet();
    }

    @Override
    public V get(long timeout, TimeUnit unit) throws TimeoutException {
      return delegate.get(timeout, unit);
    }

    @Override
    public V cancellableGet(long timeout, TimeUnit unit) throws TimeoutException {
      return delegate.cancellableGet(timeout, unit);
    }

    @Override
    public RuntimeException getFailure() {
      return delegate.getFailure();
    }

    @Override
    public <U> Promise<U> thenApply(Functions.Func1<? super V, ? extends U> function) {
      Context context = Context.current();
      return of(delegate.<U>thenApply(value -> call(context, () -> function.apply(value))));
    }

    @Override
    public <U> Promise<U> handle(
        Functions.Func2<? super V, RuntimeException, ? extends U> function) {
      Context context = Context.current();
      return of(
          delegate.<U>handle(
              (value, failure) -> call(context, () -> function.apply(value, failure))));
    }

    @Override
    public <U> Promise<U> thenCompose(Functions.Func1<? super V, ? extends Promise<U>> function) {
      Context context = Context.current();
      return of(delegate.<U>thenCompose(value -> call(context, () -> function.apply(value))));
    }

    @Override
    public Promise<V> exceptionally(Functions.Func1<Throwable, ? extends V> function) {
      Context context = Context.current();
      return of(delegate.exceptionally(failure -> call(context, () -> function.apply(failure))));
    }
  }
}
