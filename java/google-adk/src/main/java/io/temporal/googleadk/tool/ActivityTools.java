package io.temporal.googleadk.tool;

import com.google.adk.tools.FunctionTool;
import com.google.adk.tools.ToolContext;
import io.reactivex.rxjava3.core.Single;
import io.temporal.activity.ActivityInterface;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Turns methods of a Temporal activity stub into ADK function tools, so a tool call becomes a
 * durable activity execution.
 *
 * <pre>{@code
 * WeatherActivities weather =
 *     Workflow.newActivityStub(
 *         WeatherActivities.class,
 *         ActivityOptions.newBuilder().setStartToCloseTimeout(Duration.ofSeconds(30)).build());
 * LlmAgent agent =
 *     LlmAgent.builder()
 *         .name("assistant")
 *         .model(TemporalModel.create("gemini-2.5-flash"))
 *         .tools(ActivityTools.create(weather, "getWeather"))
 *         .build();
 * }</pre>
 *
 * <p>The tool's name, description, and parameter schema come from the activity interface method and
 * its {@code @Schema} annotations, exactly as for any other ADK function tool; as with any ADK
 * function tool, every parameter needs a {@code @Schema(name = ...)} annotation unless the
 * interface is compiled with {@code -parameters}. The method must be declared on an
 * {@code @ActivityInterface}; a {@code toolContext} parameter is not supported, because the
 * activity runs outside the workflow.
 *
 * <p>When the activity fails after its retries are exhausted, the failure is not reported to the
 * model: the run fails with the {@code ActivityFailure}, so the workflow decides what to do.
 */
public final class ActivityTools {

  private ActivityTools() {}

  /** Creates a tool that runs the named method of the given activity or local activity stub. */
  public static FunctionTool create(Object activityStub, String methodName) {
    return create(activityStub, methodName, false);
  }

  /**
   * Creates a tool that runs the named method of the given stub, optionally asking the user to
   * confirm each call first (see {@link Hitl}).
   */
  public static FunctionTool create(
      Object activityStub, String methodName, boolean requireConfirmation) {
    return new ActivityFunctionTool(activityStub, methodName, requireConfirmation, false);
  }

  /**
   * Creates a long-running tool that runs the named method of the given stub. As for any ADK
   * long-running tool, the call stays open after the activity returns its interim result until a
   * later turn delivers the final one (see {@link Hitl#longRunningCalls}); an activity that returns
   * nothing ends the model's turn at once.
   */
  public static FunctionTool createLongRunning(Object activityStub, String methodName) {
    return new ActivityFunctionTool(activityStub, methodName, false, true);
  }

  /**
   * Returns the method of the stub's {@code @ActivityInterface} with the given name. The stub's own
   * class is a proxy whose methods carry no parameter annotations, so the interface method is the
   * one ADK must read the schema from.
   */
  static Method activityMethod(Object activityStub, String methodName) {
    Objects.requireNonNull(activityStub, "activityStub");
    Objects.requireNonNull(methodName, "methodName");
    if (!Proxy.isProxyClass(activityStub.getClass())) {
      throw new IllegalArgumentException(
          "ActivityTools requires an activity stub created with Workflow.newActivityStub or"
              + " Workflow.newLocalActivityStub, got "
              + activityStub.getClass().getName());
    }
    Deque<Class<?>> pending = new ArrayDeque<>();
    Set<Class<?>> seen = new HashSet<>();
    for (Class<?> implemented : activityStub.getClass().getInterfaces()) {
      pending.add(implemented);
    }
    boolean activityInterfaceSeen = false;
    while (!pending.isEmpty()) {
      Class<?> type = pending.poll();
      if (!seen.add(type)) {
        continue;
      }
      if (type.isAnnotationPresent(ActivityInterface.class)) {
        activityInterfaceSeen = true;
        for (Method method : type.getMethods()) {
          if (method.getName().equals(methodName)) {
            for (Class<?> parameter : method.getParameterTypes()) {
              if (ToolContext.class.isAssignableFrom(parameter)) {
                throw new IllegalArgumentException(
                    "Activity method " + methodName + " cannot take a ToolContext");
              }
            }
            return method;
          }
        }
      }
      for (Class<?> parent : type.getInterfaces()) {
        pending.add(parent);
      }
    }
    if (!activityInterfaceSeen) {
      throw new IllegalArgumentException(
          "ActivityTools requires an activity stub whose interface is annotated with"
              + " @ActivityInterface, got "
              + activityStub.getClass().getName());
    }
    throw new IllegalArgumentException(
        "No method named '"
            + methodName
            + "' found on an @ActivityInterface implemented by the stub");
  }

  /**
   * An ADK function tool over an activity stub. ADK reports any exception a tool throws to the
   * model as a generic error; this tool instead lets Temporal failures (and errors) raised by the
   * activity call end the run, so retries and failure handling stay with the workflow.
   */
  static final class ActivityFunctionTool extends FunctionTool {
    private static final ThreadLocal<Throwable[]> RAISED = new ThreadLocal<>();

    ActivityFunctionTool(
        Object activityStub, String methodName, boolean requireConfirmation, boolean longRunning) {
      this(
          activityStub, activityMethod(activityStub, methodName), requireConfirmation, longRunning);
    }

    private ActivityFunctionTool(
        Object activityStub, Method method, boolean requireConfirmation, boolean longRunning) {
      super(recording(activityStub), method, longRunning, requireConfirmation);
    }

    /**
     * Wraps the stub so a failed call records its throwable and returns normally. ADK's function
     * tool catches every exception a tool throws and reports a generic error to the model; a
     * recorded throwable is raised again by {@link #runAsync} instead.
     */
    private static Object recording(Object activityStub) {
      return Proxy.newProxyInstance(
          activityStub.getClass().getClassLoader(),
          activityStub.getClass().getInterfaces(),
          (proxy, method, arguments) -> {
            if (method.getDeclaringClass() == Object.class) {
              // Identity semantics: the stub itself rejects Object methods.
              switch (method.getName()) {
                case "hashCode":
                  return System.identityHashCode(proxy);
                case "equals":
                  return proxy == arguments[0];
                default:
                  return "ActivityTool(" + activityStub + ")";
              }
            }
            try {
              return method.invoke(activityStub, arguments);
            } catch (InvocationTargetException e) {
              Throwable[] slot = RAISED.get();
              if (slot == null) {
                throw e.getCause();
              }
              slot[0] = e.getCause();
              return defaultValue(method.getReturnType());
            }
          });
    }

    private static Object defaultValue(Class<?> type) {
      if (!type.isPrimitive() || type == void.class) {
        return null;
      }
      if (type == boolean.class) {
        return false;
      }
      if (type == char.class) {
        return '\0';
      }
      if (type == float.class) {
        return 0f;
      }
      if (type == double.class) {
        return 0d;
      }
      if (type == long.class) {
        return 0L;
      }
      return 0;
    }

    @Override
    public Single<Map<String, Object>> runAsync(Map<String, Object> args, ToolContext toolContext) {
      Throwable[] previous = RAISED.get();
      Throwable[] slot = new Throwable[1];
      RAISED.set(slot);
      try {
        Single<Map<String, Object>> result = super.runAsync(args, toolContext);
        if (slot[0] == null) {
          return result;
        }
        if (isWorkflowThreadTeardown(slot[0])) {
          // The SDK is destroying this workflow thread (eviction, cancellation, a failed
          // sibling); a never-completing result lets it unwind without an error in the flow.
          return Single.never();
        }
        if (slot[0] instanceof Error) {
          throw (Error) slot[0];
        }
        return Single.error(slot[0]);
      } finally {
        if (previous == null) {
          RAISED.remove();
        } else {
          RAISED.set(previous);
        }
      }
    }

    private static boolean isWorkflowThreadTeardown(Throwable throwable) {
      return throwable instanceof Error
          && throwable.getClass().getName().endsWith(".DestroyWorkflowThreadError");
    }
  }
}
