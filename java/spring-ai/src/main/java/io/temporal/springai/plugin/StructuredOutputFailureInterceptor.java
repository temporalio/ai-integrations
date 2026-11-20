package io.temporal.springai.plugin;

import io.temporal.common.interceptors.WorkerInterceptorBase;
import io.temporal.common.interceptors.WorkflowInboundCallsInterceptor;
import io.temporal.common.interceptors.WorkflowInboundCallsInterceptorBase;
import io.temporal.failure.ApplicationFailure;
import io.temporal.workflow.Workflow;
import java.util.Arrays;
import org.springframework.ai.converter.BeanOutputConverter;
import tools.jackson.core.JacksonException;

/** Classifies Spring AI output conversion errors that escape Workflow code. */
final class StructuredOutputFailureInterceptor extends WorkerInterceptorBase {
  private static final String CHANGE_ID = "spring-ai-structured-output-workflow-failure";

  @Override
  public WorkflowInboundCallsInterceptor interceptWorkflow(WorkflowInboundCallsInterceptor next) {
    return new WorkflowInboundCallsInterceptorBase(next) {
      @Override
      public WorkflowOutput execute(WorkflowInput input) {
        try {
          return super.execute(input);
        } catch (JacksonException e) {
          throw classify(e);
        }
      }

      @Override
      public void handleSignal(SignalInput input) {
        try {
          super.handleSignal(input);
        } catch (JacksonException e) {
          throw classify(e);
        }
      }

      @Override
      public UpdateOutput executeUpdate(UpdateInput input) {
        try {
          return super.executeUpdate(input);
        } catch (JacksonException e) {
          throw classify(e);
        }
      }
    };
  }

  private static RuntimeException classify(JacksonException error) {
    // Spring AI converts the answer after its advisors return. The converter frame
    // identifies this failure without making arbitrary Jackson errors fail the Workflow.
    StackTraceElement[] stack = error.getStackTrace();
    boolean conversionFailed =
        Arrays.stream(stack)
            .anyMatch(
                frame ->
                    frame.getClassName().equals(BeanOutputConverter.class.getName())
                        && frame.getMethodName().equals("convert"));
    if (!conversionFailed
        || Workflow.getVersion(CHANGE_ID, Workflow.DEFAULT_VERSION, 1)
            == Workflow.DEFAULT_VERSION) {
      return error;
    }
    ApplicationFailure failure =
        ApplicationFailure.newFailureWithCause(
            error.getMessage(), error.getClass().getName(), error.getCause());
    failure.setStackTrace(stack);
    return failure;
  }
}
