package io.temporal.langchain4j.internal;

import dev.langchain4j.exception.NonRetriableException;
import dev.langchain4j.exception.ToolArgumentsException;
import io.temporal.failure.ApplicationFailure;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Classifies chat model and MCP failures for the activity's retry policy. The provider's message,
 * stack trace and causes can hold request data, such as the prompt, tool arguments or an API key in
 * a URL, so they stay in the worker log; the history records only the exception type.
 */
public final class Failures {

  private static final Logger log = LoggerFactory.getLogger(Failures.class);

  /** The failure type of a chat model error. */
  public static final String MODEL_ERROR = "LangChain4jModelError";

  /** The failure type of an MCP client error. */
  public static final String MCP_ERROR = "LangChain4jMcpError";

  /**
   * The failure type when an MCP server rejects a tool's arguments; the workflow sees it as
   * LangChain4j's {@code ToolArgumentsException}.
   */
  public static final String MCP_TOOL_ARGUMENTS_ERROR = "LangChain4jMcpToolArgumentsError";

  /** How far down a cause chain a LangChain4j exception is looked for; it also ends a cycle. */
  private static final int MAX_DEPTH = 16;

  private Failures() {}

  /**
   * Converts a chat model exception. An exception that LangChain4j marks as final, a {@link
   * NonRetriableException} such as an authentication or invalid-request error, is not retried; the
   * retry policy decides for everything else.
   */
  public static ApplicationFailure modelFailure(String model, Throwable cause) {
    log.warn("Chat model {} failed", model, cause);
    return classify("Chat model " + model, cause, MODEL_ERROR);
  }

  /**
   * Converts an MCP client exception. A rejection of the tool's arguments ({@link
   * ToolArgumentsException}) and an exception that LangChain4j marks as final are not retried; the
   * retry policy decides for everything else, such as a transport failure.
   *
   * @param subject what failed, for example {@code MCP tool 'search' of client 'docs'}
   */
  public static ApplicationFailure mcpFailure(String subject, Throwable cause) {
    log.warn("{} failed", subject, cause);
    if (find(cause, ToolArgumentsException.class) != null) {
      return ApplicationFailure.newNonRetryableFailure(
          subject + " failed: the server rejected the tool arguments", MCP_TOOL_ARGUMENTS_ERROR);
    }
    return classify(subject, cause, MCP_ERROR);
  }

  private static ApplicationFailure classify(String subject, Throwable cause, String type) {
    String message = subject + " failed with " + cause.getClass().getName();
    if (find(cause, NonRetriableException.class) != null) {
      return ApplicationFailure.newNonRetryableFailure(message, type);
    }
    return ApplicationFailure.newFailure(message, type);
  }

  @Nullable
  private static <T extends Throwable> T find(Throwable throwable, Class<T> type) {
    Throwable current = throwable;
    for (int depth = 0; current != null && depth < MAX_DEPTH; depth++) {
      if (type.isInstance(current)) {
        return type.cast(current);
      }
      Throwable next = current.getCause();
      current = next == current ? null : next;
    }
    return null;
  }
}
