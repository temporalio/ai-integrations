package io.temporal.googleadk;

/**
 * The {@code ApplicationFailure} types raised by the integration's activities, for use with {@code
 * RetryOptions.setDoNotRetry} and failure handling in workflows.
 */
public final class FailureTypes {

  /** A model call failed; GenAI API errors with status 408, 409, 429, or 5xx are retryable. */
  public static final String MODEL_ERROR = "googleadk.ModelError";

  /** An MCP toolset operation failed; unknown toolsets and tools are non-retryable. */
  public static final String MCP_ERROR = "googleadk.McpError";

  private FailureTypes() {}
}
