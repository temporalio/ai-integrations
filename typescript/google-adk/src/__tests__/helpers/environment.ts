import { TestWorkflowEnvironment } from '@temporalio/testing';

// Match only SDK provisioning failures. Connection setup can fail after the
// server starts, when the SDK no longer exposes a handle for us to clean up.
// SDK 1.25 appends the last connection error to startup timeouts after a semicolon.
const RETRYABLE_STARTUP_ERROR =
  /^Failed to start ephemeral server: (?:ephemeral server at \S+ did not start within \d+s(?:\.|; last connection error:)|failed to download ephemeral server executable:|invalid ephemeral server download: could not (?:read|extract) (?:tar|ZIP) archive(?: entry)?(?:$|:))/;

/** Connect to the test launcher's server, or start one for a direct AVA run. */
export async function createTestEnvironment(): Promise<TestWorkflowEnvironment> {
  const address = process.env.TEMPORAL_TEST_ADDRESS;
  if (address) {
    return await TestWorkflowEnvironment.createFromExistingServer({ address });
  }
  return await startLocalWithRetry(() => TestWorkflowEnvironment.createLocal());
}

/**
 * Invoke a configured local server startup callback with up to three attempts.
 * Retry only SDK startup timeouts, downloads and archive read/extraction failures.
 * The caller owns all server options. Other errors and cancellation propagate immediately.
 */
export async function startLocalWithRetry<T>(start: () => Promise<T>): Promise<T> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await start();
    } catch (error) {
      if (
        attempt >= 3 ||
        !(error instanceof Error) ||
        error.name === 'AbortError' ||
        !RETRYABLE_STARTUP_ERROR.test(error.message)
      ) {
        throw error;
      }
      console.warn(`Temporal dev server startup attempt ${attempt} failed: ${error.message}`);
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
}
