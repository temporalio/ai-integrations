// Match only SDK provisioning failures. Connection setup can fail after the
// server starts, when the SDK no longer exposes a handle for us to clean up.
const RETRYABLE_STARTUP_ERROR =
  /^Failed to start ephemeral server: (?:ephemeral server at \S+ did not start within \d+s\.|failed to download ephemeral server executable:|invalid ephemeral server download: could not (?:read|extract) (?:tar|ZIP) archive(?: entry)?(?:$|:))/;

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
