const STARTUP_TIMEOUT = /^Failed to start ephemeral server: ephemeral server at \S+ did not start within \d+s\./;

/**
 * Invoke a configured local server startup callback, retrying only SDK startup timeouts.
 * The caller owns all server options. Other errors and cancellation propagate immediately.
 */
export async function startLocalWithRetry<T>(start: () => Promise<T>): Promise<T> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await start();
    } catch (error) {
      if (attempt >= 3 || !(error instanceof Error) || !STARTUP_TIMEOUT.test(error.message)) {
        throw error;
      }
      console.warn(`Temporal dev server startup attempt ${attempt} failed: ${error.message}`);
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
}
