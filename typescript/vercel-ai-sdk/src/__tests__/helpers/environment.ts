/**
 * Invoke a configured local server startup callback with up to three attempts.
 * The caller owns all server options. AbortError cancellation propagates immediately.
 */
export async function startLocalWithRetry<T>(start: () => Promise<T>): Promise<T> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await start();
    } catch (error) {
      if (attempt >= 3 || (error instanceof Error && error.name === 'AbortError')) {
        throw error;
      }
      console.warn(`Temporal dev server setup attempt ${attempt} failed:`, error);
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
}
