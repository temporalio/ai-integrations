import { heartbeat, activityInfo } from '@temporalio/activity';

/**
 * Starts a recurring heartbeat loop driven by the active Activity's
 * `heartbeatTimeoutMs`. The loop heartbeats every `heartbeatTimeoutMs / 2`,
 * starting half the timeout in, so each heartbeat lands well inside the
 * timeout. A call that ends sooner sends none.
 *
 * Returns a `stop` function that cancels the loop. If the Activity has no
 * `heartbeatTimeoutMs` configured, no heartbeats are sent and `stop` is a no-op.
 * Each heartbeat sends what `details` returns.
 *
 * Heartbeat errors are swallowed: the Activity may already be cancelled, in
 * which case the next `heartbeat()` call would throw and we don't want that
 * to mask the Activity's primary work.
 */
export function startAdaptiveHeartbeat(details?: () => unknown): () => void {
  const info = activityInfo();
  if (!info.heartbeatTimeoutMs || info.heartbeatTimeoutMs <= 0) {
    return () => {};
  }

  const interval = info.heartbeatTimeoutMs / 2;
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const tick = () => {
    if (stopped) return;
    try {
      heartbeat(details?.());
    } catch {
      // Activity might be cancelled — ignore heartbeat errors
    }
    timer = setTimeout(tick, interval);
  };

  timer = setTimeout(tick, interval);

  return () => {
    stopped = true;
    if (timer) clearTimeout(timer);
  };
}
