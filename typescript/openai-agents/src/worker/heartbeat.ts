import { heartbeat, activityInfo } from '@temporalio/activity';

/**
 * Starts a recurring heartbeat loop driven by the active Activity's
 * `heartbeatTimeoutMs`. The loop heartbeats every `heartbeatTimeoutMs / 2`,
 * starting half the timeout after the attempt was scheduled, so each heartbeat
 * lands well inside the timeout even when the Worker took a while to reach the
 * Activity. A call that ends sooner sends none.
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

  // The heartbeat timeout counts from the attempt's start, before interceptors
  // and payload decoding ran, so count that time towards the first heartbeat.
  // The attempt is scheduled no later than it starts. That time comes from the
  // server's clock, so a Worker clock up to half the timeout behind it is still in time.
  const elapsed = Date.now() - info.currentAttemptScheduledTimestampMs;
  timer = setTimeout(tick, Math.min(interval, Math.max(0, interval - elapsed)));

  return () => {
    stopped = true;
    if (timer) clearTimeout(timer);
  };
}
