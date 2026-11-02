// Workflows for histories recorded with stateful MCP operations.
// `src/__tests__/histories/stateful-mcp-*.json` were recorded with these
// Workflows on earlier versions of the plugin, and test-openai-agents.ts
// replays them on this one.
import { proxyActivities } from '@temporalio/workflow';
import { statefulMcpServer } from '../../workflow';

// Each branch's follow-up has its own Activity type, so that a replay that
// schedules them in another order fails instead of swapping their results.
const activities = proxyActivities<Record<string, () => Promise<string>>>({ startToCloseTimeout: '10 seconds' });

function replayServer(scheduleToStartTimeout: boolean) {
  return statefulMcpServer('statefulReplay', {
    config: {
      startToCloseTimeout: '20 seconds',
      retryPolicy: { maximumAttempts: 1 },
      // Without one, only the session watches over the operations.
      ...(scheduleToStartTimeout ? {} : { scheduleToStartTimeout: 0 }),
    },
  });
}

/**
 * A stateful tool call and plain Activities in parallel branches. Branch `b<n>`
 * awaits n more promises after its Activity, as code behind async helpers
 * would, so that one of them resumes in the same step as the stateful branch.
 */
export async function statefulMcpGatherWorkflow(scheduleToStartTimeout: boolean): Promise<string[]> {
  const server = replayServer(scheduleToStartTimeout);
  await server.connect();
  try {
    const stateful = async () => {
      await server.callTool('slow', {});
      return activities.statefulReplayFollowUpStateful!();
    };
    const plain = async (n: number) => {
      await activities.statefulReplayPlain!();
      for (let i = 0; i < n; i++) await Promise.resolve();
      return activities[`statefulReplayFollowUpB${n}`]!();
    };
    return await Promise.all([stateful(), ...[0, 1, 2, 3, 4, 5].map(plain)]);
  } finally {
    await server.cleanup();
  }
}

/** A stateful tool call awaited directly, then another Activity. */
export async function statefulMcpCancelWorkflow(scheduleToStartTimeout: boolean): Promise<string> {
  const server = replayServer(scheduleToStartTimeout);
  await server.connect();
  try {
    await server.callTool('slow', {});
    return await activities.statefulReplayFollowUpStateful!();
  } finally {
    await server.cleanup();
  }
}
