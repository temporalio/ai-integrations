import { randomUUID } from 'node:crypto';
import test from 'ava';
import { Worker } from '@temporalio/worker';
import { LangSmithPlugin } from '../index';
import { InMemoryRunCollector, useSharedEnv, withTracingWorker } from './helpers';
import { NativeRunTreeWorkflow } from './workflows/native-runs';

process.env.LANGSMITH_TRACING = 'true';
const WORKFLOWS_PATH = require.resolve('./workflows/native-runs');
useSharedEnv(test);

test('native roots and children retain LangSmith semantics and identical identities on replay', async (t) => {
  const collector = new InMemoryRunCollector();
  let creates = 0;
  const create = collector.createRun;
  collector.createRun = async (run) => {
    creates++;
    await create(run);
  };
  const captures: Record<string, unknown>[][] = [];
  const sinks = { nativeRuns: { capture: {
    fn: (_info: unknown, runs: Record<string, unknown>[]) => { captures.push(runs); },
    callDuringReplay: true,
  } } };
  await withTracingWorker({
    collector,
    options: { addTemporalRuns: false, projectName: 'plugin-fallback' },
    activities: {},
    workerOptions: { workflowsPath: WORKFLOWS_PATH, maxCachedWorkflows: 0, sinks },
    body: async ({ client, taskQueue }) => {
      const handle = await client.workflow.start(NativeRunTreeWorkflow, {
        workflowId: `native-runs-${randomUUID()}`, taskQueue,
      });
      const live = await handle.result();
      const replay = new InMemoryRunCollector();
      await Worker.runReplayHistory({
        workflowsPath: WORKFLOWS_PATH,
        plugins: [new LangSmithPlugin({ client: replay.asClient(), projectName: 'plugin-fallback' })],
        sinks,
      }, await handle.fetchHistory());
      t.true(captures.length >= 2, 'capture the live execution and fresh-worker replay');
      for (const captured of captures) t.deepEqual(captured, live);
      t.deepEqual(replay.records, []);
      t.true(live.every((run) => run.native));
      t.true(live[0]!.conflictRejected);
      t.is(live[0]!.parent_run_id, null);
      t.is(live[3]!.parent_run_id, null);
      t.is(live[4]!.id, '12345678-1234-4234-8234-123456789012');
      t.is(live[4]!.start_time, 1234567890000);
      t.is(live[4]!.end_time, 1234567891000);
      for (const run of live.slice(1, 3)) {
        t.is(run.parent_run_id, live[0]!.id);
        t.true(String(run.dotted_order).startsWith(`${live[0]!.dotted_order}.`));
      }
      t.not(live[1]!.dotted_order, live[2]!.dotted_order);
      t.deepEqual(collector.byName('native-child')?.extra?.metadata, { inherited: true, completed: true });
      t.is(collector.byName('traceable-root')?.project_name, 'plugin-fallback');
      t.deepEqual(collector.byName('traceable-root')?.inputs, { processed: true });
      for (const name of ['native-root', 'native-child', 'traceable-native-child']) {
        t.deepEqual(collector.records.filter((run) => run.name === name).map((run) => run.project_name),
          ['primary', 'secondary']);
      }
    },
  });
  t.is(creates, collector.records.length, 'cache eviction does not duplicate writes');
  t.true(collector.records.every((run) => run.end_time !== undefined));
});
