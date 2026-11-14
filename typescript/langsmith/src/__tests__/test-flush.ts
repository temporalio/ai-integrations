/**
 * Flush-on-shutdown hook.
 *
 * Observability plugins must drain any in-flight trace batches when the worker
 * stops, or traces emitted late in a run are lost. The plugin implements this in
 * its `runWorker` continuation: it always awaits the LangSmith client's
 * `awaitPendingTraceBatches()` (falling back to `flush()`) in a `finally`, so the
 * drain happens whether the worker run resolves or throws.
 *
 * These tests invoke the continuation directly with a fake worker + `next`,
 * which deterministically exercises the required hook without depending on the
 * exact worker run lifecycle.
 *
 * @module
 */

import test from 'ava';
import type { InjectedSinks, WorkerOptions } from '@temporalio/worker';

import { LangSmithPlugin } from '../index';
import { LANGSMITH_SINK_NAME, type LangSmithSinks, type SerializedRun } from '../sinks';
import { InMemoryRunCollector } from './helpers';

process.env.LANGSMITH_TRACING = 'true';

type RunWorker = (worker: unknown, next: (worker: unknown) => Promise<void>) => Promise<void>;

test('flush-on-shutdown: drains pending trace batches after the worker run resolves', async (t) => {
  const collector = new InMemoryRunCollector();
  const plugin = new LangSmithPlugin({ client: collector.asClient() });
  const runWorker = (plugin as unknown as { runWorker: RunWorker }).runWorker.bind(plugin);

  let ran = false;
  await runWorker({}, async () => {
    ran = true;
  });

  t.is(ran, true);
  t.is(collector.flushCount, 1);
});

test('flush-on-shutdown: drains pending trace batches even when the worker run throws', async (t) => {
  const collector = new InMemoryRunCollector();
  const plugin = new LangSmithPlugin({ client: collector.asClient() });
  const runWorker = (plugin as unknown as { runWorker: RunWorker }).runWorker.bind(plugin);

  await t.throwsAsync(
    runWorker({}, async () => {
      throw new Error('boom');
    }),
    { message: 'boom' }
  );

  t.is(collector.flushCount, 1);
});

test('flush-on-shutdown: hands queued sink writes to the client before flushing it', async (t) => {
  const events: string[] = [];
  const client = {
    createRun: async (run: { name: string }) => {
      await new Promise((resolve) => setTimeout(resolve, 20));
      events.push(`create ${run.name}`);
    },
    updateRun: async () => {
      events.push('update');
    },
    awaitPendingTraceBatches: async () => {
      events.push('flush');
    },
  };
  const plugin = new LangSmithPlugin({ client: client as never, projectName: 'project' });
  const runWorker = (plugin as unknown as { runWorker: RunWorker }).runWorker.bind(plugin);
  const options = plugin.configureWorker({ taskQueue: 'flush' } as WorkerOptions);
  const sink = (options.sinks as InjectedSinks<LangSmithSinks>)[LANGSMITH_SINK_NAME];
  const run: SerializedRun = {
    id: '01900000-0000-7000-8000-000000000001',
    trace_id: '01900000-0000-7000-8000-000000000001',
    dotted_order: '20240101T000000000000Z01900000-0000-7000-8000-000000000001',
    name: 'run',
    run_type: 'chain',
    start_time: 0,
    project_name: 'project',
  };

  await runWorker({}, async () => {
    void sink.createRun.fn({} as never, run);
    void sink.updateRun.fn({} as never, { ...run, end_time: 1 });
  });

  t.deepEqual(events, ['create run', 'update', 'flush']);
});
