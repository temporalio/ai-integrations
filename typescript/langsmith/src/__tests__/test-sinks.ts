/**
 * Worker-side sink writes: one at a time in call order, with failures logged and never thrown.
 *
 * @module
 */

import test from 'ava';
import type { Client } from 'langsmith';
import type { WriteReplica } from 'langsmith/run_trees';
import type { Logger } from '@temporalio/common';

import { LANGSMITH_SINK_NAME, createLangSmithSinks, type SerializedRun } from '../sinks';

process.env.LANGSMITH_TRACING = 'true';

const RUN: SerializedRun = {
  id: '01900000-0000-7000-8000-000000000001',
  trace_id: '01900000-0000-7000-8000-000000000001',
  dotted_order: '20240101T000000000000Z01900000-0000-7000-8000-000000000001',
  name: 'run',
  run_type: 'chain',
  start_time: 0,
  project_name: 'project',
};

function sinks(client: Partial<Client>, errors: string[] = [], replicas?: WriteReplica[]) {
  const logger = { error: (message: string) => errors.push(message) } as unknown as Logger;
  return createLangSmithSinks(client as Client, replicas, logger)[LANGSMITH_SINK_NAME];
}

test('an update does not start before its create finishes', async (t) => {
  const order: string[] = [];
  let finishCreate!: () => void;
  const createBlocked = new Promise<void>((resolve) => (finishCreate = resolve));
  let updated!: () => void;
  const updateDone = new Promise<void>((resolve) => (updated = resolve));
  const { createRun, updateRun } = sinks({
    createRun: async () => {
      order.push('create started');
      await createBlocked;
      order.push('create finished');
    },
    updateRun: async () => {
      order.push('update started');
      updated();
    },
  });

  void createRun.fn({} as never, RUN);
  void updateRun.fn({} as never, { ...RUN, end_time: 1 });
  await new Promise((resolve) => setImmediate(resolve));
  t.deepEqual(order, ['create started']);

  finishCreate();
  await updateDone;
  t.deepEqual(order, ['create started', 'create finished', 'update started']);
});

test('a failed write is logged and later writes still run', async (t) => {
  const errors: string[] = [];
  let created!: () => void;
  const createDone = new Promise<void>((resolve) => (created = resolve));
  // LangSmith swallows a failed write to a single project, but not a failed replica update.
  const { createRun, updateRun } = sinks({
    updateRun: async () => { throw new Error('rejected'); },
    createRun: async () => created(),
  }, errors, [{ projectName: 'project' }]);

  t.notThrows(() => updateRun.fn({} as never, { ...RUN, end_time: 1 }));
  void createRun.fn({} as never, RUN);
  await createDone;

  t.deepEqual(errors, ['LangSmith background I/O error: Error: rejected']);
});
