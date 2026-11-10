/**
 * Replay determinism. Besides replaying freshly recorded histories, the
 * scenarios in `RECORDED` are replayed from histories checked in under
 * `histories/`, so a change to LangGraph's or the plugin's command stream, or
 * to the task-cache keys a continued run looks up, is caught against a frozen
 * recording. Refresh them with `UPDATE_LANGGRAPH_HISTORIES=1 make test`.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import test from 'ava';
import type { HistoryAndWorkflowId } from '@temporalio/common';
import { historyToJSON } from '@temporalio/common/lib/proto-utils';
import { Worker } from '@temporalio/worker';
import { LangGraphPlugin, defineLangGraph } from '../index';
import { registry } from './graphs';
import { REUSE_V8_CONTEXT, execute, replay, setupTestEnv, uid, withWorker, workflowBundle } from './helpers';
import { interruptWorkflow, partialExecutionWorkflow, runEntrypoint, runGraph } from './workflows';

const env = setupTestEnv(test);

const historiesDir = path.resolve(__dirname, '../../src/__tests__/histories');
const UPDATE_HISTORIES = process.env.UPDATE_LANGGRAPH_HISTORIES === '1';

for (const [kind, context] of [
  ['none', undefined],
  ['object', { userId: 'alice' }],
  ['nested', { userId: 'bob', tenant: { name: 'acme' } }],
] as const) {
  test(`a two-node run replays, context: ${kind}`, async (t) => {
    const plugin = new LangGraphPlugin(registry['twoNodes.activity']!);
    const taskQueue = uid('my-graph');
    const workflowId = uid('test-workflow');
    await withWorker(env(), { taskQueue, plugin }, () =>
      execute(env(), runGraph, {
        taskQueue,
        workflowId,
        args: ['twoNodes.activity', 'my-graph', { value: '' }, { context }],
      })
    );
    await replay(env(), plugin, workflowId);
    t.pass();
  });
}

test('an interrupted and resumed run replays', async (t) => {
  const plugin = new LangGraphPlugin(registry.interruptGraph!);
  const taskQueue = uid('interrupt-replay');
  const workflowId = uid('test-interrupt-replay');
  await withWorker(env(), { taskQueue, plugin }, () =>
    execute(env(), interruptWorkflow, { taskQueue, workflowId, args: ['', 'result'] })
  );
  await replay(env(), plugin, workflowId);
  t.pass();
});

interface Scenario {
  file: string;
  key: string;
  workflow: (...args: any[]) => Promise<unknown>;
  args: unknown[];
  check: (result: any) => boolean;
}

const RECORDED: Scenario[] = [
  {
    file: 'two_nodes_activity.json',
    key: 'twoNodes.activity',
    workflow: runGraph,
    args: ['twoNodes.activity', 'my-graph', { value: '' }, { context: { userId: 'bob', tenant: { name: 'acme' } } }],
    check: (result) => result.value === 'abob:acmebbob:acme',
  },
  {
    file: 'interrupt.json',
    key: 'interruptGraph',
    workflow: interruptWorkflow,
    args: ['', 'state'],
    check: (result) => result.value === 'yes',
  },
  {
    file: 'functional_mixed.json',
    key: 'simpleFunctionalMixed',
    workflow: runEntrypoint,
    args: ['simpleFunctionalMixed', 'e2e_simple_functional', 10],
    check: (result) => result.result === 35,
  },
  // The continued run finds the first three steps in the cache from its input.
  {
    file: 'partial_execution_continued.json',
    key: 'partialExecution',
    workflow: partialExecutionWorkflow,
    args: [{ value: 10 }],
    check: (result) => result.result === 165,
  },
];

/** Runs a scenario live and returns the history of its last run. */
async function record({ key, workflow, args, check }: Scenario) {
  const taskQueue = uid('replay-recorded');
  const workflowId = uid('replay-recorded');
  const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry[key]!) }, () =>
    execute(env(), workflow, { taskQueue, workflowId, args })
  );
  if (!check(result)) {
    throw new Error(`unexpected result ${JSON.stringify(result)}`);
  }
  return env().client.workflow.getHandle(workflowId).fetchHistory();
}

test('replays every recorded scenario live and from its checked-in fixture', async (t) => {
  const histories: HistoryAndWorkflowId[] = [];
  for (const scenario of RECORDED) {
    const fixture = path.join(historiesDir, scenario.file);
    const live = await record(scenario);
    if (UPDATE_HISTORIES) {
      writeFileSync(fixture, historyToJSON(live as Parameters<typeof historyToJSON>[0]));
      t.log(`re-recorded ${fixture}`);
    }
    histories.push({ workflowId: `${scenario.file} live`, history: live });
    // `runReplayHistories` takes a history in its JSON form too.
    histories.push({ workflowId: `${scenario.file} recorded`, history: JSON.parse(readFileSync(fixture, 'utf8')) });
  }

  // One replay Worker for every history; `runReplayHistories` reports a
  // determinism violation per history instead of throwing.
  const replayed: string[] = [];
  const options = {
    workflowBundle: await workflowBundle(),
    reuseV8Context: REUSE_V8_CONTEXT,
    plugins: [new LangGraphPlugin(defineLangGraph())],
  };
  for await (const result of Worker.runReplayHistories(options, histories)) {
    t.is(result.error, undefined, `replaying ${result.workflowId}`);
    replayed.push(result.workflowId);
  }
  t.deepEqual(
    replayed,
    histories.map((h) => h.workflowId)
  );
});
