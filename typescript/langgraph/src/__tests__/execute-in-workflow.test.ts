import test from 'ava';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { execute, scheduledActivities, setupTestEnv, uid, withWorker } from './helpers';
import type { HistoryEvent } from './helpers';
import { runGraph } from './workflows';

const env = setupTestEnv(test);

test('a Workflow node runs inline, without an Activity', async (t) => {
  const taskQueue = uid('my-graph');
  const workflowId = uid('test-workflow');
  const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.executeInWorkflow!) }, () =>
    execute(env(), runGraph, { taskQueue, workflowId, args: ['executeInWorkflow', 'my-graph', { value: '' }] })
  );
  t.deepEqual(result, { value: 'done' });
  const { events } = await env().client.workflow.getHandle(workflowId).fetchHistory();
  t.deepEqual(scheduledActivities((events ?? []) as HistoryEvent[]), []);
});
