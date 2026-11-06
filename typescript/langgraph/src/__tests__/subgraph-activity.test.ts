import test from 'ava';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { execute, scheduledActivities, setupTestEnv, uid, withWorker } from './helpers';
import type { HistoryEvent } from './helpers';
import { runGraph } from './workflows';

const env = setupTestEnv(test);

test('an Activity node runs a subgraph inside the Activity', async (t) => {
  const taskQueue = uid('subgraph');
  const workflowId = uid('test-workflow');
  const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.activitySubgraph!) }, () =>
    execute(env(), runGraph, { taskQueue, workflowId, args: ['activitySubgraph', 'parent', { value: '' }] })
  );
  t.deepEqual(result, { value: 'child' });
  const { events } = await env().client.workflow.getHandle(workflowId).fetchHistory();
  t.deepEqual(scheduledActivities((events ?? []) as HistoryEvent[]), ['parent.parent_node']);
});
