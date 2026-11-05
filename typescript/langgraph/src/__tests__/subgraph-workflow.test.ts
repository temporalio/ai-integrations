import test from 'ava';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { execute, scheduledActivities, setupTestEnv, uid, withWorker } from './helpers';
import type { HistoryEvent } from './helpers';
import { runGraph } from './workflows';

const env = setupTestEnv(test);

test('a Workflow node runs a registered subgraph whose node is an Activity', async (t) => {
  const taskQueue = uid('subgraph');
  const workflowId = uid('test-workflow');
  const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.workflowSubgraph!) }, () =>
    execute(env(), runGraph, { taskQueue, workflowId, args: ['workflowSubgraph', 'parent', { value: '' }] })
  );
  t.deepEqual(result, { value: 'child' });
  const { events } = await env().client.workflow.getHandle(workflowId).fetchHistory();
  t.deepEqual(scheduledActivities((events ?? []) as HistoryEvent[]), ['child.child_node']);
});
