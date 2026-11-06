import test from 'ava';
import { WorkflowFailedError } from '@temporalio/client';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { execute, setupTestEnv, uid, withWorker } from './helpers';
import { runGraph } from './workflows';

const env = setupTestEnv(test);

test('an Activity node fails the Workflow at its start-to-close timeout', async (t) => {
  const taskQueue = uid('my-graph');
  await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.timeout!) }, async () => {
    await t.throwsAsync(execute(env(), runGraph, { taskQueue, args: ['timeout', 'my-graph', { value: '' }] }), {
      instanceOf: WorkflowFailedError,
    });
  });
});
