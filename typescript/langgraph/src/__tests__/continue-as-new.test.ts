import test from 'ava';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { execute, setupTestEnv, uid, withWorker } from './helpers';
import { continueAsNewWorkflow } from './workflows';

const env = setupTestEnv(test);

test('continue-as-new with checkpointer state', async (t) => {
  const taskQueue = uid('my-graph');
  const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.continueAsNew!) }, () =>
    execute(env(), continueAsNewWorkflow, { taskQueue, args: [{ value: '' }] })
  );
  t.deepEqual(result, { value: 'aaa' });
});
