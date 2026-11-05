import test from 'ava';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { execute, setupTestEnv, uid, withWorker } from './helpers';
import { runEntrypoint } from './workflows';

const env = setupTestEnv(test);

test('a sync task runs as an Activity', async (t) => {
  const taskQueue = uid('sync-task');
  const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.syncTaskFunctional!) }, () =>
    execute(env(), runEntrypoint, { taskQueue, args: ['syncTaskFunctional', 'sync-task', 41] })
  );
  t.deepEqual(result, { result: 42 });
});
