import test from 'ava';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { execute, setupTestEnv, uid, withWorker } from './helpers';
import { runGraph } from './workflows';

const env = setupTestEnv(test);

test('a sync node runs as an Activity', async (t) => {
  const taskQueue = uid('sync-node');
  const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.syncNode!) }, () =>
    execute(env(), runGraph, { taskQueue, args: ['syncNode', 'my-graph', { value: 'hello' }] })
  );
  t.deepEqual(result, { value: 'hello!' });
});
