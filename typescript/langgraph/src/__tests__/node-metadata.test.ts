import test from 'ava';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { execute, setupTestEnv, uid, withWorker } from './helpers';
import { runGraph } from './workflows';

const env = setupTestEnv(test);

test('user metadata is readable in the node, plugin options are not', async (t) => {
  const taskQueue = uid('my-graph');
  const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.nodeMetadata!) }, () =>
    execute(env(), runGraph, { taskQueue, args: ['nodeMetadata', 'my-graph', { value: 'prefix-' }] })
  );
  t.deepEqual(result, { value: 'prefix-my_value:false' });
});
