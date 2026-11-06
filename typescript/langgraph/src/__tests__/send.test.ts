import test from 'ava';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { execute, setupTestEnv, uid, withWorker } from './helpers';
import { runGraph } from './workflows';

const env = setupTestEnv(test);

test('conditional edges fan out with Send', async (t) => {
  const taskQueue = uid('send');
  const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.send!) }, () =>
    execute(env(), runGraph, { taskQueue, args: ['send', 'my-graph', { items: ['a', 'b', 'c'], results: [] }] })
  );
  t.deepEqual(result.items, ['a', 'b', 'c']);
  t.deepEqual([...result.results].sort(), ['A', 'B', 'C']);
});
