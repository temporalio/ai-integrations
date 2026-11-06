import test from 'ava';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { execute, setupTestEnv, uid, withWorker } from './helpers';
import { runGraph } from './workflows';

const env = setupTestEnv(test);

const EXPECTED_TYPES = {
  message: 'AIMessage:search',
  when: 'Date:2024-01-02T03:04:05.000Z',
  data: 'Uint8Array:0,255',
};

for (const first of ['activity', 'workflow'] as const) {
  for (const [contextKind, context, user] of [
    ['none', undefined, ''],
    ['object', { userId: 'alice' }, 'alice'],
    ['nested', { userId: 'alice', tenant: { name: 'acme' } }, 'alice:acme'],
  ] as const) {
    test(`two nodes, first in ${first}, ${contextKind} context`, async (t) => {
      const taskQueue = uid('my-graph');
      const plugin = new LangGraphPlugin(registry[`twoNodes.${first}`]!);
      const result = await withWorker(env(), { taskQueue, plugin }, () =>
        execute(env(), runGraph, {
          taskQueue,
          args: [`twoNodes.${first}`, 'my-graph', { value: '' }, { context }],
        })
      );
      t.is(result.value, `a${user}b${user}`);
      t.deepEqual(result.typesB, EXPECTED_TYPES);
      t.is(result.message.kwargs.tool_calls[0].name, 'search');
    });
  }
}

for (const first of ['activity', 'workflow'] as const) {
  test(`two nodes with a zod state schema, first in ${first}`, async (t) => {
    const taskQueue = uid('my-graph');
    const plugin = new LangGraphPlugin(registry[`zodState.${first}`]!);
    const result = await withWorker(env(), { taskQueue, plugin }, () =>
      execute(env(), runGraph, { taskQueue, args: [`zodState.${first}`, 'my-graph', { value: '', messages: [] }] })
    );
    t.is(result.messages[0].kwargs.tool_calls[0].name, 'search');
    delete result.messages;
    t.deepEqual(result, { value: '', answer: { value: 1 }, answer_type_b: 'object', message_b: 'AIMessage:search' });
  });
}
