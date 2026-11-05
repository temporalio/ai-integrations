import test from 'ava';
import { LangGraphPlugin } from '../index';
import { cacheKey } from '../task-cache';
import { MARKER, executionCounts, registry, resetExecutionCounts } from './graphs';
import { execute, setupTestEnv, uid, withWorker } from './helpers';
import { cachedGraphWorkflow } from './workflows';

const env = setupTestEnv(test);

// The tests count Activity executions in this process, so they run serially.

async function runCached(key: string, graphName: string, input: unknown, phases: number) {
  const taskQueue = uid(graphName);
  return withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry[key]!) }, () =>
    execute(env(), cachedGraphWorkflow, { taskQueue, args: [{ key, graphName, input, phases }] })
  );
}

test.serial('each node executes once across three continue-as-new cycles', async (t) => {
  resetExecutionCounts();
  // 10 * 3 = 30 -> + 100 = 130 -> * 2 = 260
  const { state } = await runCached('cachedGraph', 'cached-graph', { value: 10 }, 3);
  t.deepEqual(state, { value: 260 });
  t.deepEqual(executionCounts, { multiply: 1, add: 1, double: 1 });
});

test.serial('a cached Command with Send survives continue-as-new and still routes', async (t) => {
  resetExecutionCounts();
  const { state } = await runCached('cachedFanOut', 'cached-fan-out', { items: ['a', 'b'], results: [] }, 2);
  t.deepEqual([...state.results].sort(), ['A', 'B']);
  t.deepEqual(executionCounts, { dispatch: 1, 'upper:a': 1, 'upper:b': 1 });
});

test.serial('a cached Command keeps its update and routing after continue-as-new', async (t) => {
  resetExecutionCounts();
  const { state } = await runCached('cachedCommand', 'cached-command', { value: 1 }, 2);
  t.deepEqual(state, { value: 11 });
  t.deepEqual(executionCounts, { goto_a: 1, goto_b: 1 });
});

for (const [name, marker] of [
  ['string', 'ordinary'],
  ['Command-shaped object', { graph: null, update: { value: 1 }, resume: null, goto: 'elsewhere' }],
] as const) {
  test.serial(`a cached result is never mistaken for a Command: ${name}`, async (t) => {
    resetExecutionCounts();
    const { state, cache } = await runCached('cachedOrdinary', 'cached-ordinary', { [MARKER]: marker }, 2);
    t.deepEqual(state, { [MARKER]: marker });
    t.deepEqual(Object.values(cache!), [{ [MARKER]: marker }]);
    t.deepEqual(executionCounts, { ordinary: 1 });
  });
}

class Tags {
  constructor(readonly names: Set<string>) {}
}

test('the cache key reads values, not their identity or order', (t) => {
  t.is(cacheKey('task', [new Tags(new Set(['a', 'b']))], {}), cacheKey('task', [new Tags(new Set(['b', 'a']))], {}));
  t.is(cacheKey('task', [{ b: 1, a: 2 }], null), cacheKey('task', [{ a: 2, b: 1 }], null));
  t.not(cacheKey('task', [new Tags(new Set(['a']))], {}), cacheKey('task', [{ names: ['a'] }], {}));
  t.not(cacheKey('task', [new Set(['a'])], {}), cacheKey('task', [['a']], {}));
  t.not(cacheKey('task', [new Date(0)], {}), cacheKey('task', [new Date(0).toISOString()], {}));
  t.not(cacheKey('task', [1], { user: 'a' }), cacheKey('task', [1], { user: 'b' }));
  t.not(cacheKey('task-a', [1], null), cacheKey('task-b', [1], null));
  t.regex(cacheKey('task', [], null), /^[0-9a-f]{32}$/);
});
