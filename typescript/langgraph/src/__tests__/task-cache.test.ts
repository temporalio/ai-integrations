import { createHash } from 'node:crypto';
import test from 'ava';
import { cacheKey } from '../task-cache';

// Keys are carried through continue-as-new, so a change here breaks resumed runs.
for (const [name, taskId, args, context, key] of [
  ['no arguments', 'task', [], null, '930230e41ba553a51dc4c199a5c25ab5'],
  [
    'object argument and context',
    'my-graph.node',
    [{ value: 'hello' }],
    { userId: 'alice' },
    'a1f8711bf91f5a178b4c898780631921',
  ],
  [
    'non-ASCII text and tagged types',
    'task_b',
    [10, 'é日本語🦜', new Date(0), new Uint8Array([1, 2, 3]), new Set(['b', 'a']), new Map([['k', 1]])],
    { tenant: { name: 'acme' } },
    'c21cc168b17021ea635bd23b56edb1df',
  ],
] as const) {
  test(`cacheKey is stable: ${name}`, (t) => {
    t.is(cacheKey(taskId, [...args], context), key);
  });
}

test('cacheKey is the first 32 hex digits of the SHA-256 of the canonical JSON, as in Python', (t) => {
  const text = JSON.stringify(['my-graph.node', [{ value: 'hello' }], { userId: 'alice' }]);
  t.is(
    cacheKey('my-graph.node', [{ value: 'hello' }], { userId: 'alice' }),
    createHash('sha256').update(text, 'utf8').digest('hex').slice(0, 32)
  );
});
