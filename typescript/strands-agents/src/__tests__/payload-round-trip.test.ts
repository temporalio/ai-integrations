import test from 'ava';
import { findPayloadChange, type PayloadChange } from './helpers/payload-round-trip';

test('JSON values survive the payload converter', (t) => {
  t.is(findPayloadChange({ a: [1, 'two', null, { b: true }], c: { 'x-y': 'z' } }), undefined);
});

test('a missing field and an undefined field count as the same', (t) => {
  t.is(findPayloadChange({ a: undefined, b: { c: undefined } }), undefined);
});

test('a top-level Uint8Array survives as a binary payload', (t) => {
  t.is(findPayloadChange(new Uint8Array([0, 1, 255])), undefined);
});

const date = new Date(0);
const changed: [string, unknown, PayloadChange][] = [
  [
    'nested Uint8Array',
    { file: { data: new Uint8Array([1, 2]) } },
    { path: '$.file.data', sent: 'Uint8Array', received: 'Object' },
  ],
  [
    'URL',
    { parts: [{ url: new URL('https://example.test/a.png') }] },
    { path: '$.parts[0].url', sent: 'URL', received: '"https://example.test/a.png"' },
  ],
  [
    'Date',
    { response: { timestamp: date } },
    { path: '$.response.timestamp', sent: 'Date', received: JSON.stringify(date.toISOString()) },
  ],
  [
    'AbortSignal',
    { abortSignal: new AbortController().signal },
    { path: '$.abortSignal', sent: 'AbortSignal', received: 'Object' },
  ],
  [
    'Map',
    { headers: new Map([['a', 'b']]) },
    { path: '$.headers', sent: 'Map', received: 'Object' },
  ],
  [
    'undefined array item',
    { values: ['a', undefined] },
    { path: '$.values[1]', sent: 'undefined', received: 'null' },
  ],
  [
    'NaN under a non-identifier key',
    { 'x-y': NaN },
    { path: '$["x-y"]', sent: 'NaN', received: 'null' },
  ],
];

for (const [name, value, expected] of changed) {
  test(`${name} does not survive the payload converter`, (t) => {
    t.deepEqual(findPayloadChange(value), expected);
  });
}
