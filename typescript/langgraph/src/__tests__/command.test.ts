import test from 'ava';
import { Command, Send } from '@langchain/langgraph';
import { ApplicationFailure, defaultPayloadConverter } from '@temporalio/common';
import { LangGraphPlugin } from '../index';
import { UNKNOWN_VALUE_TYPE_FAILURE_TYPE } from '../workflow';
import { commandFromPayload, commandToPayload, encodeValue, restoreValue } from '../codec';
import type { ValueTypes } from '../codec';
import { registry } from './graphs';
import { execute, setupTestEnv, uid, withWorker } from './helpers';
import { runGraph } from './workflows';

const env = setupTestEnv(test);

test('command goto and update', async (t) => {
  const taskQueue = uid('command');
  const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.command!) }, () =>
    execute(env(), runGraph, { taskQueue, args: ['command', 'my-graph', { value: '' }] })
  );
  t.deepEqual(result, { value: 'ab' });
});

for (const [mode, expected] of [
  ['single', ['A']],
  ['list', ['A', 'B']],
  ['mixed', ['A', 'mixed']],
  ['update', ['A', 'dispatched']],
  ['timeout', ['A']],
  ['command_list', ['A', 'listed']],
] as const) {
  test(`command goto Send: ${mode}`, async (t) => {
    const taskQueue = uid('command-send');
    const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.commandSend!) }, () =>
      execute(env(), runGraph, { taskQueue, args: ['commandSend', 'send-graph', { mode, results: [] }] })
    );
    t.deepEqual([...result.results].sort(), [...expected].sort());
  });
}

/** `Command` normalizes `goto` to an array; its declared type doesn't say so. */
function gotoOf(command: Command): (string | Send)[] {
  return command.goto as (string | Send)[];
}

function roundTrip(command: Command): Command {
  const payload = defaultPayloadConverter.toPayload(commandToPayload(command));
  return commandFromPayload(defaultPayloadConverter.fromPayload(payload));
}

test('command payload round-trips Send', (t) => {
  const timeout = { runTimeout: 1500, idleTimeout: 2000, refreshOn: 'heartbeat' as const };
  const command = new Command({
    update: { results: ['x'] },
    goto: ['tail', new Send('worker', { item: 'a' }, { timeout })],
  });

  const restored = roundTrip(command);

  t.deepEqual(restored.update, { results: ['x'] });
  const goto = gotoOf(restored);
  t.is(goto[0], 'tail');
  const send = goto[1];
  t.true(send instanceof Send);
  t.is((send as Send).node, 'worker');
  t.deepEqual((send as Send).args, { item: 'a' });
  t.deepEqual((send as Send).timeout, timeout);

  const single = roundTrip(new Command({ goto: new Send('worker', { item: 'a' }) }));
  t.true(gotoOf(single)[0] instanceof Send);
  t.deepEqual((gotoOf(single)[0] as Send).args, { item: 'a' });
});

for (const [name, command] of [
  ['update and goto', new Command({ update: { value: 'a' }, goto: 'node_b' })],
  ['goto list', new Command({ goto: ['node_a', 'node_b'] })],
  ['parent graph', new Command({ graph: Command.PARENT, update: { value: 'a' }, goto: [] })],
  ['resume', new Command({ resume: 'yes', goto: [] })],
] as const) {
  test(`command payload round-trips ${name}`, (t) => {
    const restored = roundTrip(command);
    t.true(restored instanceof Command);
    t.is(restored.graph, command.graph);
    t.deepEqual(restored.update, command.update);
    t.deepEqual(restored.resume, command.resume);
    t.deepEqual(restored.goto, command.goto);
  });
}

test('an unknown value type ID fails with a non-retryable, typed ApplicationFailure', (t) => {
  const error = t.throws(() => restoreValue({ value: 1 }, [[['value'], 'future.Type']]), {
    instanceOf: ApplicationFailure,
  });
  t.is(error?.type, UNKNOWN_VALUE_TYPE_FAILURE_TYPE);
  t.true(error?.nonRetryable);
});

test('encoded values survive the payload converter in place', (t) => {
  const valueTypes: ValueTypes = [];
  const value = {
    result: [new Command({ goto: new Send('worker', { item: 'a', when: new Date(0) }) })],
    bytes: new Uint8Array([1, 2]),
    plain: { untouched: true },
  };
  const encoded = encodeValue(value, [], valueTypes);
  t.is((encoded as typeof value).plain, value.plain);
  const decoded = defaultPayloadConverter.fromPayload(defaultPayloadConverter.toPayload(encoded));
  const restored = restoreValue(decoded, valueTypes) as typeof value;

  const command = restored.result[0]!;
  t.true(command instanceof Command);
  t.true(gotoOf(command)[0] instanceof Send);
  t.deepEqual(restored.bytes, new Uint8Array([1, 2]));
  t.deepEqual(restored.plain, { untouched: true });
});
