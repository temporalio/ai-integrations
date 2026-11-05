import test from 'ava';
import { SimplePlugin } from '@temporalio/plugin';
import { NativeConnection, Worker } from '@temporalio/worker';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { setupTestEnv, uid } from './helpers';

const env = setupTestEnv(test);

const LATER_REPLACED =
  /^Duplicate registration of Activity 'my-graph\.node_a': a plugin applied after LangGraphPlugin replaced/;

test.serial('runWorker refuses an Activity that a later plugin replaced, before polling', async (t) => {
  const connection = await NativeConnection.connect({ address: env().address });
  const later = new SimplePlugin({ name: 'later', activities: { 'my-graph.node_a': async () => 'other' } });
  const worker = await Worker.create({
    connection,
    namespace: env().namespace,
    taskQueue: uid('langgraph-later-plugin'),
    plugins: [new LangGraphPlugin(registry['twoNodes.activity']!), later],
  });
  await t.throwsAsync(worker.run(), { instanceOf: TypeError, message: LATER_REPLACED });
  t.is(worker.getState(), 'STOPPED');
  // Throws if the Worker still holds the connection.
  await connection.close();
});

test.serial('runWorker refuses a later replacement with one instance on the connection and the Worker', async (t) => {
  const plugin = new LangGraphPlugin(registry['twoNodes.activity']!);
  const later = new SimplePlugin({ name: 'later', activities: { 'my-graph.node_a': async () => 'other' } });
  const connection = await NativeConnection.connect({ address: env().address, plugins: [plugin, later] });
  const worker = await Worker.create({
    connection,
    namespace: env().namespace,
    taskQueue: uid('langgraph-later-twice'),
    plugins: [plugin],
  });
  await t.throwsAsync(worker.run(), { instanceOf: TypeError, message: LATER_REPLACED });
  t.is(worker.getState(), 'STOPPED');
  await connection.close();
});

test.serial('runWorker starts the Worker at once, so runUntil stops it after its callback', async (t) => {
  const connection = await NativeConnection.connect({ address: env().address });
  // runUntil stops the Worker after its callback only if the Worker is already
  // running. A callback that throws synchronously finishes before run() yields.
  const alone = await Worker.create({
    connection,
    namespace: env().namespace,
    taskQueue: uid('langgraph-run-until-throws'),
    plugins: [new LangGraphPlugin(registry['twoNodes.activity']!)],
  });
  await t.throwsAsync(
    alone.runUntil(() => {
      throw new Error('boom');
    }),
    { message: 'boom' }
  );
  t.is(alone.getState(), 'STOPPED');
  // Each plugin's check runs before the Worker starts; none may yield.
  const definitions = registry['twoNodes.activity']!;
  const two = await Worker.create({
    connection,
    namespace: env().namespace,
    taskQueue: uid('langgraph-run-until-two'),
    plugins: [new LangGraphPlugin(definitions), new LangGraphPlugin(definitions)],
  });
  t.is(await two.runUntil(() => Promise.resolve('done')), 'done');
  t.is(two.getState(), 'STOPPED');
  await connection.close();
});
