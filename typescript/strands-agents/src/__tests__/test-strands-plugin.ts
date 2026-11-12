import { readFileSync } from 'node:fs';
import path from 'node:path';
import type { McpClient, Model } from '@strands-agents/sdk';
import type { TestFn } from 'ava';
import { SimplePlugin } from '@temporalio/plugin';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest } from '@temporalio/test-helpers';
import { NativeConnection, Worker, type WorkerOptions } from '@temporalio/worker';
import { StrandsPlugin } from '..';
import { setupTestEnv, StubMcpClient, StubModel } from './helpers/strands';

const test = anyTest as TestFn<BaseContext>;

setupTestEnv(test);

test('StrandsPlugin refuses to replace a different Activity with the same name', (t) => {
  type Activities = Record<string, unknown>;
  const claude = (): Model => new StubModel([]);
  const beta = (): McpClient => new StubMcpClient([]);
  const base = { taskQueue: 'q' } as WorkerOptions;
  const duplicate = (name: string) => ({
    instanceOf: TypeError,
    message:
      `Duplicate registration of Activity '${name}': StrandsPlugin would replace a different Activity with this ` +
      'name. Rename or remove the other Activity, run the other integration on its own Worker and task queue, or ' +
      'use one StrandsPlugin instance on this Worker.',
  });
  const plugin = new StrandsPlugin({ models: { claude }, mcpClients: { beta } });
  const callerActivity = async (): Promise<string> => 'caller';

  // A caller Activity is refused.
  for (const name of ['invokeModel', 'beta-callTool']) {
    t.throws(() => plugin.configureWorker({ ...base, activities: { [name]: callerActivity } }), duplicate(name));
  }
  // So is a StrandsPlugin with different options: no `models` or another `models` object (compared by
  // identity), another factory for an MCP server, or another idle timeout for it.
  const withModels = new StrandsPlugin({ models: { claude } }).configureWorker(base);
  t.throws(() => new StrandsPlugin({ mcpClients: { beta } }).configureWorker(withModels), duplicate('invokeModel'));
  t.throws(() => new StrandsPlugin({ models: { claude } }).configureWorker(withModels), duplicate('invokeModel'));
  const withBeta = new StrandsPlugin({ mcpClients: { beta } }).configureWorker(base);
  const otherFactory = new StrandsPlugin({ mcpClients: { beta: () => beta() } });
  t.throws(() => otherFactory.configureWorker(withBeta), duplicate('beta-listTools'));
  const otherTimeout = new StrandsPlugin({ mcpClients: { beta }, mcpConnectionIdleTimeout: '1 minute' });
  t.throws(() => otherTimeout.configureWorker(withBeta), duplicate('beta-listTools'));

  // Caller Activities with other names stay beside the plugin's.
  const once = plugin.configureWorker({ ...base, activities: { lookUp: callerActivity } });
  t.is((once.activities as Activities).lookUp, callerActivity);
  // Worker.create runs configureWorker twice when one instance is on both the NativeConnection
  // and the Worker.
  const twice = plugin.configureWorker(once);
  t.is((twice.activities as Activities).invokeModel, (once.activities as Activities).invokeModel);
  // Instances built from one options object, two default instances, and an omitted idle timeout
  // next to its five-minute default still start.
  const shared = { models: { claude }, mcpClients: { beta } };
  t.notThrows(() => new StrandsPlugin(shared).configureWorker(new StrandsPlugin(shared).configureWorker(base)));
  t.notThrows(() => new StrandsPlugin().configureWorker(new StrandsPlugin().configureWorker(base)));
  const fiveMinutes = new StrandsPlugin({ mcpClients: { beta }, mcpConnectionIdleTimeout: '5 minutes' });
  t.notThrows(() => fiveMinutes.configureWorker(withBeta));
  // Different MCP server names from two instances are both kept.
  const both = new StrandsPlugin({ mcpClients: { gamma: beta } }).configureWorker(withBeta).activities as Activities;
  t.is(typeof both['beta-callTool'], 'function');
  t.is(typeof both['gamma-callTool'], 'function');
});

test('runWorker refuses an Activity that a later plugin replaced, before polling', async (t) => {
  const connection = await NativeConnection.connect({ address: t.context.env.address });
  // AiSdkPlugin also registers invokeModel; this plugin stands in for it.
  const later = new SimplePlugin({ name: 'later', activities: { invokeModel: async () => 'other' } });
  const worker = await Worker.create({
    connection,
    namespace: t.context.env.namespace,
    taskQueue: 'test-strands-later-plugin',
    plugins: [new StrandsPlugin({ models: { test: () => new StubModel([]) } }), later],
  });
  await t.throwsAsync(worker.run(), {
    instanceOf: TypeError,
    message: /^Duplicate registration of Activity 'invokeModel': a plugin applied after StrandsPlugin replaced/,
  });
  t.is(worker.getState(), 'STOPPED');
  // Throws if the Worker still holds the connection.
  await connection.close();
});

test('StrandsPlugin reports the package version for worker heartbeats', (t) => {
  t.is(new StrandsPlugin({ models: { test: () => new StubModel([]) } }).version, JSON.parse(readFileSync(path.join(__dirname, '../../package.json'), 'utf8')).version);
});
