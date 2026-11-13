import { readFileSync } from 'node:fs';
import path from 'node:path';
import { SimplePlugin } from '@temporalio/plugin';
import { createBaseBundlerOptions, Worker } from '@temporalio/test-helpers';
import type { Worker as CoreWorker, WorkerOptions } from '@temporalio/worker';
import { bundleWorkflowCode, DefaultLogger, NativeConnection } from '@temporalio/worker';
import { AiSdkPlugin, createActivities } from '..';
import type { AiSdkPluginOptions } from '..';
import { refuseLaterReplacement } from '../activity-sources';
import { localDownload, makeAiSdkTestFunction } from './helpers/ai-sdk';
import { helloWorkflowGenerator, TestProvider } from './stubs/ai-sdk';

const test = makeAiSdkTestFunction();

test('configureWorker refuses to replace a different Activity with the same name', (t) => {
  const provider = new TestProvider(helloWorkflowGenerator());
  const files = async () => ({}) as any;
  const base = { taskQueue: 'tq' } as WorkerOptions;
  const duplicate = (name: string) => ({
    instanceOf: TypeError,
    message: new RegExp(`^Duplicate registration of Activity '${name}': AiSdkPlugin would replace`),
  });

  // A caller Activity, another instance's provider, or another instance's MCP
  // client factory under the same server name is refused.
  const callerActivities = { ...base, activities: { invokeModel: async () => 'caller' } };
  t.throws(() => new AiSdkPlugin({ modelProvider: provider }).configureWorker(callerActivities), duplicate('invokeModel'));
  const withMcp = new AiSdkPlugin({ modelProvider: provider, mcpClientFactories: { files } }).configureWorker(base);
  const otherProvider = new AiSdkPlugin({ modelProvider: new TestProvider(helloWorkflowGenerator()) });
  t.throws(() => otherProvider.configureWorker(withMcp), duplicate('invokeModel'));
  const otherFactory = new AiSdkPlugin({ modelProvider: provider, mcpClientFactories: { files: async () => ({}) as any } });
  t.throws(() => otherFactory.configureWorker(withMcp), duplicate('files-listTools'));
  const otherDownload = new AiSdkPlugin({ modelProvider: provider, download: localDownload });
  t.throws(() => otherDownload.configureWorker(withMcp), duplicate('invokeModel'));
  const otherIdle = new AiSdkPlugin({
    modelProvider: provider,
    mcpClientFactories: { files },
    mcpConnectionIdleTimeout: '1 minute',
  });
  t.throws(() => otherIdle.configureWorker(withMcp), duplicate('files-listTools'));
  // Another operation built from the same provider, registered under this name.
  const aliased = { ...base, activities: { invokeModel: (createActivities(provider) as any).invokeEmbeddingModel } };
  t.throws(() => new AiSdkPlugin({ modelProvider: provider }).configureWorker(aliased), duplicate('invokeModel'));

  // configureWorker accepts the same instance twice (NativeConnection and
  // Worker), the same options twice, and createActivities with the same
  // provider.
  const plugin = new AiSdkPlugin({ modelProvider: provider, mcpClientFactories: { files } });
  t.notThrows(() => plugin.configureWorker(plugin.configureWorker(base)));
  t.notThrows(() => new AiSdkPlugin({ modelProvider: provider, mcpClientFactories: { files } }).configureWorker(withMcp));
  const callerSet = { ...base, activities: createActivities(provider) };
  t.notThrows(() => new AiSdkPlugin({ modelProvider: provider }).configureWorker(callerSet));
});

test('duplicate Activity errors say where the other Activity is from and what to do', (t) => {
  const base = { taskQueue: 'tq' } as WorkerOptions;
  const plugin = (options: Partial<AiSdkPluginOptions> = {}) =>
    new AiSdkPlugin({ modelProvider: new TestProvider(helloWorkflowGenerator()), ...options });
  const files = async () => ({}) as any;
  const refusal = (options: Partial<AiSdkPluginOptions>, activities: object) =>
    t.throws(() => plugin(options).configureWorker({ ...base, activities: activities as any }))
      ?.message ?? '';

  const caller = refusal({}, { invokeModel: async () => 'caller' });
  t.regex(caller, /in the Worker's `activities` or from an earlier plugin\. Rename or remove/);
  t.regex(caller, /or run the other integration on its own Worker and task queue\.$/);
  const callerMcp = refusal({ mcpClientFactories: { files } }, { 'files-listTools': files });
  t.regex(callerMcp, /Rename the MCP server or the other Activity/);

  const first = plugin({ mcpClientFactories: { files } }).configureWorker(base).activities as any;
  const provider = refusal({}, { invokeModel: first.invokeModel });
  t.regex(provider, /another `AiSdkPlugin` or `createActivities` call\. Use one `AiSdkPlugin`/);
  t.regex(provider, /`customProvider\(\{ languageModels, fallbackProvider \}\)` from `ai`\.$/);
  const factory = refusal(
    { mcpClientFactories: { files: async () => ({}) as any } },
    { 'files-listTools': first['files-listTools'] }
  );
  t.regex(factory, /same factory function and `mcpConnectionIdleTimeout` for this MCP server/);
});

test('AiSdkPlugin reports the package version for worker heartbeats', (t) => {
  const plugin = new AiSdkPlugin({ modelProvider: new TestProvider(helloWorkflowGenerator()) });
  t.is(plugin.version, JSON.parse(readFileSync(path.join(__dirname, '../../package.json'), 'utf8')).version);
});

test('runWorker refuses an Activity that a later plugin replaced, before polling', async (t) => {
  const connection = await NativeConnection.connect({ address: t.context.env.address });
  // A plugin that registers `invokeModel` without its own duplicate check, for example
  // a third-party plugin or `@temporalio/strands-agents` 1.24.0.
  const later = new SimplePlugin({ name: 'later', activities: { invokeModel: async () => 'other' } });
  const worker = await Worker.create({
    connection,
    namespace: t.context.env.namespace,
    taskQueue: 'test-ai-later-plugin',
    plugins: [new AiSdkPlugin({ modelProvider: new TestProvider(helloWorkflowGenerator()) }), later],
  });
  await t.throwsAsync(worker.run(), {
    instanceOf: TypeError,
    message: /^Duplicate registration of Activity 'invokeModel': a plugin applied after AiSdkPlugin replaced/,
  });
  t.is(worker.getState(), 'STOPPED');
  // Throws if the Worker still holds the connection.
  await connection.close();
});

test('runWorker refuses an Activity that a NativeConnection plugin replaced', async (t) => {
  const activities = { invokeModel: async () => 'other' };
  const connection = await NativeConnection.connect({
    address: t.context.env.address,
    plugins: [new SimplePlugin({ name: 'later', activities })],
  });
  const worker = await Worker.create({
    connection,
    namespace: t.context.env.namespace,
    taskQueue: 'test-ai-connection-plugin',
    plugins: [new AiSdkPlugin({ modelProvider: new TestProvider(helloWorkflowGenerator()) })],
  });
  await t.throwsAsync(worker.run(), {
    instanceOf: TypeError,
    message: /^Duplicate registration of Activity 'invokeModel': a plugin applied after/,
  });
  t.is(worker.getState(), 'STOPPED');
  await connection.close();
});

test('refuseLaterReplacement rejects with the duplicate error without nativeWorker', async (t) => {
  const added = createActivities(new TestProvider(helloWorkflowGenerator()));
  const worker = {
    options: { activities: new Map([['invokeModel', async () => 'other']]) },
    nativeWorker: {},
  } as unknown as CoreWorker;
  let ran = false;
  const next = async () => {
    ran = true;
  };
  await t.throwsAsync(async () => refuseLaterReplacement('AiSdkPlugin', worker, added, next), {
    instanceOf: TypeError,
    message: /^Duplicate registration of Activity 'invokeModel'/,
  });
  t.false(ran);
});

test('a Workflow bundle accepts two AiSdkPlugins', async (t) => {
  const options = { modelProvider: new TestProvider(helloWorkflowGenerator()) };
  const bundle = await bundleWorkflowCode({
    ...createBaseBundlerOptions(),
    workflowsPath: require.resolve('./workflows/ai-sdk'),
    logger: new DefaultLogger('WARN'),
    plugins: [new AiSdkPlugin(options), new AiSdkPlugin(options)],
  });
  t.true(bundle.code.length > 0);
});

test('runWorker starts the Worker at once, so runUntil stops it after its callback', async (t) => {
  const connection = await NativeConnection.connect({ address: t.context.env.address });
  // runUntil stops the Worker after its callback only if the Worker is already
  // running. A callback that throws synchronously finishes before run() yields.
  const alone = await Worker.create({
    connection,
    namespace: t.context.env.namespace,
    taskQueue: 'test-ai-run-until-throws',
    plugins: [new AiSdkPlugin({ modelProvider: new TestProvider(helloWorkflowGenerator()) })],
  });
  await t.throwsAsync(
    alone.runUntil(() => {
      throw new Error('boom');
    }),
    { message: 'boom' }
  );
  t.is(alone.getState(), 'STOPPED');
  // Each plugin's check runs before the Worker starts; none may yield.
  const options = { modelProvider: new TestProvider(helloWorkflowGenerator()) };
  const two = await Worker.create({
    connection,
    namespace: t.context.env.namespace,
    taskQueue: 'test-ai-run-until-two',
    plugins: [new AiSdkPlugin(options), new AiSdkPlugin(options)],
  });
  t.is(await two.runUntil(() => Promise.resolve('done')), 'done');
  t.is(two.getState(), 'STOPPED');
  await connection.close();
});
