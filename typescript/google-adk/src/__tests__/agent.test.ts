/**
 * Native-runtime integration test — the load-bearing proof that an existing
 * ADK agent becomes durable with no rewrite. A vanilla `LlmAgent` is driven by
 * the SDK's own `InMemoryRunner.runEphemeral` loop inside a Workflow; the only
 * change from plain ADK is the one-line `model: new TemporalModel(...)` swap plus
 * registering `GoogleAdkPlugin` on the worker. The agent's loop runs
 * deterministically in the Workflow while each model turn becomes an Activity.
 */

import test from 'ava';
import { SimplePlugin } from '@temporalio/plugin';
import { NativeConnection, Worker } from '@temporalio/worker';

import { GoogleAdkPlugin } from '../index';
import { fakeModelProvider } from '../testing';
import { countScheduledActivities, setupTestEnv, uid, withWorker } from './helpers';
import { agentRunnerWorkflow } from './workflows';

const getEnv = setupTestEnv(test);

// Native ADK Runner integration (E2E)
test.serial('runsLlmAgentThroughRunnerWithDurableModelCalls', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent');
  const workflowId = uid('wf-agent');
  const plugin = new GoogleAdkPlugin({ modelProvider: fakeModelProvider() });

  // Cache size 0 forces a full replay each task, so the scheduled-activity
  // count below reflects exactly the model turns the native runner made.
  const result = await withWorker(env, { taskQueue, plugins: [plugin], maxCachedWorkflows: 0 }, () =>
    env.client.workflow.execute(agentRunnerWorkflow, {
      taskQueue,
      workflowId,
      args: ['What is durable execution?'],
    })
  );

  // The native `InMemoryRunner` surfaced the model's final text, proving the
  // whole agent loop ran to completion inside the Workflow.
  t.is(result, 'fake-response:fake-model');

  // ...and that final text came from a durable Activity: the `LlmAgent`'s
  // single model turn routed through `adk-invokeModel` exactly once.
  const { events } = await env.client.workflow.getHandle(workflowId).fetchHistory();
  t.is(countScheduledActivities(events ?? [], 'adk-invokeModel'), 1);
});

test.serial('runWorker refuses an Activity that a later plugin replaced, before polling', async (t) => {
  const env = getEnv();
  const connection = await NativeConnection.connect({ address: env.address });
  const later = new SimplePlugin({ name: 'later', activities: { 'adk-invokeModel': async () => 'other' } });
  const worker = await Worker.create({
    connection,
    namespace: env.namespace,
    taskQueue: uid('adk-later-plugin'),
    plugins: [new GoogleAdkPlugin(), later],
  });
  await t.throwsAsync(worker.run(), {
    instanceOf: TypeError,
    message: /^Duplicate registration of Activity 'adk-invokeModel': a plugin applied after GoogleAdkPlugin replaced/,
  });
  t.is(worker.getState(), 'STOPPED');
  // Throws if the Worker still holds the connection.
  await connection.close();
});

test.serial('runWorker starts the Worker at once, so runUntil stops it after its callback', async (t) => {
  const env = getEnv();
  const connection = await NativeConnection.connect({ address: env.address });
  // runUntil stops the Worker after its callback only if the Worker is already
  // running. A callback that throws synchronously finishes before run() yields.
  const alone = await Worker.create({
    connection,
    namespace: env.namespace,
    taskQueue: uid('adk-run-until-throws'),
    plugins: [new GoogleAdkPlugin()],
  });
  await t.throwsAsync(
    alone.runUntil(() => {
      throw new Error('boom');
    }),
    { message: 'boom' }
  );
  t.is(alone.getState(), 'STOPPED');
  // Each plugin's check runs before the Worker starts; none may yield.
  const options = { modelProvider: fakeModelProvider() };
  const two = await Worker.create({
    connection,
    namespace: env.namespace,
    taskQueue: uid('adk-run-until-two'),
    plugins: [new GoogleAdkPlugin(options), new GoogleAdkPlugin(options)],
  });
  t.is(await two.runUntil(() => Promise.resolve('done')), 'done');
  t.is(two.getState(), 'STOPPED');
  await connection.close();
});
