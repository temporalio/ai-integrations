import test from 'ava';
import { WorkflowStreamClient } from '@temporalio/workflow-streams/client';
import type { TestWorkflowEnvironment } from '@temporalio/testing';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { replay, setupTestEnv, uid, withWorker } from './helpers';
import { ackDone, astreamPublishWorkflow, streamingWorkflow } from './workflows';

const env = setupTestEnv(test);

/** Reads `topic` until a `{ done: true }` chunk, then acknowledges it. */
async function collect(testEnv: TestWorkflowEnvironment, workflowId: string, topic: string): Promise<unknown[]> {
  const handle = testEnv.client.workflow.getHandle(workflowId);
  const chunks: Record<string, unknown>[] = [];
  const stream = WorkflowStreamClient.create(testEnv.client, workflowId);
  for await (const item of stream.topic<Record<string, unknown>>(topic).subscribe(0, { pollCooldown: '10ms' })) {
    chunks.push(item.data);
    if (item.data.done) {
      await handle.signal(ackDone);
      break;
    }
  }
  return chunks;
}

for (const executeIn of ['activity', 'workflow'] as const) {
  for (const kind of ['sync', 'async'] as const) {
    test(`a ${kind} ${executeIn} node streams through the WorkflowStream`, async (t) => {
      const taskQueue = uid('streaming-ws');
      const workflowId = uid('test-streaming-ws');
      const key = `streaming.${executeIn}-${kind}`;
      const plugin = new LangGraphPlugin(registry[key]!);
      const { chunks, result } = await withWorker(
        env(),
        { taskQueue, plugin },
        async () => {
          const handle = await env().client.workflow.start(streamingWorkflow, {
            taskQueue,
            workflowId,
            args: [key, ''],
            workflowExecutionTimeout: '30s',
          });
          const chunks = await collect(env(), workflowId, 'tokens');
          return { chunks, result: await handle.result() };
        }
      );
      await replay(env(), plugin, workflowId);
      t.is(result, 'abc');
      t.deepEqual(chunks, [{ token: 'a' }, { token: 'b' }, { token: 'c' }, { done: true }]);
    });
  }
}

test('the Workflow publishes stream() chunks', async (t) => {
  const taskQueue = uid('astream-publish');
  const workflowId = uid('test-astream-publish');
  const plugin = new LangGraphPlugin(registry.astreamPublish!);
  const chunks = await withWorker(env(), { taskQueue, plugin }, async () => {
    const handle = await env().client.workflow.start(astreamPublishWorkflow, {
      taskQueue,
      workflowId,
      args: [''],
      workflowExecutionTimeout: '30s',
    });
    const chunks = await collect(env(), workflowId, 'astream');
    await handle.result();
    return chunks;
  });
  await replay(env(), plugin, workflowId);
  t.deepEqual(chunks, [{ node_a: { value: 'a' } }, { node_b: { value: 'ab' } }, { done: true }]);
});
