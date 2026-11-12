/**
 * E2E tests for SSE streaming. With `streamingTopic` set, a streamed model call
 * publishes incremental `LlmResponse` chunks via `@temporalio/workflow-streams`
 * while the Workflow still receives the full, ordered transcript via the
 * Activity result (the deterministic, replay-safe channel).
 */

import test from 'ava';
import { BaseLlm, type BaseLlmConnection, type LlmRequest, type LlmResponse } from '@google/adk';
import { ApplicationFailure, CancelledFailure } from '@temporalio/common';
import { MockActivityEnvironment } from '@temporalio/testing';
import { FlushTimeoutError, WorkflowStreamClient } from '@temporalio/workflow-streams/client';

import { createModelActivities } from '../activities';
import { GoogleAdkPlugin } from '../index';
import { fakeModelProvider } from '../testing';
import { findInCauseChain, setupTestEnv, uid, withWorker } from './helpers';
import { closeStream, streamingModelCall, streamingModelCallSubscribed } from './workflows';

const chunks: LlmResponse[] = [
  { content: { role: 'model', parts: [{ text: 'Hello ' }] }, partial: true },
  { content: { role: 'model', parts: [{ text: 'streaming ' }] }, partial: true },
  { content: { role: 'model', parts: [{ text: 'world' }] }, turnComplete: true },
];

const getEnv = setupTestEnv(test);

// Streaming (SSE) model boundary (E2E). Nothing subscribes to the topic here:
// publishing to an unsubscribed `streamingTopic` must not stall or drop the
// transcript the Workflow gets back.
test.serial('streamsPartialResponses', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-stream');
  const plugin = new GoogleAdkPlugin({ modelProvider: fakeModelProvider(chunks) });
  const result = await withWorker(env, { taskQueue, plugins: [plugin] }, () =>
    env.client.workflow.execute(streamingModelCall, {
      taskQueue,
      workflowId: uid('wf-stream'),
    })
  );
  t.is(result.chunks, 3);
  t.is(result.text, 'Hello streaming world');
});

// E2E over the side-channel itself: subscribe to the stream topic and assert
// the published chunks — not just the transcript returned to the Workflow.
test.serial('publishesChunksToStreamTopic', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-stream-topic');
  const plugin = new GoogleAdkPlugin({ modelProvider: fakeModelProvider(chunks) });
  await withWorker(env, { taskQueue, plugins: [plugin] }, async () => {
    const handle = await env.client.workflow.start(streamingModelCallSubscribed, {
      taskQueue,
      workflowId: uid('wf-stream-topic'),
      args: ['500 milliseconds'],
    });

    const streamClient = new WorkflowStreamClient(handle);
    const published: LlmResponse[] = [];
    const gen = streamClient.subscribe<LlmResponse>('adk-test-stream', 0, { pollCooldown: 0, resultType: true });
    for await (const item of gen) {
      published.push(item.data);
      if (published.length >= chunks.length) {
        await gen.return();
        break;
      }
    }
    await handle.signal(closeStream);
    const result = await handle.result();

    t.is(result.chunks, 3);
    t.is(result.text, 'Hello streaming world');
    t.deepEqual(
      published.map((r) => r.content?.parts?.[0]?.text),
      ['Hello ', 'streaming ', 'world']
    );
  });
});

// E2E: the chunks arrive back to back, so they share one publish batch. With
// 300,000-character chunks the batch (about 2.4 MB) is over the dev server's 2 MiB
// payload limit; with 600,000-character chunks (about 4.8 MB) it is over the 4 MiB
// gRPC message limit. The server refuses the batch, so the Workflow must fail with a
// non-retryable stream error, not complete while subscribers get nothing, and the
// model must not be called again.
test.serial('refusedStreamPublishFailsTheModelCall', async (t) => {
  const env = getEnv();
  for (const [size, refusal] of [
    [300_000, /Blob data size exceeds limit/],
    [600_000, /larger than max/],
  ] as const) {
    const taskQueue = uid('adk-stream-refused');
    const third = 'x'.repeat(size);
    const responses: LlmResponse[] = [
      { content: { role: 'model', parts: [{ text: third }] }, partial: true },
      { content: { role: 'model', parts: [{ text: third }] }, partial: true },
      { content: { role: 'model', parts: [{ text: third }] }, partial: true },
      { content: { role: 'model', parts: [{ text: third.repeat(3) }] }, turnComplete: true },
    ];
    const provider = fakeModelProvider(responses);
    let modelCalls = 0;
    const plugin = new GoogleAdkPlugin({
      modelProvider: (model) => {
        modelCalls++;
        return provider(model);
      },
    });
    await withWorker(env, { taskQueue, plugins: [plugin] }, async () => {
      let caught: unknown;
      try {
        await env.client.workflow.execute(streamingModelCall, {
          taskQueue,
          workflowId: uid('wf-stream-refused'),
          workflowExecutionTimeout: '1 minute',
        });
      } catch (err) {
        caught = err;
      }
      const appFailure = findInCauseChain(caught, ApplicationFailure);
      t.not(appFailure, undefined, `chunk ${size}`);
      t.is(appFailure?.type, 'GoogleAdkStreamError');
      t.is(appFailure?.nonRetryable, true);
      t.regex(appFailure?.message ?? '', /'fake-model'.*'adk-test-stream'/);
      t.regex(appFailure?.message ?? '', refusal);
      t.not(appFailure?.cause, undefined);
      t.is(modelCalls, 1);
    });
  }
});

/** A model that yields one chunk, then raises a retryable (HTTP 503) error. */
class MidStreamThrowingLlm extends BaseLlm {
  override async *generateContentAsync(
    _llmRequest: LlmRequest,
    _stream?: boolean,
    _abortSignal?: AbortSignal
  ): AsyncGenerator<LlmResponse, void> {
    yield { content: { role: 'model', parts: [{ text: 'partial ' }] }, partial: true };
    throw Object.assign(new Error('mid-stream failure'), { status: 503 });
  }

  override async connect(_llmRequest: LlmRequest): Promise<BaseLlmConnection> {
    throw new Error('MidStreamThrowingLlm does not connect.');
  }
}

// Unit test (MockActivityEnvironment): when the model throws mid-stream, the
// surfaced failure must be the classified model error — a rejection from the
// stream's dispose must not mask it — and the stream must still be disposed.
test.serial('midStreamModelErrorIsClassifiedAndStreamDisposed', async (t) => {
  const original = WorkflowStreamClient.fromWithinActivity;
  let disposed = false;
  let published = 0;
  WorkflowStreamClient.fromWithinActivity = (() => ({
    topic: () => ({
      publish: () => {
        published++;
      },
    }),
    [Symbol.asyncDispose]: async () => {
      disposed = true;
      throw new Error('dispose failed');
    },
  })) as unknown as typeof WorkflowStreamClient.fromWithinActivity;
  try {
    const activities = createModelActivities({
      modelProvider: (model) => new MidStreamThrowingLlm({ model }),
    });
    const mockEnv = new MockActivityEnvironment();
    const err = await t.throwsAsync(
      mockEnv.run(activities['adk-invokeModelStreaming'], {
        model: 'mid-stream',
        request: {} as never,
        streamingTopic: 'adk-test-stream',
      })
    );
    t.true(err instanceof ApplicationFailure, `expected ApplicationFailure, got: ${err}`);
    t.is((err as ApplicationFailure).type, 'GoogleAdkModelError.503');
    t.true(disposed);
    t.is(published, 1);
  } finally {
    WorkflowStreamClient.fromWithinActivity = original;
  }
});

// Unit test (MockActivityEnvironment): the model call succeeds and the stream's
// dispose rejects. Rate limiting stays retryable, a batch Workflow Streams gave up
// on is final, and a cancel before a refused dispose is reported as a cancel.
test.serial('streamDisposeFailureIsClassified', async (t) => {
  const original = WorkflowStreamClient.fromWithinActivity;
  const stubDispose = (dispose: () => Promise<void>) => {
    WorkflowStreamClient.fromWithinActivity = (() => ({
      topic: () => ({ publish: () => {} }),
      [Symbol.asyncDispose]: dispose,
    })) as unknown as typeof WorkflowStreamClient.fromWithinActivity;
  };
  const run = (mockEnv: MockActivityEnvironment) =>
    t.throwsAsync(
      mockEnv.run(createModelActivities({ modelProvider: fakeModelProvider(chunks) })['adk-invokeModelStreaming'], {
        model: 'fake-model',
        request: {} as never,
        streamingTopic: 'adk-test-stream',
      })
    );
  try {
    for (const [name, disposeError, nonRetryable] of [
      ['rate limited', Object.assign(new Error('namespace rate limit exceeded'), { code: 8 }), false],
      ['gave up', new FlushTimeoutError('gave up'), true],
    ] as const) {
      stubDispose(async () => {
        throw disposeError;
      });
      const err = await run(new MockActivityEnvironment());
      t.true(err instanceof ApplicationFailure, `${name}: ${err}`);
      t.is((err as ApplicationFailure).type, 'GoogleAdkStreamError', name);
      t.is((err as ApplicationFailure).nonRetryable, nonRetryable, name);
    }
    const mockEnv = new MockActivityEnvironment();
    stubDispose(async () => {
      mockEnv.cancel();
      throw Object.assign(new Error('Blob data size exceeds limit.'), { code: 3 });
    });
    const err = await run(mockEnv);
    t.true(err instanceof CancelledFailure, `cancelled: ${err}`);
  } finally {
    WorkflowStreamClient.fromWithinActivity = original;
  }
});
