import test from 'ava';
import type { Model, ModelProvider, ModelRequest, StreamEvent } from '@openai/agents-core';
import type { Client } from '@temporalio/client';
import { CancelledFailure } from '@temporalio/common';
import { MockActivityEnvironment } from '@temporalio/testing';
import { createModelActivity } from '../worker/activities';
import type { SerializedModelRequest } from '../common/serialized-model';

function minimalRequest(): SerializedModelRequest {
  return {
    input: 'hi',
    modelSettings: {},
    tools: [],
    toolsExplicitlyProvided: false,
    outputType: 'text',
    handoffs: [],
    tracing: false,
    overridePromptModel: false,
  };
}

/** Rejects like a model client does when its request signal aborts (a plain Error). */
function untilAborted(request: ModelRequest, onStart: () => void): Promise<never> {
  return new Promise((_resolve, reject) => {
    const signal = request.signal;
    if (!signal) return; // no signal: the call hangs, like a model that ignores cancellation
    signal.addEventListener('abort', () => reject(new Error('Request was aborted.')), { once: true });
    onStart();
  });
}

function provider(model: Model): ModelProvider {
  return { getModel: () => model };
}

const noopClient = {
  withAbortSignal: <R>(_signal: AbortSignal, fn: () => Promise<R>) => fn(),
  workflow: { getHandle: () => ({ async signal() {} }) },
} as unknown as Client;

test('cancelling invokeModelActivity aborts the model call and surfaces a cancellation', async (t) => {
  let started!: () => void;
  const startedP = new Promise<void>((resolve) => (started = resolve));
  const { invokeModelActivity } = createModelActivity(
    provider({
      getResponse: (request) => untilAborted(request, started),
      async *getStreamedResponse() {},
    })
  );
  const env = new MockActivityEnvironment();
  const run = env.run(invokeModelActivity, { modelName: 'm', request: minimalRequest() });
  await startedP;
  env.cancel();
  const err = await t.throwsAsync(run);
  t.true(err instanceof CancelledFailure, `expected a cancellation, got ${err}`);
});

test('cancelling invokeModelStreamActivity aborts the streamed call and surfaces a cancellation', async (t) => {
  let started!: () => void;
  const startedP = new Promise<void>((resolve) => (started = resolve));
  const { invokeModelStreamActivity } = createModelActivity(
    provider({
      async getResponse() {
        throw new Error('unused');
      },
      async *getStreamedResponse(request): AsyncIterable<StreamEvent> {
        await untilAborted(request, started);
        yield* [] as StreamEvent[];
      },
    })
  );
  const env = new MockActivityEnvironment(undefined, { client: noopClient });
  const run = env.run(invokeModelStreamActivity, {
    modelName: 'm',
    request: minimalRequest(),
    streamingTopic: 'events',
    streamingBatchInterval: '10ms',
  });
  await startedP;
  env.cancel();
  const err = await t.throwsAsync(run);
  t.true(err instanceof CancelledFailure, `expected a cancellation, got ${err}`);
});
