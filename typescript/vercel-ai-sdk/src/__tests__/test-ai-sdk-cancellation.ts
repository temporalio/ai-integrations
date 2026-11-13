import type { LanguageModelV4CallOptions } from '@ai-sdk/provider';
import { temporal } from '@temporalio/proto';
import { assertEventually, helpers } from '@temporalio/test-helpers';
import { MockActivityEnvironment } from '@temporalio/testing';
import { AiSdkPlugin, createActivities } from '..';
import { makeAiSdkTestFunction } from './helpers/ai-sdk';
import { AbortRecordingProvider } from './stubs/ai-sdk';
import {
  abortableEmbedWorkflow,
  abortableGenerateWorkflow,
  abortableStreamWorkflow,
  abortDuringEmbedWorkflow,
  abortRequest,
} from './workflows/ai-sdk';
import EventType = temporal.api.enums.v1.EventType;

const test = makeAiSdkTestFunction();

// Serial, because these Workers hold a model call open, and that slows the tests that
// run at the same time until their Workflows time out.
test.serial('abortSignal that never aborts does not reach the Activity payload', async (t) => {
  const provider = new AbortRecordingProvider(false);
  const { createWorker, executeWorkflow } = helpers(t);
  const worker = await createWorker({ plugins: [new AiSdkPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    t.is(await executeWorkflow(abortableGenerateWorkflow, { args: [false] }), 'text:not aborted');
    t.is(await executeWorkflow(abortableEmbedWorkflow), 2);
  });

  t.is(provider.signals.length, 2);
  for (const signal of provider.signals) {
    t.true(signal instanceof AbortSignal);
    t.false((signal as AbortSignal).aborted);
  }
});

test.serial('abortSignal that is already aborted does not schedule the Activity', async (t) => {
  const provider = new AbortRecordingProvider(false);
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({ plugins: [new AiSdkPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(abortableGenerateWorkflow, { args: [true] });
    t.is(await handle.result(), 'error:AbortError|true');
    const { events } = await handle.fetchHistory();
    t.false(events!.some((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED));
  });
  t.is(provider.signals.length, 0);
});

test.serial('abortSignal that aborts during generateText cancels the Activity', async (t) => {
  const provider = new AbortRecordingProvider(true);
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({ plugins: [new AiSdkPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(abortableGenerateWorkflow, { args: [false] });
    await assertEventually(t, (tt) => {
      tt.true(provider.started);
    }, 10_000);
    await handle.signal(abortRequest);
    // The AI SDK rethrows the abort reason, which `isAbortError` must recognize.
    t.is(await handle.result(), 'error:AbortError|true');
    await assertEventually(t, (tt) => {
      tt.true(provider.providerAborted);
    }, 10_000);
    const { events } = await handle.fetchHistory();
    t.true(
      events!.some((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_CANCEL_REQUESTED)
    );
  });
});

test.serial('abortSignal that aborts during streamText ends the stream with an abort part', async (t) => {
  const provider = new AbortRecordingProvider(true);
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({ plugins: [new AiSdkPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(abortableStreamWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });
    await assertEventually(t, (tt) => {
      tt.true(provider.started);
    }, 10_000);
    await handle.signal(abortRequest);
    const types = await handle.result();
    t.is(types.at(-1), 'abort');
    t.false(types.includes('error'));
    await assertEventually(t, (tt) => {
      tt.true(provider.providerAborted);
    }, 10_000);
  });
});

test.serial('An abort during embedMany cancels the Activity in a new execution', async (t) => {
  const provider = new AbortRecordingProvider(true);
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({ plugins: [new AiSdkPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(abortDuringEmbedWorkflow);
    await assertEventually(t, (tt) => {
      tt.true(provider.started);
    }, 10_000);
    await handle.signal(abortRequest);
    t.is(await handle.result(), 'error:AbortError|true');
    const { events } = await handle.fetchHistory();
    t.true(
      events!.some((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_CANCEL_REQUESTED)
    );
  });
});

test('model Activities pass the Activity cancellation signal to the model', async (t) => {
  const provider = new AbortRecordingProvider(false);
  type Activity = (args: unknown) => Promise<unknown>;
  const activities = createActivities(provider) as Record<string, Activity>;
  const env = new MockActivityEnvironment();
  const prompt: LanguageModelV4CallOptions['prompt'] = [
    { role: 'user', content: [{ type: 'text', text: 'hi' }] },
  ];
  // Histories recorded before the fix hold `abortSignal: {}` in the Activity input.
  const options = { prompt, abortSignal: {} } as unknown as LanguageModelV4CallOptions;

  await env.run(activities.invokeModel!, { modelId: 'm', options });
  await env.run(activities.invokeEmbeddingModel!, { modelId: 'e', values: ['a'] });

  t.is(provider.signals.length, 2);
  for (const signal of provider.signals) t.is(signal, env.context.cancellationSignal);
  env.cancel();
  t.true(env.context.cancellationSignal.aborted);
});
