import type { Message, ModelStreamEvent, StreamOptions } from '@strands-agents/sdk';
import type { TestFn } from 'ava';
import { WorkflowFailedError } from '@temporalio/client';
import { CancelledFailure } from '@temporalio/common';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, helpers } from '@temporalio/test-helpers';
import { ActivityCancellationType } from '@temporalio/workflow';
import { StrandsPlugin } from '..';
import { heartbeatTimeouts, setupTestEnv, StubModel, textTurn } from './helpers/strands';
import { cancellableAgent, cancellableModelCall, helloAgent } from './workflows/strands';

const test = anyTest as TestFn<BaseContext>;

setupTestEnv(test);

for (const streaming of [false, true]) {
  const path = streaming ? 'invokeModelStreaming' : 'invokeModel';
  test(`cancelling the workflow aborts the model call in ${path} and cancels the workflow`, async (t) => {
    t.timeout(60_000);
    const { createWorker, startWorkflow } = helpers(t);
    let receivedSignal: unknown;
    let markStarted!: () => void;
    const started = new Promise<void>((resolve) => (markStarted = resolve));
    // Waits for its cancel signal, then fails as OpenAI's client does: an error named `Error`.
    class AbortingStub extends StubModel {
      override async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
        receivedSignal = options?.cancelSignal;
        const aborted = await new Promise<boolean>((resolve) => {
          options?.cancelSignal?.addEventListener('abort', () => resolve(true));
          markStarted();
        });
        if (aborted) throw new Error('Request was aborted.');
        yield* super.stream(messages, options);
      }
    }

    const worker = await createWorker({
      plugins: [new StrandsPlugin({ models: { test: () => new AbortingStub([]) } })],
    });
    await worker.runUntil(async () => {
      const handle = await startWorkflow(cancellableModelCall, { args: [streaming] });
      await started;
      await handle.cancel();
      const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
      t.true(err!.cause instanceof CancelledFailure, `expected a CancelledFailure, got ${err!.cause}`);
      const { events } = await handle.fetchHistory();
      t.true(events!.some((e) => e.activityTaskCanceledEventAttributes), 'expected ActivityTaskCanceled');
      t.false(events!.some((e) => e.activityTaskFailedEventAttributes), 'expected no ActivityTaskFailed');
      t.true(events!.some((e) => e.workflowExecutionCanceledEventAttributes), 'expected WorkflowExecutionCanceled');
    });
    t.true(receivedSignal instanceof AbortSignal && receivedSignal.aborted, 'expected the model signal to abort');
  });
}

for (const cancellationType of [
  ActivityCancellationType.TRY_CANCEL,
  ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
]) {
  test(`cancelling a TemporalAgent workflow with ${cancellationType} aborts the model call and cancels the workflow`, async (t) => {
    t.timeout(60_000);
    const { createWorker, startWorkflow } = helpers(t);
    let markStarted!: () => void;
    const started = new Promise<void>((resolve) => (markStarted = resolve));
    let markAborted!: () => void;
    const aborted = new Promise<void>((resolve) => (markAborted = resolve));
    // Waits for its cancel signal, then fails as OpenAI's client does: an error named `Error`.
    class AbortingStub extends StubModel {
      // eslint-disable-next-line require-yield
      override async *stream(_messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
        await new Promise<void>((resolve) => {
          options?.cancelSignal?.addEventListener('abort', () => resolve());
          markStarted();
        });
        markAborted();
        throw new Error('Request was aborted.');
      }
    }

    const worker = await createWorker({
      plugins: [new StrandsPlugin({ models: { test: () => new AbortingStub([]) } })],
    });
    await worker.runUntil(async () => {
      const handle = await startWorkflow(cancellableAgent, { args: [cancellationType] });
      await started;
      await handle.cancel();
      const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
      t.true(err!.cause instanceof CancelledFailure, `expected a CancelledFailure, got ${err!.cause}`);
      // The model's signal aborts while the worker runs, not because it shuts down.
      await aborted;
      const { events } = await handle.fetchHistory();
      // The heartbeat timeout set on the agent wins. The config read can't heartbeat, so it gets none.
      t.deepEqual(heartbeatTimeouts(events), [
        ['strandsGetModelConfig', 0],
        ['invokeModel', 5_000],
      ]);
      t.true(events!.some((e) => e.activityTaskCancelRequestedEventAttributes), 'expected ActivityTaskCancelRequested');
      t.false(events!.some((e) => e.workflowTaskFailedEventAttributes), 'expected no WorkflowTaskFailed');
      t.true(events!.some((e) => e.workflowExecutionCanceledEventAttributes), 'expected WorkflowExecutionCanceled');
      if (cancellationType === ActivityCancellationType.WAIT_CANCELLATION_COMPLETED) {
        t.true(events!.some((e) => e.activityTaskCanceledEventAttributes), 'expected ActivityTaskCanceled');
      }
    });
  });

  test(`cancelling a TemporalAgent workflow with ${cancellationType} while the model ignores its signal`, async (t) => {
    t.timeout(60_000);
    const { createWorker, startWorkflow } = helpers(t);
    let markStarted!: () => void;
    const started = new Promise<void>((resolve) => (markStarted = resolve));
    let markAborted!: () => void;
    const aborted = new Promise<void>((resolve) => (markAborted = resolve));
    // Keeps going once its cancel signal aborts, as models that ignore `cancelSignal` do.
    class IgnoringStub extends StubModel {
      override async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
        await new Promise<void>((resolve) => {
          options?.cancelSignal?.addEventListener('abort', () => resolve());
          markStarted();
        });
        markAborted();
        yield* super.stream(messages, options);
      }
    }

    const worker = await createWorker({
      plugins: [new StrandsPlugin({ models: { test: () => new IgnoringStub([textTurn('late result')]) } })],
    });
    await worker.runUntil(async () => {
      const handle = await startWorkflow(cancellableAgent, { args: [cancellationType] });
      await started;
      await handle.cancel();
      if (cancellationType === ActivityCancellationType.WAIT_CANCELLATION_COMPLETED) {
        // The workflow waits for the activity, which completes, so the run completes with its result.
        t.is((await handle.result()).trim(), 'late result');
        const { events } = await handle.fetchHistory();
        t.true(events!.some((e) => e.activityTaskCompletedEventAttributes), 'expected ActivityTaskCompleted');
        t.true(events!.some((e) => e.workflowExecutionCompletedEventAttributes), 'expected WorkflowExecutionCompleted');
        t.false(events!.some((e) => e.workflowExecutionCanceledEventAttributes), 'expected no WorkflowExecutionCanceled');
      } else {
        const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
        t.true(err!.cause instanceof CancelledFailure, `expected a CancelledFailure, got ${err!.cause}`);
        const { events } = await handle.fetchHistory();
        t.true(events!.some((e) => e.workflowExecutionCanceledEventAttributes), 'expected WorkflowExecutionCanceled');
      }
      await aborted;
    });
  });
}

test('a cancel during a model activity cancels the run', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  let modelStarted!: () => void;
  const started = new Promise<void>((resolve) => (modelStarted = resolve));
  // A model call that takes 3 seconds, so the cancel arrives while it runs.
  class SlowModel extends StubModel {
    override async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
      modelStarted();
      await new Promise((resolve) => setTimeout(resolve, 3000));
      yield* super.stream(messages, options);
    }
  }
  const worker = await createWorker({
    plugins: [new StrandsPlugin({ models: { test: () => new SlowModel([textTurn('late')]) } })],
  });
  await worker.runUntil(async () => {
    // Before, the workflow task failed on every attempt and the run stayed open.
    const handle = await startWorkflow(helloAgent, { args: ['hello'], workflowExecutionTimeout: '30 seconds' });
    await started;
    await handle.cancel();
    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.true(err?.cause instanceof CancelledFailure);
  });
});
