import type { Message, ModelStreamEvent, StreamOptions } from '@strands-agents/sdk';
import { ContextWindowOverflowError } from '@strands-agents/sdk';
import type { TestFn } from 'ava';
import { WorkflowFailedError, WorkflowUpdateFailedError } from '@temporalio/client';
import { ActivityFailure, ApplicationFailure } from '@temporalio/common';
import { temporal } from '@temporalio/proto';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, createBaseBundlerOptions, helpers } from '@temporalio/test-helpers';
import { StrandsPlugin } from '..';
import { setupTestEnv, StubModel, textTurn } from './helpers/strands';
import { doneSignal, helloAgent, turnsAgent, turnUpdate, updateTurnsAgent } from './workflows/strands';

const test = anyTest as TestFn<BaseContext>;

setupTestEnv(test);

/** A stub model whose context holds `limit` messages: a longer request overflows, as a provider's would. */
class SmallContextModel extends StubModel {
  constructor(
    private readonly limit: number,
    turns: ModelStreamEvent[][]
  ) {
    super(turns);
  }

  override async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
    if (messages.length > this.limit) {
      throw new ContextWindowOverflowError(`${messages.length} messages do not fit`);
    }
    yield* super.stream(messages, options);
  }
}

/** Convenience: a reply cut off at the model's token limit, so Strands throws `MaxTokensError`. */
function maxTokensTurn(text: string): ModelStreamEvent[] {
  return [...textTurn(text).slice(0, -1), { type: 'modelMessageStopEvent', stopReason: 'maxTokens' }];
}

const BENIGN = temporal.api.enums.v1.ApplicationErrorCategory.APPLICATION_ERROR_CATEGORY_BENIGN;

test('a context overflow in the model activity lets the agent trim the conversation', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  // Also with a caller's converter that decodes the overflow to a `TemporalFailure` that is not an
  // `ApplicationFailure`. The worker bundles the workflows itself so the bundle uses that converter.
  const otherFailureClass = {
    workflowBundle: undefined,
    workflowsPath: require.resolve('./workflows/strands'),
    bundlerOptions: createBaseBundlerOptions(),
    dataConverter: { failureConverterPath: require.resolve('./domain-failure-converter') },
  };
  for (const options of [{}, otherFailureClass]) {
    // Three turns fill the six messages the model holds; the fourth overflows until the agent trims.
    const model = new SmallContextModel(6, ['one', 'two', 'three', 'four'].map(textTurn));
    const worker = await createWorker({ ...options, plugins: [new StrandsPlugin({ models: { test: () => model } })] });
    const history = await worker.runUntil(async () => {
      const handle = await startWorkflow(turnsAgent, {
        args: [['1', '2', '3', '4']],
        workflowExecutionTimeout: '30 seconds',
      });
      t.deepEqual(await handle.result(), ['four', 6]);
      return handle.fetchHistory();
    });
    // The overflow failed the model activity once, as a benign failure.
    const failures = (history.events ?? []).flatMap((e) => {
      const info = e.activityTaskFailedEventAttributes?.failure?.applicationFailureInfo;
      return info ? [[info.type, info.category]] : [];
    });
    t.deepEqual(failures, [['ContextWindowOverflowError', BENIGN]]);
  }
});

test('a context overflow the agent cannot trim fails the run as non-retryable', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  // The model holds no message, so the overflow survives every trim.
  const worker = await createWorker({
    plugins: [new StrandsPlugin({ models: { test: () => new SmallContextModel(0, []) } })],
  });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(helloAgent, { args: ['hello'], workflowExecutionTimeout: '30 seconds' });
    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.deepEqual(err?.cause instanceof ApplicationFailure && [err.cause.type, err.cause.nonRetryable], [
      'ContextWindowOverflowError',
      true,
    ]);
    // The model activity's failure stays in the cause chain.
    const causes: Error[] = [];
    for (let cause: unknown = err?.cause; cause instanceof Error; cause = cause.cause) causes.push(cause);
    t.true(causes.some((cause) => cause instanceof ActivityFailure));
  });
});

test('a Strands terminal error thrown in workflow code fails the run as non-retryable', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [new StrandsPlugin({ models: { test: () => new StubModel([maxTokensTurn('cut off')]) } })],
  });
  await worker.runUntil(async () => {
    // Before, the workflow task failed on every attempt until this timeout.
    const handle = await startWorkflow(helloAgent, { args: ['hello'], workflowExecutionTimeout: '30 seconds' });
    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.deepEqual(err?.cause instanceof ApplicationFailure && [err.cause.type, err.cause.nonRetryable], [
      'MaxTokensError',
      true,
    ]);
  });
});

test('a Strands terminal error in an update handler fails the update', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [new StrandsPlugin({ models: { test: () => new StubModel([maxTokensTurn('cut off')]) } })],
  });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(updateTurnsAgent, { workflowExecutionTimeout: '30 seconds' });
    const err = await t.throwsAsync(handle.executeUpdate(turnUpdate, { args: ['hello'] }), {
      instanceOf: WorkflowUpdateFailedError,
    });
    t.deepEqual(err?.cause instanceof ApplicationFailure && [err.cause.type, err.cause.nonRetryable], [
      'MaxTokensError',
      true,
    ]);
    // The run itself goes on.
    await handle.signal(doneSignal);
    t.is(await handle.result(), 'finished');
  });
});

test('a model activity that fails for good fails the run', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  class RejectingModel extends StubModel {
    // eslint-disable-next-line require-yield
    override async *stream(): AsyncIterable<ModelStreamEvent> {
      throw ApplicationFailure.nonRetryable('the provider rejected the request', 'ProviderError');
    }
  }
  const worker = await createWorker({
    plugins: [new StrandsPlugin({ models: { test: () => new RejectingModel([]) } })],
  });
  await worker.runUntil(async () => {
    // Before, the workflow task failed on every attempt and the run stayed open.
    const handle = await startWorkflow(helloAgent, { args: ['hello'], workflowExecutionTimeout: '30 seconds' });
    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.true(err?.cause instanceof ActivityFailure);
    t.deepEqual(err?.cause?.cause instanceof ApplicationFailure && err.cause.cause.type, 'ProviderError');
  });
});
