import { StructuredOutputError } from '@strands-agents/sdk';
import type { TestFn } from 'ava';
import { WorkflowFailedError } from '@temporalio/client';
import { ApplicationFailure } from '@temporalio/common';
import { temporal } from '@temporalio/proto';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, createBaseBundlerOptions, helpers } from '@temporalio/test-helpers';
import { STRANDS_MALFORMED_INTERRUPT_TYPE, StrandsPlugin } from '..';
import { confirmState, confirmThings, deleteState, deleteThing, interruptError } from './activities/strands';
import { setupTestEnv, StubModel, textTurn, toolCallTurn } from './helpers/strands';
import { approveSignal, failActivities, interruptAgent } from './workflows/strands';

const test = anyTest as TestFn<BaseContext>;

setupTestEnv(test);

interface ActivityFailureRecord {
  message?: string | null;
  type?: string | null;
  nonRetryable?: boolean | null;
}

/** Each failed activity's message, type and non-retryable flag in a run's history. */
function activityFailures(history: temporal.api.history.v1.IHistory): ActivityFailureRecord[] {
  return (history.events ?? []).flatMap((e) => {
    const failure = e.activityTaskFailedEventAttributes?.failure;
    const info = failure?.applicationFailureInfo;
    return failure ? [{ message: failure.message, type: info?.type, nonRetryable: info?.nonRetryable }] : [];
  });
}

const CALLER_FAILURE_CONVERTER = require.resolve('./encoding-failure-converter');

const BENIGN = temporal.api.enums.v1.ApplicationErrorCategory.APPLICATION_ERROR_CATEGORY_BENIGN;

test('activity-tool interrupt pauses the agent and resumes', async (t) => {
  deleteState.calls = 0;
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          test: () => new StubModel([toolCallTurn('deleteThing', 'call_1', { name: 'foo' }), textTurn('done')]),
        },
      }),
    ],
    activities: { deleteThing },
  });

  const history = await worker.runUntil(async () => {
    const handle = await startWorkflow(interruptAgent, {
      args: ['delete foo'],
      workflowExecutionTimeout: '30 seconds',
    });
    await handle.signal(approveSignal, 'approve');
    t.is(await handle.result(), 'done');
    return handle.fetchHistory();
  });
  // First activity invocation threw the interrupt; second succeeded after resume.
  t.is(deleteState.calls, 2);
  // The interrupt failed the first activity once, with no retry; the second call came from the
  // activity the resumed agent scheduled.
  t.deepEqual(activityFailures(history), [
    { message: 'interrupt:approval', type: 'StrandsInterrupt', nonRetryable: true },
  ]);
  // That failure came from the activity's first attempt, is benign, and keeps the stack of the error
  // it threw.
  const failed = history.events?.find((e) => e.activityTaskFailedEventAttributes)?.activityTaskFailedEventAttributes;
  const startedId = failed?.startedEventId?.toString();
  const started = history.events?.find((e) => e.eventId?.toString() === startedId);
  t.is(started?.activityTaskStartedEventAttributes?.attempt, 1);
  t.is(failed?.failure?.applicationFailureInfo?.category, BENIGN);
  t.regex(failed?.failure?.stackTrace ?? '', /interruptError/);
});

test('an activity interrupt with two entries pauses the agent once for each', async (t) => {
  confirmState.calls = 0;
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          test: () => new StubModel([toolCallTurn('confirmThings', 'call_1', { name: 'foo' }), textTurn('done')]),
        },
      }),
    ],
    activities: { confirmThings },
  });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(interruptAgent, {
      args: ['confirm foo', 'confirmThings'],
      workflowExecutionTimeout: '30 seconds',
    });
    await handle.signal(approveSignal, 'yes');
    await handle.signal(approveSignal, 'yes');
    t.is(await handle.result(), 'done');
  });
  // The first call paused at the approval, the second at the confirmation, and the third had both
  // responses.
  t.is(confirmState.calls, 3);
});

test("a worker's own failure converter still applies with the plugin", async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  let inboundCalls = 0;
  const worker = await createWorker({
    // The worker bundles the workflows itself, so the bundle's failure converter comes from the
    // worker's options and the plugins, as it does with `workflowsPath`.
    workflowBundle: undefined,
    workflowsPath: require.resolve('./workflows/strands'),
    bundlerOptions: createBaseBundlerOptions(),
    dataConverter: { failureConverterPath: CALLER_FAILURE_CONVERTER },
    plugins: [new StrandsPlugin()],
    // The worker's own interceptors still run with the plugin's.
    interceptors: {
      activityInbound: [
        () => ({
          execute: (input, next) => {
            inboundCalls += 1;
            return next(input);
          },
        }),
      ],
    },
    activities: {
      async failingActivity(message: string): Promise<never> {
        throw message === 'interrupt' ? interruptError('foo') : new StructuredOutputError('no output');
      },
    },
  });
  const history = await worker.runUntil(async () => {
    const handle = await startWorkflow(failActivities, { args: [['interrupt', 'terminal']] });
    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    // Workflow code decoded each activity failure as `TemporalActivityTool` expects.
    t.deepEqual(err?.cause instanceof ApplicationFailure && err.cause.details, [
      {
        type: 'StrandsInterrupt',
        nonRetryable: true,
        details: [{ id: 'delete:foo', name: 'approval', reason: 'delete foo?', source: 'tool' }],
      },
      { type: 'StructuredOutputError', nonRetryable: true, details: [] },
    ]);
    return handle.fetchHistory();
  });
  // The caller's converter encoded every failure, and the plugin made the Strands errors
  // non-retryable.
  t.deepEqual(activityFailures(history), [
    { message: 'Encoded failure', type: 'StrandsInterrupt', nonRetryable: true },
    { message: 'Encoded failure', type: 'StructuredOutputError', nonRetryable: true },
  ]);
  const workflowFailure = history.events?.find((e) => e.workflowExecutionFailedEventAttributes)
    ?.workflowExecutionFailedEventAttributes?.failure;
  t.is(workflowFailure?.message, 'Encoded failure');
  t.is(inboundCalls, 2);
});

test('a malformed activity interrupt fails without retries', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const malformedInterrupts: Record<string, unknown[]> = {
    'no toJSON': [{ toJSON: () => ({ name: 'approval' }) }, {}],
    'plain entry': [{ name: 'approval', reason: 'delete foo?' }],
    'repeated name': [{ toJSON: () => ({ name: 'approval' }) }, { toJSON: () => ({ name: 'approval' }) }],
    'BigInt reason': [{ toJSON: () => ({ name: 'approval', reason: BigInt(1) }) }],
    'hole only': new Array<unknown>(1),
    // eslint-disable-next-line no-sparse-arrays
    'hole between': [{ toJSON: () => ({ name: 'approval' }) }, , { toJSON: () => ({ name: 'confirmation' }) }],
  };
  const worker = await createWorker({
    plugins: [new StrandsPlugin()],
    activities: {
      async failingActivity(message: string): Promise<never> {
        throw Object.assign(new Error('interrupt'), { interrupts: malformedInterrupts[message] });
      },
    },
  });
  const history = await worker.runUntil(async () => {
    const handle = await startWorkflow(failActivities, { args: [Object.keys(malformedInterrupts)] });
    await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    return handle.fetchHistory();
  });
  const malformed = {
    message:
      'malformed interrupt (interrupt): each entry needs a toJSON() that returns an object with a unique string name, and JSON-encodable values',
    type: STRANDS_MALFORMED_INTERRUPT_TYPE,
    nonRetryable: true,
  };
  t.deepEqual(
    activityFailures(history),
    Object.keys(malformedInterrupts).map(() => malformed)
  );
});
