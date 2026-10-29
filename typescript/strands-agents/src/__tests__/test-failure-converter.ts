import type { TestFn } from 'ava';
import { StructuredOutputError } from '@strands-agents/sdk';
import { WorkflowFailedError } from '@temporalio/client';
import { ApplicationFailure } from '@temporalio/common';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, helpers, TestWorkflowEnvironment } from '@temporalio/test-helpers';
import { StrandsPlugin } from '..';
import { startLocalWithRetry } from './helpers/environment';
import { failActivities } from './workflows/strands';

// Its own file, so the second workflow bundle this test builds lives in its own test process and
// does not add to the memory the main suite's concurrent workers use.
const test = anyTest as TestFn<BaseContext>;

const failureConverterPath = require.resolve('./encoding-failure-converter');

test.before(async (t) => {
  t.context = {
    env: await startLocalWithRetry(() =>
      TestWorkflowEnvironment.createLocal({ client: { dataConverter: { failureConverterPath } } })
    ),
  } as BaseContext;
});

test.after.always(async (t) => {
  await t.context.env?.teardown();
});

const interrupt = { id: 'delete:foo', name: 'approval', reason: 'delete foo?', source: 'tool' };

test("a worker's own failure converter still applies with the plugin", async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  // The worker bundles the workflows itself, so the bundle gets its failure converter.
  const worker = await createWorker({
    workflowsPath: require.resolve('./workflows/strands'),
    dataConverter: { failureConverterPath },
    plugins: [new StrandsPlugin()],
    activities: {
      async failingActivity(message: string): Promise<never> {
        if (message === 'interrupt') {
          throw Object.assign(new Error('interrupt'), { interrupts: [{ toJSON: () => interrupt }] });
        }
        throw new StructuredOutputError('no output');
      },
    },
  });
  const history = await worker.runUntil(async () => {
    const handle = await startWorkflow(failActivities, { args: [['interrupt', 'terminal']] });
    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    // Workflow code decoded each activity failure as `TemporalActivityTool` expects.
    t.deepEqual(err?.cause instanceof ApplicationFailure && err.cause.details, [
      ['StrandsInterrupt', true, [interrupt]],
      ['StructuredOutputError', true, []],
    ]);
    return handle.fetchHistory();
  });

  // The caller's converter encoded every failure, and the plugin made the Strands errors
  // non-retryable.
  const events = history.events ?? [];
  const activityFailures = events.flatMap((e) => {
    const failure = e.activityTaskFailedEventAttributes?.failure;
    const info = failure?.applicationFailureInfo;
    return failure ? [[failure.message, info?.type, info?.nonRetryable]] : [];
  });
  t.deepEqual(activityFailures, [
    ['Encoded failure', 'StrandsInterrupt', true],
    ['Encoded failure', 'StructuredOutputError', true],
  ]);
  const workflowFailure = events.find((e) => e.workflowExecutionFailedEventAttributes)
    ?.workflowExecutionFailedEventAttributes?.failure;
  t.is(workflowFailure?.message, 'Encoded failure');
});
