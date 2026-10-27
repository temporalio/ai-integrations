import type { TestFn } from 'ava';
import { ApplicationFailure, type RetryPolicy } from '@temporalio/common';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, helpers, TestWorkflowEnvironment } from '@temporalio/test-helpers';
import { StrandsPlugin } from '..';
import { startLocalWithRetry } from './helpers/environment';
import { failFrozen, failTwice, lastFailureOfRetry } from './workflows/strands';

// Its own file, so the second workflow bundle this test builds lives in its own test process and
// does not add to the memory the main suite's concurrent workers use.
const test = anyTest as TestFn<BaseContext>;

test.before(async (t) => {
  t.context = { env: await startLocalWithRetry(() => TestWorkflowEnvironment.createLocal()) } as BaseContext;
});

test.after.always(async (t) => {
  await t.context.env?.teardown();
});

test("a worker's own failure converter still applies with the plugin", async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  // The worker bundles the workflows itself, so the bundle gets its failure converter.
  const worker = await createWorker({
    workflowsPath: require.resolve('./workflows/strands'),
    dataConverter: { failureConverterPath: require.resolve('./encoding-failure-converter') },
    plugins: [new StrandsPlugin()],
    activities: {
      async failingActivity(message: string): Promise<never> {
        throw ApplicationFailure.nonRetryable(`activity failed: ${message}`);
      },
    },
  });
  const [history, frozenHistory, lastFailure] = await worker.runUntil(async () => {
    const handle = await startWorkflow(failTwice, { args: ['workflow failed'] });
    await t.throwsAsync(handle.result());
    // A frozen failure's `cause` cannot be assigned; Strands' rewrite of it must not fail the
    // workflow task, which would leave the run open until it times out.
    const frozen = await startWorkflow(failFrozen, {
      args: ['frozen failure'],
      workflowExecutionTimeout: '30 seconds',
    });
    await t.throwsAsync(frozen.result());
    // The SDK decodes a retried run's last failure before the run starts; it still gets the
    // caller's converter.
    const retry: RetryPolicy = { maximumAttempts: 2, initialInterval: '10 milliseconds' };
    const retried = await startWorkflow(lastFailureOfRetry, { retry });
    return [await handle.fetchHistory(), await frozen.fetchHistory(), await retried.result()] as const;
  });

  // Both failures were stored as the caller's converter writes them, and Strands still marked the
  // workflow failure's StructuredOutputError cause as non-retryable.
  const activityFailure = history.events?.find((e) => e.activityTaskFailedEventAttributes)
    ?.activityTaskFailedEventAttributes?.failure;
  const workflowFailure = history.events?.find((e) => e.workflowExecutionFailedEventAttributes)
    ?.workflowExecutionFailedEventAttributes?.failure;
  const cause = workflowFailure?.cause?.applicationFailureInfo;
  t.deepEqual(
    [activityFailure?.message, workflowFailure?.message, cause?.type, cause?.nonRetryable, lastFailure],
    ['Encoded failure', 'Encoded failure', 'StructuredOutputError', true, 'decoded by the caller']
  );
  // The frozen failure was stored the same way, with its cause rewritten.
  const frozenFailure = frozenHistory.events?.find((e) => e.workflowExecutionFailedEventAttributes)
    ?.workflowExecutionFailedEventAttributes?.failure;
  const frozenCause = frozenFailure?.cause?.applicationFailureInfo;
  t.deepEqual(
    [frozenFailure?.message, frozenCause?.type, frozenCause?.nonRetryable],
    ['Encoded failure', 'StructuredOutputError', true]
  );
});
