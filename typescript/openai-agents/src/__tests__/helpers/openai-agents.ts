import { setTracingDisabled } from '@openai/agents-core';
import type { ExecutionContext, TestFn } from 'ava';
import type { WorkflowFailedError, WorkflowHandle } from '@temporalio/client';
import { ApplicationFailure } from '@temporalio/common';
import type { BaseContext } from '@temporalio/test-helpers';
import { Worker } from '@temporalio/worker';
import { DEDICATED_WORKER_FAILURE_TYPE, OpenAIAgentsPlugin } from '../..';
import { FakeModelProvider } from '../stubs/openai-agents';
import { makeTestFunction } from './test-fn';

/** Register the local server and Workflow bundle for one integration test file. */
export function makeOpenAIAgentsTestFunction(): TestFn<BaseContext> {
  // AVA runs one file at a time; keep feature suites small to bound concurrent Workers.
  // Upstream disables agent-SDK tracing under NODE_ENV=test; opt back in for these tests.
  setTracingDisabled(false);
  return makeTestFunction({
    workflowsPath: require.resolve('../workflows/openai-agents'),
    workflowInterceptorModules: [require.resolve('@temporalio/openai-agents/workflow-interceptor')],
    // The plugin prepends the polyfill installer to the webpack entry array.
    // The model provider stub is never invoked at bundle time.
    plugins: [new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) })],
  });
}

/** Replays a Workflow's history with a Worker whose model provider must not be called. */
export async function replay(t: ExecutionContext<BaseContext>, handle: WorkflowHandle): Promise<void> {
  await Worker.runReplayHistory(
    {
      workflowBundle: t.context.workflowBundle,
      plugins: [new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) })],
    },
    await handle.fetchHistory()
  );
}

/** Checks that a run failed because its retried session Activity was refused. */
export function assertSessionRefused(t: ExecutionContext, err: WorkflowFailedError | undefined): void {
  // What leaves the server's block is the type the docs name.
  t.true(err?.cause instanceof ApplicationFailure, String(err?.cause));
  t.is((err?.cause as ApplicationFailure).type, DEDICATED_WORKER_FAILURE_TYPE);
  // Refused by the retried session Activity, not only by an operation timing out.
  let cause: unknown = err?.cause;
  while (cause instanceof Error && !(cause instanceof ApplicationFailure && cause.nonRetryable)) cause = cause.cause;
  t.true(cause instanceof ApplicationFailure, String(cause));
  t.is((cause as ApplicationFailure).type, DEDICATED_WORKER_FAILURE_TYPE);
}
