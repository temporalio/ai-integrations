import type { WorkflowInterceptorsFactory } from '@temporalio/workflow';
import { rewriteTerminalError } from './strands-errors';

async function rethrowTerminalError<I, R>(input: I, next: (input: I) => Promise<R>): Promise<R> {
  try {
    return await next(input);
  } catch (err) {
    throw rewriteTerminalError(err);
  }
}

/**
 * Workflow interceptors that {@link StrandsPlugin} adds to workflow bundles. A
 * Strands terminal model error that workflow code, or a signal or update
 * handler, lets escape becomes a typed, non-retryable failure. Then the run, or
 * the update, fails with it instead of the workflow task failing on every
 * attempt.
 *
 * Interrupts are not rewritten here: the agent loop handles them in workflow
 * code, and one that escapes is a bug that should keep failing the workflow
 * task, which a code fix can recover from, rather than fail the run.
 */
// ts-prune-ignore-next (loaded by path from workflowInterceptorModules)
export const interceptors: WorkflowInterceptorsFactory = () => ({
  inbound: [
    {
      execute: rethrowTerminalError,
      handleSignal: rethrowTerminalError,
      handleUpdate: rethrowTerminalError,
    },
  ],
});
