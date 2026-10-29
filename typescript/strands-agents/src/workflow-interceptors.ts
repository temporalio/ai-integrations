import type { WorkflowInterceptorsFactory } from '@temporalio/workflow';
import { rewriteTerminalError } from './strands-errors';

async function rethrowTerminalError<T>(run: () => Promise<T>): Promise<T> {
  try {
    return await run();
  } catch (err) {
    throw rewriteTerminalError(err);
  }
}

/**
 * Workflow interceptors that {@link StrandsPlugin} adds to workflow bundles. A
 * Strands terminal model error that workflow code, or a signal or update
 * handler, lets escape becomes a typed, non-retryable failure. It then fails the
 * run, or rejects the update, instead of failing the workflow task on every
 * attempt.
 */
// ts-prune-ignore-next (loaded by path from workflowInterceptorModules)
export const interceptors: WorkflowInterceptorsFactory = () => ({
  inbound: [
    {
      execute: (input, next) => rethrowTerminalError(() => next(input)),
      handleSignal: (input, next) => rethrowTerminalError(() => next(input)),
      handleUpdate: (input, next) => rethrowTerminalError(() => next(input)),
    },
  ],
});
