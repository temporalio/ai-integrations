/**
 * Workflow interceptor module `LangGraphPlugin` adds to every Workflow bundle.
 * It marks each run as running under the plugin and drops the run's state
 * when the Workflow is disposed.
 *
 * @module
 * @internal
 */

import { workflowInfo } from '@temporalio/workflow';
import type { WorkflowInterceptorsFactory } from '@temporalio/workflow';
import { disposeRunState, runState } from './run-state';

export const interceptors: WorkflowInterceptorsFactory = () => {
  const { runId } = workflowInfo();
  runState().pluginActive = true;
  return {
    internals: [
      {
        dispose(input, next) {
          disposeRunState(runId);
          next(input);
        },
      },
    ],
  };
};
