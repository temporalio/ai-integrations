/**
 * Client-side LangSmith interceptor. Each call runs inside a marker run that ends when the call returns. Execution ops
 * (`start`, `signalWithStart`, `startUpdateWithStart`) propagate the **ambient** context (the remote run is a
 * sibling); messaging ops (`signal`, `query`, `startUpdate`) propagate the **marker** context (the remote handler
 * nests under it).
 *
 * @module
 */

import { isTracingEnabled } from 'langsmith';
import { getCurrentRunTree } from 'langsmith/traceable';
import type { WorkflowClientInterceptor, WorkflowStartInput } from '@temporalio/client';

import { withContextHeader, type LangSmithTraceContext } from './propagation';
import {
  RUN_TYPE,
  buildRunTree,
  queryWorkflowRunName,
  runHeaders,
  signalWithStartRunName,
  signalWorkflowRunName,
  startUpdateWithStartRunName,
  startWorkflowRunName,
  startWorkflowUpdateRunName,
  wrapWithRun,
} from './run-tree';
import type { EmitterConfig } from './sinks';

type NextFn<I, O> = (input: I) => Promise<O>;
type Headers = WorkflowStartInput['headers'];

export function createClientInterceptor(config: EmitterConfig): WorkflowClientInterceptor {
  /** Run `call` inside a marker run, passing it the context to propagate: the marker's, or the ambient one when `peer`. */
  const traced = async <O>(
    name: string,
    peer: boolean,
    call: (context: LangSmithTraceContext | undefined) => Promise<O>
  ): Promise<O> => {
    if (!isTracingEnabled()) {
      return call(undefined);
    }
    const ambient = getCurrentRunTree(true);
    if (!config.addTemporalRuns) {
      return call(runHeaders(ambient));
    }
    const marker = buildRunTree(config, { name, runType: RUN_TYPE.CHAIN, parent: ambient });
    const context = runHeaders(peer ? ambient : marker);
    if (!marker.tracingEnabled) {
      return call(context);
    }
    await marker.postRun();
    return wrapWithRun(marker, () => call(context));
  };

  const withHeaders = <I extends { headers: Headers }, O>(input: I, next: NextFn<I, O>, name: string, peer: boolean) =>
    traced(name, peer, (context) => next({ ...input, headers: withContextHeader(input.headers, context) }));

  return {
    start(input, next) {
      return withHeaders(input, next, startWorkflowRunName(input.workflowType), true);
    },
    startWithDetails(input, next) {
      return withHeaders(input, next, startWorkflowRunName(input.workflowType), true);
    },
    signal(input, next) {
      return withHeaders(input, next, signalWorkflowRunName(input.signalName), false);
    },
    signalWithStart(input, next) {
      return withHeaders(input, next, signalWithStartRunName(input.workflowType), true);
    },
    query(input, next) {
      return withHeaders(input, next, queryWorkflowRunName(input.queryType), false);
    },
    startUpdate(input, next) {
      return withHeaders(input, next, startWorkflowUpdateRunName(input.updateName), false);
    },
    startUpdateWithStart(input, next) {
      return traced(startUpdateWithStartRunName(input.workflowType), true, (context) =>
        next({
          ...input,
          workflowStartHeaders: withContextHeader(input.workflowStartHeaders, context),
          updateHeaders: withContextHeader(input.updateHeaders, context),
        })
      );
    },
  };
}
