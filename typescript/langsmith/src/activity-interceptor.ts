/**
 * Activity-side and Nexus-handler-side LangSmith interceptors. Both run in the
 * real Node worker process and install the reconstructed run via `withRunTree`
 * so a user's unchanged body `traceable` calls nest under it.
 *
 * @module
 */

import { isTracingEnabled } from 'langsmith';
import { RunTree } from 'langsmith/run_trees';
import { withRunTree } from 'langsmith/traceable';
import type { Context as ActivityContext } from '@temporalio/activity';
import type { ActivityInboundCallsInterceptorFactory, NexusInboundCallsInterceptor } from '@temporalio/worker';

import { HEADER_KEY, decodeContextString, readContextHeader, type LangSmithTraceContext } from './propagation';
import {
  RUN_TYPE,
  buildRunTree,
  runActivityRunName,
  runCancelNexusHandlerRunName,
  runStartNexusHandlerRunName,
  runTreeFromContext,
  wrapWithRun,
} from './run-tree';
import type { EmitterConfig } from './sinks';
import { bindReplicaCredentials, destination } from './routing';
import { withRunTreeDefaults } from './worker-run-tree';

/**
 * Reconstruct the propagated parent run from a LangSmith trace context, wired to
 * the plugin-configured client so descendant runs emit to the right place. The
 * returned run is the *parent* — never posted itself — used only to parent
 * the operation run (or directly as ambient when `addTemporalRuns` is off).
 */
function reconstructParentRun(config: EmitterConfig, ctx: LangSmithTraceContext | undefined): RunTree | undefined {
  const parsed = runTreeFromContext(ctx);
  if (!parsed) {
    return undefined;
  }
  return new RunTree({
    name: parsed.name || 'parent',
    run_type: parsed.run_type || RUN_TYPE.CHAIN,
    id: parsed.id,
    trace_id: parsed.trace_id,
    dotted_order: parsed.dotted_order,
    start_time: parsed.start_time,
    parent_run_id: parsed.parent_run_id,
    ...destination(parsed, { project_name: config.projectName, address: config.address }),
    replicas: bindReplicaCredentials(parsed.replicas, config.replicas),
    distributedParentId: parsed.distributedParentId,
    client: config.client,
    // Force-enable so nested body `traceable` runs emit; tracing gate is `isTracingEnabled()`.
    tracingEnabled: true,
  });
}

/**
 * Run a handler under `parent`. With `addTemporalRuns`, wrap it in the operation run from `build`, emitted unless
 * LangSmith turned tracing off for it (an env it can't address). Shared by activity and Nexus handlers.
 */
async function traceHandler<T>(
  config: EmitterConfig,
  parent: RunTree | undefined,
  build: () => RunTree,
  fn: () => Promise<T>
): Promise<T> {
  // A propagated parent keeps tracing on, as LangSmith does; the env only gates traces that start here.
  if (!parent && !isTracingEnabled()) {
    return fn();
  }
  if (!config.addTemporalRuns) {
    // Without a parent, native roots fall back to the plugin's routing defaults.
    return parent ? withRunTree(parent, fn) : withRunTreeDefaults(config, fn);
  }
  const run = build();
  if (!run.tracingEnabled) {
    return withRunTree(run, fn);
  }
  await run.postRun();
  return wrapWithRun(run, () => withRunTree(run, fn));
}

/**
 * Build the activity-inbound interceptor factory. The worker calls the factory
 * once per activity with that activity's {@link ActivityContext}, from which we
 * read the activity type for the run name.
 */
export function createActivityInboundInterceptor(config: EmitterConfig): ActivityInboundCallsInterceptorFactory {
  return (ctx: ActivityContext) => ({
    async execute(input, next) {
      const parent = reconstructParentRun(config, readContextHeader(input.headers));
      const build = () => {
        const { activityType, activityId, workflowExecution } = ctx.info;
        return buildRunTree(config, {
          name: runActivityRunName(activityType),
          runType: RUN_TYPE.TOOL,
          parent,
          metadata: {
            temporalWorkflowID: workflowExecution?.workflowId ?? '',
            temporalRunID: workflowExecution?.runId ?? '',
            temporalActivityID: activityId,
          },
        });
      };
      return traceHandler(config, parent, build, () => next(input));
    },
  });
}

/**
 * Build the Nexus-handler interceptor. `startOperation` opens a
 * `RunStartNexusOperationHandler:` run (so workflows the handler starts nest
 * under it); `cancelOperation` opens `RunCancelNexusOperationHandler:`.
 */
export function createNexusInboundInterceptor(config: EmitterConfig): NexusInboundCallsInterceptor {
  const handle = async <I extends { ctx: { service: string; operation: string; headers: Record<string, string> } }, O>(
    input: I,
    next: (input: I) => Promise<O>,
    nameOf: (service: string, operation: string) => string
  ): Promise<O> => {
    const parent = reconstructParentRun(config, decodeContextString(input.ctx.headers[HEADER_KEY]));
    const build = () =>
      buildRunTree(config, { name: nameOf(input.ctx.service, input.ctx.operation), runType: RUN_TYPE.TOOL, parent });
    return traceHandler(config, parent, build, () => next(input));
  };

  return {
    startOperation(input, next) {
      return handle(input, next, runStartNexusHandlerRunName);
    },
    cancelOperation(input, next) {
      return handle(input, next, runCancelNexusHandlerRunName);
    },
  };
}
