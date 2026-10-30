/**
 * Replay-safe LangSmith run construction for workflow code, plus the pure
 * run-name builders shared by every interceptor.
 *
 * Inside the workflow isolate LangSmith's default `RunTree` is unsafe on two
 * counts that the workflow lifecycle overrides fix:
 *
 *  1. **Non-deterministic ids.** Its default random-tailed uuid7 would mint a
 *     fresh id on every replay, flooding the backend with duplicates.
 *  2. **Network I/O on replay.** `postRun` / `patchRun` would re-POST runs on
 *     every history replay.
 *
 * Internal: every export here is consumed only by sibling modules in this
 * package; none is reachable through a package entry point.
 *
 * @module
 * @internal
 */

import { RunTree, type RunTreeConfig } from 'langsmith/run_trees';
import { ApplicationFailure, ApplicationFailureCategory } from '@temporalio/common';

import { scrubSensitive, type LangSmithTraceContext } from './propagation';
import type { EmitterConfig, SerializedRun } from './sinks';
import { destination, normalizeReplicaHeaders, replicaHeaders, replicaRouting } from './routing';

/** LangSmith run-type constants used for Temporal-operation runs. */
export const RUN_TYPE = {
  /** Workflow / child-workflow / scheduling / handler spans. */
  CHAIN: 'chain',
  /** Activity-body execution spans. */
  TOOL: 'tool',
} as const;

/** `StartWorkflow:<type>` — client/parent-side workflow-start marker. */
export const startWorkflowRunName = (workflowType: string): string => `StartWorkflow:${workflowType}`;
/** `RunWorkflow:<type>` — the workflow-execution span (inbound). */
export const runWorkflowRunName = (workflowType: string): string => `RunWorkflow:${workflowType}`;
/** `StartActivity:<name>` — workflow-side activity-schedule marker. */
export const startActivityRunName = (activityType: string): string => `StartActivity:${activityType}`;
/** `RunActivity:<name>` — the activity-execution span (activity inbound). */
export const runActivityRunName = (activityType: string): string => `RunActivity:${activityType}`;
/** `StartChildWorkflow:<type>` — workflow-side child-start marker. */
export const startChildWorkflowRunName = (workflowType: string): string => `StartChildWorkflow:${workflowType}`;
/** `StartNexusOperation:<service>/<op>` — workflow-side Nexus-start marker. */
export const startNexusOperationRunName = (service: string, operation: string): string =>
  `StartNexusOperation:${service}/${operation}`;
/** `RunStartNexusOperationHandler:<service>/<op>` — Nexus start-handler span. */
export const runStartNexusHandlerRunName = (service: string, operation: string): string =>
  `RunStartNexusOperationHandler:${service}/${operation}`;
/** `RunCancelNexusOperationHandler:<service>/<op>` — Nexus cancel-handler span. */
export const runCancelNexusHandlerRunName = (service: string, operation: string): string =>
  `RunCancelNexusOperationHandler:${service}/${operation}`;
/** `HandleSignal:<name>` — inbound signal handler span. */
export const handleSignalRunName = (signalName: string): string => `HandleSignal:${signalName}`;
/** `HandleQuery:<name>` — inbound query handler span. */
export const handleQueryRunName = (queryName: string): string => `HandleQuery:${queryName}`;
/** `HandleUpdate:<name>` — inbound update handler span. */
export const handleUpdateRunName = (updateName: string): string => `HandleUpdate:${updateName}`;
/** `ValidateUpdate:<name>` — inbound update validator span. */
export const validateUpdateRunName = (updateName: string): string => `ValidateUpdate:${updateName}`;
/** `SignalWorkflow:<name>` — client/outbound signal marker. */
export const signalWorkflowRunName = (signalName: string): string => `SignalWorkflow:${signalName}`;
/** `SignalChildWorkflow:<name>` — workflow-side signal-child marker. */
export const signalChildWorkflowRunName = (signalName: string): string => `SignalChildWorkflow:${signalName}`;
/** `SignalExternalWorkflow:<name>` — workflow-side signal-external marker. */
export const signalExternalWorkflowRunName = (signalName: string): string => `SignalExternalWorkflow:${signalName}`;
/** `SignalWithStartWorkflow:<type>` — client signal-with-start marker. */
export const signalWithStartRunName = (workflowType: string): string => `SignalWithStartWorkflow:${workflowType}`;
/** `QueryWorkflow:<name>` — client query marker. */
export const queryWorkflowRunName = (queryName: string): string => `QueryWorkflow:${queryName}`;
/** `StartWorkflowUpdate:<name>` — client update-start marker. */
export const startWorkflowUpdateRunName = (updateName: string): string => `StartWorkflowUpdate:${updateName}`;
/** `StartUpdateWithStartWorkflow:<name>` — client update-with-start marker. */
export const startUpdateWithStartRunName = (updateName: string): string => `StartUpdateWithStartWorkflow:${updateName}`;

/** Flatten a {@link RunTree} into the plain wire shape consumed by the Sink. */
export function serializeRun(run: RunTree): SerializedRun {
  const extra = run.extra ? { ...run.extra } : undefined;
  if (extra && extra.metadata && typeof extra.metadata === 'object') {
    extra.metadata = scrubSensitive(extra.metadata as Record<string, unknown>);
  }
  return {
    id: run.id,
    trace_id: run.trace_id,
    dotted_order: run.dotted_order,
    parent_run_id: run.parent_run?.id ?? run.parent_run_id,
    name: run.name,
    run_type: run.run_type,
    start_time: run.start_time,
    end_time: run.end_time,
    inputs: run.inputs as Record<string, unknown> | undefined,
    outputs: run.outputs as Record<string, unknown> | undefined,
    error: run.error,
    extra,
    tags: run.tags,
    project_name: run.project_name,
    address: run.address,
    replicas: replicaRouting(run.replicas),
    distributedParentId: run.distributedParentId,
    events: run.events as Record<string, unknown>[] | undefined,
  };
}

/**
 * Render an error into the LangSmith `error` string, honoring Temporal's
 * benign-failure category.
 *
 * A `BENIGN`-category `ApplicationFailure` is an expected, non-alarming outcome
 * (e.g. a control-flow signal), so it leaves the run's `error` unset. All other
 * failures produce `"<type>: <message>"`.
 */
export function describeError(err: unknown): string | undefined {
  if (err instanceof ApplicationFailure) {
    if (err.category === ApplicationFailureCategory.BENIGN) {
      return undefined;
    }
    return `${err.type ?? 'ApplicationError'}: ${err.message}`;
  }
  if (err instanceof Error) {
    return `${err.name}: ${err.message}`;
  }
  return String(err);
}

/** Coerce an arbitrary operation result into a LangSmith outputs object. */
export function asOutputs(result: unknown): Record<string, unknown> {
  if (result !== null && typeof result === 'object' && !Array.isArray(result)) {
    return result as Record<string, unknown>;
  }
  return { result };
}

/** Runs suppressed because the worker has no usable destination. */
const untracedRuns = new WeakSet<RunTree>();

export function isUntraced(run: RunTree): boolean {
  return untracedRuns.has(run);
}

export function suppressRun(run: RunTree): void {
  untracedRuns.add(run);
}

/**
 * The trace context to propagate for `run`, or `undefined` when no LangSmith run exists to parent under: no run, a
 * run with no destination, or a run LangSmith won't trace (`tracingEnabled: false`). LangSmith turns tracing off for
 * a run built in an env it can't use, such as a client marker.
 */
export function runHeaders(run: RunTree | undefined): LangSmithTraceContext | undefined {
  if (!run || run.tracingEnabled === false || isUntraced(run)) {
    return undefined;
  }
  const headers = run.toHeaders() as LangSmithTraceContext;
  const baggage = headers.baggage.split(',').filter(
    (item) => !item.startsWith('langsmith-replicas=') && !item.startsWith('langsmith-metadata=')
  );
  const metadata = scrubSensitive(run.extra?.metadata as Record<string, unknown> | undefined);
  if (metadata && Object.keys(metadata).length > 0) {
    baggage.push(`langsmith-metadata=${encodeURIComponent(JSON.stringify(metadata))}`);
  }
  const replicas = replicaRouting(run.replicas);
  if (replicas !== undefined) baggage.push(`langsmith-replicas=${encodeURIComponent(JSON.stringify(replicaHeaders(replicas)))}`);
  headers.baggage = baggage.filter(Boolean).join(',');
  return headers;
}

/** Emit a short-lived marker run, unless LangSmith turned tracing off for it (an env it can't address). */
export async function emitMarkerRun(marker: RunTree): Promise<void> {
  if (!marker.tracingEnabled) {
    return;
  }
  await marker.postRun();
  await marker.end({});
  await marker.patchRun();
}

/**
 * A run's destination: a LangSmith project, or an agent address (LangSmith's `RunTreeConfig.address`, in the
 * installed version's own form).
 */
export type Destination = Pick<RunTreeConfig, 'project_name' | 'address'>;

/**
 * Reconstruct a propagated parent {@link RunTree} from a trace context; never throws. LangSmith's `RunTree.fromHeaders`
 * lets the context's destination outrank `fallback`, so `fallback` applies only when the context names none.
 */
export function runTreeFromContext(
  ctx: LangSmithTraceContext | undefined,
  fallback?: Destination
): RunTree | undefined {
  if (!ctx) {
    return undefined;
  }
  try {
    const run = RunTree.fromHeaders({ ...ctx, baggage: normalizeReplicaHeaders(ctx.baggage ?? '') }, fallback);
    if (!run) return undefined;
    run.replicas = replicaRouting(run.replicas);
    return run;
  } catch {
    return undefined;
  }
}

interface RunTreeParams {
  name: string;
  runType: string;
  parent: RunTree | undefined;
  inputs?: Record<string, unknown>;
}

/**
 * Build an emitter-side {@link RunTree} (client / activity / Nexus); force-enables
 * tracing so nested body `traceable` runs emit — tracing gate is `isTracingEnabled()`.
 */
export function buildRunTree(config: EmitterConfig, params: RunTreeParams): RunTree {
  return new RunTree({
    name: params.name,
    run_type: params.runType,
    inputs: params.inputs ?? {},
    parent_run: params.parent,
    client: config.client,
    ...destination(params.parent, { project_name: config.projectName, address: config.address }),
    replicas: params.parent?.replicas ?? config.replicas,
    tags: config.tags,
    extra: { metadata: scrubSensitive(config.metadata) ?? {} },
    tracingEnabled: true,
  });
}

/** An invisible worker-side factory gives unparented traceables the plugin's destination and client. */
export function rootRunTree(config: EmitterConfig): RunTree {
  const factory = buildRunTree(config, { name: 'temporal-routing-parent', runType: RUN_TYPE.CHAIN, parent: undefined });
  factory.createChild = (child) => new RunTree({
    ...child,
    ...destination(child, factory),
    replicas: child.replicas ?? factory.replicas,
    client: config.client,
    tracingEnabled: true,
  });
  return factory;
}
