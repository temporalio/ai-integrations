/**
 * Replay-safe LangSmith run construction for workflow code, plus the pure
 * run-name builders shared by every interceptor.
 *
 * Inside the workflow isolate LangSmith's default `RunTree` is unsafe on two
 * counts that {@link ReplaySafeRunTree} fixes:
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

import { RunTree, convertToDottedOrderFormat, type RunTreeConfig } from 'langsmith/run_trees';
import type { Client } from 'langsmith';
import { ApplicationFailure, ApplicationFailureCategory } from '@temporalio/common';
import { getRandomStream, proxySinks, workflowInfo } from '@temporalio/workflow';

import { scrubSensitive, type LangSmithTraceContext } from './propagation';
import { LANGSMITH_SINK_NAME, type EmitterConfig, type LangSmithSinks, type SerializedRun } from './sinks';
import { destination, normalizeReplicaHeaders, replicaHeaders, replicaRouting } from './routing';

/** Named random stream for recorded-path run ids; isolated from the workflow's main RNG. */
const LANGSMITH_RUN_TREE_STREAM = 'package-langsmith-tracing';

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

/**
 * True only while replaying genuine history events, when emission must be
 * suppressed. Deliberately `isReplayingHistoryEvents`, NOT `isReplaying`: for
 * live read-only handlers (queries, update validators) `isReplaying` is `true`
 * but `isReplayingHistoryEvents` is `false`, so their runs still emit. Matches
 * how the SDK gates `callDuringReplay: false` sinks.
 */
function isReplaying(): boolean {
  return workflowInfo().unsafe.isReplayingHistoryEvents;
}

let cachedSinks: LangSmithSinks | undefined;
/** Lazily-resolved workflow Sink proxy (must be called inside a workflow). */
function sinks(): LangSmithSinks {
  if (cachedSinks === undefined) {
    cachedSinks = proxySinks<LangSmithSinks>();
  }
  return cachedSinks;
}

/**
 * A no-op LangSmith client. {@link ReplaySafeRunTree} never lets the underlying
 * `RunTree` perform network I/O — emission goes through the Sink — so the
 * client it carries must do nothing and must never read env / open sockets
 * inside the isolate.
 */
const NOOP_CLIENT = {
  createRun: async (): Promise<void> => {},
  updateRun: async (): Promise<void> => {},
} as unknown as Client;

/** Flatten a {@link RunTree} into the plain wire shape consumed by the Sink. */
function serializeRun(run: RunTree): SerializedRun {
  const extra = run.extra ? { ...run.extra } : undefined;
  if (extra && extra.metadata && typeof extra.metadata === 'object') {
    extra.metadata = scrubSensitive(extra.metadata as Record<string, unknown>);
  }
  return {
    id: run.id,
    trace_id: run.trace_id,
    dotted_order: run.dotted_order,
    parent_run_id: run.parent_run_id,
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

/** True for a run the workflow isolate built with no destination (see {@link ReplaySafeRunTree.untraced}). */
function isUntraced(run: RunTree): boolean {
  return (run as { untraced?: boolean }).untraced === true;
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
    // fromHeaders defaults to Date.now(); the recorded order already has the parent's timestamp.
    const stamp = run.dotted_order.split('.').at(-1)!;
    const time = Date.parse(
      `${stamp.slice(0, 4)}-${stamp.slice(4, 6)}-${stamp.slice(6, 8)}T` +
      `${stamp.slice(9, 11)}:${stamp.slice(11, 13)}:${stamp.slice(13, 15)}.${stamp.slice(15, 18)}Z`
    );
    if (Number.isFinite(time)) run.start_time = time;
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

/**
 * A {@link RunTree} subclass safe to construct and drive from workflow code.
 *
 * Subclasses rather than wraps because LangSmith's `traceable` resolves its
 * parent with `instanceof RunTree` before calling `parent.createChild(...)`, so
 * a plain wrapper would refuse to nest under it.
 *
 * @internal
 */
export class ReplaySafeRunTree extends RunTree {
  // Per-invocation run-id factory. Supplied by read-only handlers (queries, update
  // validators) so their ids don't draw from the Workflow main PRNG; left
  // undefined on recorded paths, where id minting falls back to the isolated
  // LANGSMITH_RUN_TREE_STREAM named stream. Propagated to every child so the whole
  // subtree mints from one factory.
  protected readonly generateNewUuid4?: () => string;
  /**
   * Set when the plugin has no `projectName` and LangSmith rejected the env the workflow bundle was built in, on the
   * plugin's own runs in the workflow sandbox, and inherited by each child that names no project or address of its own.
   * Such a run is never sent or put in a trace header: the plugin, having no destination, sends none of its own sandbox
   * runs, wherever they sit, and in that env LangSmith doesn't trace runs that name no destination. LangSmith's
   * `tracingEnabled: false` won't do, since {@link emitMarkerRun} skips its awaits for a run with tracing off, so the
   * workflow's awaits would vary with the env and break replay.
   */
  readonly untraced: boolean;

  constructor(config: RunTreeConfig, generateNewUuid4?: () => string, untraced = false) {
    super(ReplaySafeRunTree.fill(config, generateNewUuid4));
    this.generateNewUuid4 = generateNewUuid4;
    this.untraced = untraced;
  }

  private static fill(config: RunTreeConfig, generateNewUuid4?: () => string): RunTreeConfig {
    const id =
      config.id ?? (generateNewUuid4 ? generateNewUuid4() : getRandomStream(LANGSMITH_RUN_TREE_STREAM).uuid4());
    const start_time = config.start_time ?? Date.now();
    const trace_id = config.trace_id ?? config.parent_run?.trace_id ?? id;
    let dotted_order = config.dotted_order;
    if (dotted_order == null) {
      const startMs = typeof start_time === 'number' ? start_time : Date.parse(String(start_time));
      const { dottedOrder } = convertToDottedOrderFormat(startMs, id);
      const parentDotted = config.parent_run?.dotted_order;
      dotted_order = parentDotted ? `${parentDotted}.${dottedOrder}` : dottedOrder;
    }
    return {
      ...config,
      id,
      start_time,
      trace_id,
      dotted_order,
      parent_run_id: config.parent_run_id ?? config.parent_run?.id,
      client: NOOP_CLIENT,
      tracingEnabled: config.tracingEnabled ?? true,
    };
  }

  /**
   * Produce a replay-safe child so deeply-nested `traceable` runs stay deterministic. With `untraced`, the child
   * is flagged as having no destination even under a run that has one.
   */
  override createChild(config: RunTreeConfig, untraced = false): ReplaySafeRunTree {
    const namesDestination = Boolean(config.project_name || config.address);
    const child = new ReplaySafeRunTree(
      {
        ...config,
        run_type: config.run_type ?? RUN_TYPE.CHAIN,
        parent_run: this,
        parent_run_id: this.id,
        trace_id: this.trace_id,
        ...(namesDestination ? {} : { project_name: this.project_name, address: this.address }),
        replicas: config.replicas ?? this.replicas?.map(({ reroot: _reroot, ...replica }) => replica),
      },
      this.generateNewUuid4,
      untraced || (this.untraced && !namesDestination)
    );
    this.child_runs.push(child);
    return child;
  }

  /** Emit a `createRun` via the Sink, unless replaying or the run has no destination. */
  override async postRun(): Promise<void> {
    if (isReplaying() || this.untraced) {
      return;
    }
    sinks()[LANGSMITH_SINK_NAME].createRun(serializeRun(this));
  }

  /**
   * Record outputs/end-time locally then emit an `updateRun` via the Sink, unless replaying or the run has no
   * destination.
   */
  override async patchRun(): Promise<void> {
    if (isReplaying() || this.untraced) {
      return;
    }
    sinks()[LANGSMITH_SINK_NAME].updateRun(serializeRun(this));
  }

  /** Record end-time / outputs / error on the run object (pure, replay-safe). */
  override async end(outputs?: Record<string, unknown>, error?: string, endTime?: number): Promise<void> {
    if (outputs !== undefined) {
      this.outputs = outputs;
    }
    if (error !== undefined) {
      this.error = error;
    }
    this.end_time = endTime ?? Date.now();
  }
}

/**
 * A placeholder, never-emitted parent whose {@link createChild} produces
 * independent root children. Installed as the ambient so a workflow-body
 * `traceable` takes LangSmith's `createChild` branch (deterministic id) instead
 * of the no-parent branch that mints a uuid via `crypto`, which the isolate lacks.
 *
 * @internal
 */
export class _RootReplaySafeRunTreeFactory extends ReplaySafeRunTree {
  /**
   * Produce a replay-safe child with no link back to this factory; `untraced` as for
   * {@link ReplaySafeRunTree.createChild}.
   */
  override createChild(config: RunTreeConfig, untraced = false): ReplaySafeRunTree {
    const namesDestination = Boolean(config.project_name || config.address);
    return new ReplaySafeRunTree(
      {
        ...config,
        run_type: config.run_type ?? RUN_TYPE.CHAIN,
        ...destination(config, this),
        replicas: config.replicas ?? this.replicas,
      },
      this.generateNewUuid4,
      untraced || (this.untraced && !namesDestination)
    );
  }

  override async postRun(): Promise<void> {}

  override async patchRun(): Promise<void> {}
}
