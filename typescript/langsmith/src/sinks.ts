/**
 * Worker-side LangSmith emission, exposed as a Temporal {@link InjectedSinks}.
 *
 * @module
 */

import { isTracingEnabled, type Client } from 'langsmith';
import { RunTree } from 'langsmith/run_trees';
import type { Sinks } from '@temporalio/workflow';
import type { InjectedSinks } from '@temporalio/worker';

import { ADDRESS_BAGGAGE_ITEM, PROJECT_BAGGAGE_ITEM } from './propagation';

/** Plain-JSON description of a LangSmith run crossing the isolate→worker boundary. */
export interface SerializedRun {
  id: string;
  trace_id: string;
  dotted_order: string;
  parent_run_id?: string;
  name: string;
  run_type: string;
  start_time: number;
  /** End time (epoch ms); present only on the update record. */
  end_time?: number;
  inputs?: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  /** Error string; non-null marks the run errored. */
  error?: string;
  extra?: Record<string, unknown>;
  tags?: string[];
  project_name?: string;
  /** Agent address LRN; set instead of `project_name` for an addressed run. */
  address?: string;
  events?: Record<string, unknown>[];
}

type Destination = Pick<RunTree, 'project_name' | 'address'>;

/** Runtime configuration shared by the client and activity interceptors. */
export interface EmitterConfig {
  /** The LangSmith client runs are emitted to. */
  client: Client;
  /** When false, no Temporal-operation runs are emitted (propagation only). */
  addTemporalRuns: boolean;
  /** Target LangSmith project. */
  projectName?: string;
  /** Tags attached to every emitted run. */
  tags?: string[];
  /** Metadata merged into every emitted run. */
  metadata?: Record<string, unknown>;
}

/** The injected sink's wire name. Reserved-prefixed; allowlisted in `@temporalio/common`. */
export const LANGSMITH_SINK_NAME = '__temporal_langsmith' as const;

/** The Temporal sink surface this plugin injects. */
export interface LangSmithSinks extends Sinks {
  [LANGSMITH_SINK_NAME]: {
    /** Emit a `createRun` for a newly started run. */
    createRun(run: SerializedRun): void;
    /** Emit an `updateRun` (end / outputs / error / events) for a run. */
    updateRun(run: SerializedRun): void;
  };
}

/**
 * The run's destination as LangSmith resolves it on the Worker, where env is readable: the run's own,
 * else LangSmith's env defaults. `undefined` when LangSmith won't trace it.
 */
function resolveDestination(client: Client, run: SerializedRun): Destination | undefined {
  // `fromHeaders` is the one LangSmith API that rebuilds an address from its LRN on every 0.10.x.
  const baggage = run.address
    ? `${ADDRESS_BAGGAGE_ITEM}=${encodeURIComponent(run.address)}`
    : run.project_name
      ? `${PROJECT_BAGGAGE_ITEM}=${encodeURIComponent(run.project_name)}`
      : '';
  const resolved = RunTree.fromHeaders(
    { 'langsmith-trace': run.dotted_order, baggage },
    { client, tracingEnabled: true }
  );
  return resolved?.tracingEnabled ? { project_name: resolved.project_name, address: resolved.address } : undefined;
}

// langsmith's CreateRunParams omits tags, but createRun spreads them into the request at runtime.
function toCreateParams(
  run: SerializedRun,
  destination: Destination
): Parameters<Client['createRun']>[0] & { tags?: string[] } {
  return {
    id: run.id,
    trace_id: run.trace_id,
    dotted_order: run.dotted_order,
    parent_run_id: run.parent_run_id,
    name: run.name,
    run_type: run.run_type,
    start_time: run.start_time,
    inputs: run.inputs ?? {},
    extra: run.extra,
    tags: run.tags,
    project_name: destination.project_name,
    address: destination.address,
  };
}

function toUpdateParams(run: SerializedRun): Parameters<Client['updateRun']>[1] {
  return {
    end_time: run.end_time,
    outputs: run.outputs,
    error: run.error,
    extra: run.extra,
    tags: run.tags,
    events: run.events,
    dotted_order: run.dotted_order,
    trace_id: run.trace_id,
    parent_run_id: run.parent_run_id,
  };
}

/** Build the worker-side {@link InjectedSinks} routing serialized runs to a real LangSmith client. */
export function createLangSmithSinks(client: Client): InjectedSinks<LangSmithSinks> {
  return {
    [LANGSMITH_SINK_NAME]: {
      createRun: {
        fn: (_info, run) => {
          // Honor the LangSmith tracing gate at the real-env emission point.
          if (!isTracingEnabled()) {
            return;
          }
          const destination = resolveDestination(client, run);
          if (!destination) {
            return;
          }
          void Promise.resolve(client.createRun(toCreateParams(run, destination))).catch(() => {
            /* swallow: emission must never fail the workflow */
          });
        },
        callDuringReplay: false,
      },
      updateRun: {
        fn: (_info, run) => {
          if (!isTracingEnabled() || !resolveDestination(client, run)) {
            return;
          }
          void Promise.resolve(client.updateRun(run.id, toUpdateParams(run))).catch(() => {
            /* swallow: emission must never fail the workflow */
          });
        },
        callDuringReplay: false,
      },
    },
  };
}
