/**
 * Worker-side LangSmith emission, exposed as a Temporal {@link InjectedSinks}.
 *
 * @module
 */

import { isTracingEnabled, type Client } from 'langsmith';
import { RunTree, type RunTreeConfig, type WriteReplica } from 'langsmith/run_trees';
import type { Sinks } from '@temporalio/workflow';
import type { InjectedSinks } from '@temporalio/worker';
import { bindReplicaCredentials, type ReplicaRouting } from './routing';

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
  address?: RunTreeConfig['address'];
  replicas?: ReplicaRouting[];
  distributedParentId?: string;
  events?: Record<string, unknown>[];
}

/** Runtime configuration shared by the client and activity interceptors. */
export interface EmitterConfig {
  /** The LangSmith client runs are emitted to. */
  client: Client;
  /** When false, no Temporal-operation runs are emitted (propagation only). */
  addTemporalRuns: boolean;
  /** Target LangSmith project. */
  projectName?: string;
  address?: RunTreeConfig['address'];
  /** Full replica configuration is local to the client/worker process. */
  replicas?: WriteReplica[];
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

function emissionRun(run: SerializedRun, client: Client, replicas: WriteReplica[] | undefined): RunTree {
  const tree = new RunTree({
    ...run,
    inputs: run.inputs ?? {},
    client,
    replicas: bindReplicaCredentials(run.replicas, replicas),
    tracingEnabled: true,
  });
  tree.events = run.events as RunTree['events'];
  return tree;
}

/** Build the worker-side {@link InjectedSinks} routing serialized runs to a real LangSmith client. */
export function createLangSmithSinks(client: Client, replicas?: WriteReplica[]): InjectedSinks<LangSmithSinks> {
  return {
    [LANGSMITH_SINK_NAME]: {
      createRun: {
        fn: (_info, run) => {
          // Honor the LangSmith tracing gate at the real-env emission point.
          if (!isTracingEnabled()) {
            return;
          }
          void emissionRun(run, client, replicas).postRun().catch(() => {
            /* swallow: emission must never fail the workflow */
          });
        },
        callDuringReplay: false,
      },
      updateRun: {
        fn: (_info, run) => {
          if (!isTracingEnabled()) {
            return;
          }
          void emissionRun(run, client, replicas).patchRun().catch(() => {
            /* swallow: emission must never fail the workflow */
          });
        },
        callDuringReplay: false,
      },
    },
  };
}
