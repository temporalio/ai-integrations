/**
 * Per-run state for the Workflow side of the plugin.
 *
 * The SDK evaluates Workflow modules once per run by default, but a Worker can
 * share modules between runs (`preloadModules`), so state is keyed by run ID
 * and dropped when the run's Workflow is disposed (see
 * `./workflow-interceptors`).
 *
 * This module is part of the Workflow-sandbox import graph: it must not import
 * any Worker-only module.
 *
 * @module
 * @internal
 */

import { workflowInfo } from '@temporalio/workflow';
import type { WorkflowStream } from '@temporalio/workflow-streams/workflow';
import type { TaskCache } from './task-cache';

export interface RunState {
  /** Set by the plugin's Workflow interceptor module when the run starts. */
  pluginActive: boolean;
  cache: TaskCache;
  stream?: WorkflowStream;
  warnedStore: boolean;
}

const states = new Map<string, RunState>();

/** The current run's state, created on first use. */
export function runState(): RunState {
  const { runId } = workflowInfo();
  let state = states.get(runId);
  if (state === undefined) {
    state = { pluginActive: false, cache: {}, warnedStore: false };
    states.set(runId, state);
  }
  return state;
}

/** Drops a run's state. */
export function disposeRunState(runId: string): void {
  states.delete(runId);
}
