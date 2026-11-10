/**
 * Internal descriptions of registered nodes, tasks and entrypoints, and the
 * Activity wire types. Shared by the Worker and the Workflow sandbox.
 *
 * @module
 * @internal
 */

import type { Duration } from '@temporalio/common';
import type { ResolvedOptions } from './options';
import type { ValueTypes } from './codec';

/** A LangGraph node function, as `StateGraph.addNode` stores it. */
export type NodeFunc = (state: unknown, config: Record<string, unknown>) => unknown;

/** A LangGraph task's user function. */
export type TaskFunc = (...args: unknown[]) => unknown;

/** A Graph API node the plugin runs as an Activity or inline. */
export interface NodeSpec extends ResolvedOptions {
  kind: 'node';
  graphName: string;
  nodeName: string;
  activityName: string;
  /** The user's original node function. */
  func: NodeFunc;
}

/** A Functional API task the plugin runs as an Activity or inline. */
export interface TaskSpec extends ResolvedOptions {
  kind: 'task';
  name: string;
  activityName: string;
  /** The user's original task function. */
  func: TaskFunc;
}

/** The structural slice of a `StateGraph` the plugin reads and patches. */
export interface StateGraphLike {
  nodes: Record<string, StateGraphNodeLike>;
  compile(...args: unknown[]): unknown;
  _nodeDefaults?: { retryPolicy?: unknown };
}

export interface StateGraphNodeLike {
  runnable: unknown;
  metadata?: Record<string, unknown>;
  retryPolicy?: unknown;
  isErrorHandler?: boolean;
}

/** LangGraph's `RunnableCallable`: what `addNode` wraps a function in. */
export interface RunnableCallableLike {
  func: (input: unknown, config: Record<string, unknown>) => unknown;
  lc_namespace: string[];
}

/** The structural slice of an `entrypoint()` program the plugin patches. */
export interface PregelLike {
  nodes: Record<string, { bound?: unknown }>;
  invoke(...args: unknown[]): unknown;
}

/** Everything the plugin knows about one `defineLangGraph()` call. */
export interface Internals {
  graphs: Record<string, StateGraphLike>;
  entrypoints: Record<string, PregelLike>;
  nodes: NodeSpec[];
  tasks: TaskSpec[];
  streamingTopic?: string;
  streamingBatchInterval: Duration;
  /** Whether this Workflow's copies of the graphs and entrypoints were patched. */
  installed: boolean;
}

/** The parts of a LangGraph `RunnableConfig` that cross the Activity boundary. */
export interface LangGraphConfigPayload {
  tags?: string[];
  metadata?: Record<string, unknown>;
  runName?: string;
  runId?: string;
  recursionLimit?: number;
  configurable: Record<string, unknown>;
  scratchpad: { resume: unknown[]; nullResume: unknown };
  hasCheckpointer: boolean;
  context?: unknown;
  executionInfo?: unknown;
}

/** Input of a node or task Activity. */
export interface ActivityInput {
  args: unknown[];
  langgraph_config: LangGraphConfigPayload;
  value_types: ValueTypes | null;
}

/** Output of a node or task Activity: its result, or the interrupts it raised. */
export interface ActivityOutput {
  result?: unknown;
  langgraph_interrupts?: unknown[] | null;
  value_types: ValueTypes | null;
}
