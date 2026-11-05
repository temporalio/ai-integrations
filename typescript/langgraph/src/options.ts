/**
 * Option types shared by the Worker and the Workflow sandbox.
 *
 * @module
 */

import type { ActivityOptions } from '@temporalio/workflow';

/** Where a node or task runs: as a Temporal Activity, or inline in the Workflow. */
export type ExecuteIn = 'activity' | 'workflow';

/**
 * Computes a summary from a node's or task's arguments: `[state]` for a Graph
 * API node, the call arguments for a Functional API task. An empty result means
 * no summary.
 */
export type SummaryFn = (args: unknown[]) => string | null | undefined;

/**
 * Activity options for nodes and tasks: any Temporal {@link ActivityOptions}
 * field, plus {@link summaryFn}.
 *
 * `summary` and `summaryFn` are two forms of one setting. Set at most one of
 * them in each options object; a node or task that sets either overrides an
 * inherited default of either form.
 */
export type LangGraphActivityOptions = { [K in keyof ActivityOptions]: ActivityOptions[K] } & {
  /**
   * Computes the summary from the node's or task's arguments. On an Activity
   * node or task it becomes the Activity summary; on a Workflow node or task it
   * sets the Workflow's current details (an empty result clears them).
   */
  summaryFn?: SummaryFn;
};

/** Options for one node (Graph API `metadata`) or task (Functional API `activityOptions`). */
export type NodeOptions = LangGraphActivityOptions & {
  /** Required: run the node or task as an Activity or inline in the Workflow. */
  executeIn: ExecuteIn;
};

/**
 * The option keys the plugin reads out of a node's `metadata`. Every other key
 * stays on the node, where LangGraph exposes it to the node as
 * `config.metadata`.
 */
export const OPTION_KEYS: ReadonlySet<string> = new Set([
  'executeIn',
  'summaryFn',
  'activityId',
  'taskQueue',
  'heartbeatTimeout',
  'retry',
  'startToCloseTimeout',
  'scheduleToStartTimeout',
  'scheduleToCloseTimeout',
  'cancellationType',
  'allowEagerDispatch',
  'versioningIntent',
  'summary',
  'priority',
]);

/** A node or task's resolved execution settings. */
export interface ResolvedOptions {
  executeIn: ExecuteIn;
  activityOptions: ActivityOptions;
  summaryFn?: SummaryFn;
}

/** Splits `metadata` into plugin options and the user metadata left on the node. */
export function splitMetadata(metadata: Record<string, unknown> | undefined): {
  options: Record<string, unknown>;
  rest: Record<string, unknown>;
} {
  const options: Record<string, unknown> = {};
  const rest: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(metadata ?? {})) {
    if (OPTION_KEYS.has(key)) options[key] = value;
    else rest[key] = value;
  }
  return { options, rest };
}

/**
 * Layers a node's or task's options over the defaults. A node or task that
 * sets `summary` or `summaryFn` overrides an inherited default of either form,
 * so the two never meet in one merged options object.
 */
export function mergeOptions(
  defaults: LangGraphActivityOptions | undefined,
  specific: Record<string, unknown>
): Record<string, unknown> {
  const merged: Record<string, unknown> = { ...defaults };
  if ('summary' in specific || 'summaryFn' in specific) {
    delete merged.summary;
    delete merged.summaryFn;
  }
  return { ...merged, ...specific };
}

/** Validates merged options and normalizes `summary` into a `summaryFn`. */
export function resolveOptions(label: string, merged: Record<string, unknown>): ResolvedOptions {
  const { executeIn, summaryFn, summary, ...activityOptions } = merged;
  if (executeIn !== 'activity' && executeIn !== 'workflow') {
    throw new TypeError(`Invalid executeIn value: ${String(executeIn)}`);
  }
  if (summaryFn != null && summary != null) {
    throw new TypeError(`${label}: set either 'summary' or 'summaryFn', not both.`);
  }
  if (summaryFn != null && typeof summaryFn !== 'function') {
    throw new TypeError(`${label}: 'summaryFn' must be a function.`);
  }
  const fn =
    summary != null ? () => String(summary) : (summaryFn as SummaryFn | undefined);
  return { executeIn, activityOptions: activityOptions as ActivityOptions, summaryFn: fn };
}

/** Validates the plugin-wide default options. */
export function validateDefaults(defaults: LangGraphActivityOptions | undefined): void {
  if (defaults == null) return;
  if ('executeIn' in defaults) {
    throw new TypeError(
      'executeIn cannot be set in defaultActivityOptions. Set it on each node\'s metadata (Graph API) or in ' +
        'activityOptions[taskName] (Functional API).'
    );
  }
  if (defaults.summary != null && defaults.summaryFn != null) {
    throw new TypeError("Set either 'summary' or 'summaryFn' in defaultActivityOptions, not both.");
  }
}
