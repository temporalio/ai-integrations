import type { Agent, AgentInputItem } from '@openai/agents-core';
import type { Duration } from '@temporalio/common';
import type { ActivityOptions } from '@temporalio/workflow';

export interface ModelSummaryProvider {
  /**
   * Generate a human-readable summary for the model Activity shown in the Temporal UI.
   * Runs inside the Workflow sandbox, so it must be deterministic and perform no non-deterministic I/O.
   */
  provide(
    agent: Agent<any, any> | undefined,
    instructions: string | undefined,
    input: string | AgentInputItem[]
  ): string;
}

/** Model execution settings; scheduling uses the SDK's own ActivityOptions. */
export interface TemporalModelOptions {
  /** Temporal SDK scheduling options, merged per field with inherited defaults. */
  activityOptions?: ActivityOptions;
  /** Use local Activities instead of regular Activities. @default false */
  useLocalActivity?: boolean;
  /** Deterministic dynamic Activity summary; static text uses activityOptions.summary. */
  summaryProvider?: ModelSummaryProvider;
  /**
   * Topic that the streaming model Activity publishes raw model stream events to
   * for live external consumption. Required when running with `{ stream: true }`;
   * `run()` throws before scheduling any Activity if it is unset. The Workflow must
   * host a `WorkflowStream` (from `@temporalio/workflow-streams/workflow`) to receive
   * the publishes.
   *
   * @experimental Streaming support is experimental and may change without notice.
   */
  streamingTopic?: string;
  /**
   * Batch flush interval for the stream publisher used by the streaming Activity.
   *
   * @experimental Streaming support is experimental and may change without notice.
   */
  streamingBatchInterval?: Duration;
}

/**
 * Plugin-side model Activity options propagated to Workflows via the
 * `__openai_agents_config` header. summaryProvider is Workflow-authored because
 * functions cannot survive JSON serialization.
 */
export type SerializableModelOptions = Omit<TemporalModelOptions, 'summaryProvider'>;

export type ResolvedModelOptions = TemporalModelOptions & { activityOptions: ActivityOptions };

export const DEFAULT_MODEL_OPTIONS: ResolvedModelOptions = {
  activityOptions: { startToCloseTimeout: '60s' },
  useLocalActivity: false,
};

/** Merge SDK scheduling options per field and keep execution settings separate. */
export function mergeModelOptions(...layers: (Readonly<TemporalModelOptions> | undefined)[]): ResolvedModelOptions {
  const result: ResolvedModelOptions = { activityOptions: {} };
  for (const layer of [DEFAULT_MODEL_OPTIONS, ...layers]) {
    if (layer === undefined) continue;
    // A higher-layer static or dynamic summary replaces the inherited summary choice.
    if (layer.activityOptions?.summary !== undefined || layer.summaryProvider !== undefined) {
      delete result.activityOptions.summary;
      delete result.summaryProvider;
    }
    for (const [key, value] of Object.entries(layer)) {
      if (key !== 'activityOptions' && value !== undefined) Object.assign(result, { [key]: value });
    }
    for (const [key, value] of Object.entries(layer.activityOptions ?? {})) {
      if (value !== undefined) Object.assign(result.activityOptions, { [key]: value });
    }
  }
  return result;
}

export const STREAMING_TOPIC_NOT_CONFIGURED = {
  type: 'StreamingTopicNotConfigured',
  message:
    'Streaming requires a streamingTopic. Set it on the Client plugin, runner, TemporalModel, or run options. ' +
    'Host a WorkflowStream in your Workflow to use run(agent, input, { stream: true }).',
} as const;

export const STREAMING_LOCAL_ACTIVITY_UNSUPPORTED = {
  type: 'StreamingLocalActivityUnsupported',
  message:
    'Streaming is incompatible with useLocalActivity. Local Activities cannot heartbeat, so a ' +
    'long-running streaming model call cannot extend past its startToCloseTimeout. ' +
    'Set useLocalActivity: false (the default) for streaming runs.',
} as const;
