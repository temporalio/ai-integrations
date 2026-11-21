import type { Duration } from '@temporalio/common';
import type { ActivityOptions } from '@temporalio/workflow';
import {
  DEFAULT_MODEL_ACTIVITY_OPTIONS,
  type ModelActivityOptions,
  type SerializableModelActivityOptions,
  type ModelSummaryProvider,
} from './model-activity-options';

export type { ModelSummaryProvider };

/** Model execution settings; scheduling uses the SDK's own ActivityOptions. */
export interface TemporalModelOptions {
  /** Temporal SDK scheduling options, merged per field. Regular Activities default to a 10-second heartbeat timeout. */
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
export type SerializableModelOptions = Omit<TemporalModelOptions, 'summaryProvider'> & {
  /** @deprecated Use activityOptions and separate execution options instead. */
  modelParams?: SerializableModelActivityOptions;
};

/** Workflow defaults, including the deprecated configuration spelling. */
export interface RunnerModelOptions extends TemporalModelOptions {
  /** @deprecated Use activityOptions and separate execution options instead. */
  defaultModelParams?: ModelActivityOptions;
}

export type ResolvedModelOptions = TemporalModelOptions & { activityOptions: ActivityOptions };

// Workflow-only provenance: local calls forward the fields supported by the old API.
// Keep the other fields until the call chooses its execution mode, so a switch to
// regular Activities retains them.
const legacyActivityFields = Symbol('legacyActivityFields');
type ModelOptionsLayer = TemporalModelOptions & { [legacyActivityFields]?: ReadonlySet<string> };

export const DEFAULT_MODEL_OPTIONS: ResolvedModelOptions = {
  activityOptions: { startToCloseTimeout: '60s' },
  useLocalActivity: false,
};

/** Merge SDK scheduling options per field and keep execution settings separate. */
export function mergeModelOptions(...layers: (Readonly<TemporalModelOptions> | undefined)[]): ResolvedModelOptions {
  const legacyFields = new Set<string>();
  const result: ResolvedModelOptions & ModelOptionsLayer = { activityOptions: {}, [legacyActivityFields]: legacyFields };
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
      if (value === undefined) continue;
      Object.assign(result.activityOptions, { [key]: value });
      if ((layer as ModelOptionsLayer)[legacyActivityFields]?.has(key)) legacyFields.add(key);
      else legacyFields.delete(key);
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


/** Convert legacy flat settings without adding defaults to a partial layer. */
export function modelOptionsFromLegacy(options: ModelActivityOptions | undefined): ModelOptionsLayer | undefined {
  if (options === undefined) return undefined;
  const { useLocalActivity, streamingTopic, streamingBatchInterval, summary, ...activityOptions } = options;
  return {
    activityOptions: { ...activityOptions, ...(typeof summary === 'string' ? { summary } : {}) },
    useLocalActivity,
    streamingTopic,
    streamingBatchInterval,
    ...(typeof summary === 'object' && summary !== null ? { summaryProvider: summary } : {}),
    [legacyActivityFields]: new Set(Object.keys(activityOptions)),
  };
}

const legacyLocalModelActivityOptions = new Set([
  'summary',
  'startToCloseTimeout',
  'scheduleToCloseTimeout',
  'retry',
  'cancellationType',
]);

export const LOCAL_MODEL_ACTIVITY_OPTIONS = new Set([
  ...legacyLocalModelActivityOptions,
  'scheduleToStartTimeout',
  'eventGroups',
]);

/** Layer deprecated and new spellings without changing Client precedence. */
export function mergeRunnerModelOptions(
  options: RunnerModelOptions | undefined,
  header: SerializableModelOptions | undefined
): ResolvedModelOptions {
  return mergeModelOptions(
    modelOptionsFromLegacy(options?.defaultModelParams),
    options,
    modelOptionsFromLegacy(header?.modelParams),
    header
  );
}

/** Apply legacy local handling only after model and run overrides are resolved. */
export function activityOptionsForModel(options: ResolvedModelOptions): ActivityOptions {
  const result = { ...options.activityOptions };
  if (!options.useLocalActivity) return result;
  for (const key of (options as ModelOptionsLayer)[legacyActivityFields] ?? []) {
    if (!legacyLocalModelActivityOptions.has(key)) delete result[key as keyof ActivityOptions];
  }
  return result;
}

/** @deprecated Use activityOptions and separate execution options with mergeModelOptions. */
export function layerModelParams(
  defaults: ModelActivityOptions | undefined,
  header: SerializableModelActivityOptions | undefined
): ModelActivityOptions {
  const result: ModelActivityOptions = { ...DEFAULT_MODEL_ACTIVITY_OPTIONS };
  for (const layer of [defaults, header]) {
    for (const [key, value] of Object.entries(layer ?? {})) {
      if (value !== undefined) Object.assign(result, { [key]: value });
    }
  }
  return result;
}
