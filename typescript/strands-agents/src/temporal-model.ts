import { Model } from '@strands-agents/sdk';
import type { BaseModelConfig, Message, ModelStreamEvent, StreamOptions } from '@strands-agents/sdk';
import * as workflow from '@temporalio/workflow';
import type { ActivityOptions } from '@temporalio/workflow';
import type { Duration } from '@temporalio/common/lib/time';
import type {
  GetModelConfigInput,
  InvokeModelInput,
  InvokeModelStreamingInput,
  ModelConfigInfo,
} from './model-activity';

/** Name of the activity that returns the worker model's {@link ModelConfigInfo}. */
export const GET_MODEL_CONFIG_ACTIVITY = 'strandsGetModelConfig';

/**
 * Options for {@link TemporalModel}.
 *
 * The {@link ActivityOptions} are applied to every activity invocation
 * `stream()` schedules. `streamingTopic` and `streamingBatchInterval`
 * forward {@link ModelStreamEvent}s on a {@link WorkflowStream} as
 * they are produced inside the model activity.
 */
export interface TemporalModelOptions {
  modelName?: string;
  activityOptions?: ActivityOptions;
  streamingTopic?: string;
  streamingBatchInterval?: Duration;
}

/**
 * A Strands {@link Model} whose {@link Model.stream} runs as a Temporal
 * activity. Constructing a `TemporalModel` does no I/O, so instances are
 * safe to create at module load time.
 *
 * The `modelName` selects which factory the plugin will invoke worker-side;
 * it must match a key in {@link StrandsPlugin}'s `models` map. A name not
 * present in `models` raises an error inside the model activity at first call.
 */
export class TemporalModel extends Model<BaseModelConfig> {
  private readonly modelName?: string;
  private readonly activityOptions: ActivityOptions;
  private readonly streamingTopic?: string;
  private readonly streamingBatchInterval: Duration;
  private config: ModelConfigInfo = {};
  private configLoaded = false;

  constructor(options: TemporalModelOptions = {}) {
    super();
    this.modelName = options.modelName;
    this.activityOptions = options.activityOptions ?? {};
    this.streamingTopic = options.streamingTopic;
    this.streamingBatchInterval = options.streamingBatchInterval ?? '100 milliseconds';
  }

  override updateConfig(_config: BaseModelConfig): void {
    // No-op; the real model is configured worker-side via the plugin's factories.
  }

  /**
   * The worker model's id and context window, once {@link loadConfig} has read them; otherwise
   * empty.
   */
  override getConfig(): BaseModelConfig {
    return { ...this.config };
  }

  /**
   * Read the worker model's id and context window through an activity, once per instance, so that
   * Strands' traces and proactive compression use the real model's values. {@link TemporalAgent}
   * calls this before Strands initializes the agent.
   *
   * Returns `undefined`, without awaiting anything, when the config was already read, so callers
   * can go on synchronously.
   */
  loadConfig(): Promise<void> | undefined {
    if (this.configLoaded) return undefined;
    const activities = workflow.proxyActivities<{
      [GET_MODEL_CONFIG_ACTIVITY](input: GetModelConfigInput): Promise<ModelConfigInfo>;
    }>({
      startToCloseTimeout: '10 minutes',
      ...this.activityOptions,
    });
    return activities[GET_MODEL_CONFIG_ACTIVITY]({ modelName: this.modelName }).then((config) => {
      this.config = config;
      this.configLoaded = true;
    });
  }

  override async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
    const proxiedActivities = workflow.proxyActivities<{
      invokeModel(input: InvokeModelInput): Promise<ModelStreamEvent[]>;
      invokeModelStreaming(input: InvokeModelStreamingInput): Promise<ModelStreamEvent[]>;
    }>({
      startToCloseTimeout: '10 minutes',
      ...this.activityOptions,
    });

    const events = this.streamingTopic
      ? await proxiedActivities.invokeModelStreaming({
          modelName: this.modelName,
          messages,
          options,
          streamingTopic: this.streamingTopic,
          streamingBatchInterval: this.streamingBatchInterval,
        })
      : await proxiedActivities.invokeModel({
          modelName: this.modelName,
          messages,
          options,
        });

    for (const event of events) {
      yield event;
    }
  }
}
