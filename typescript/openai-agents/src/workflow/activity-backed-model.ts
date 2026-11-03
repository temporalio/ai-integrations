import {
  Usage,
  withGenerationSpan,
  type Agent,
  type AgentOutputItem,
  type Model,
  type ModelProvider,
  type ModelRequest,
  type ModelResponse,
  type ModelSettings,
  type StreamEvent,
  type UsageInput,
} from '@openai/agents-core';
import {
  proxyActivities,
  proxyLocalActivities,
  type ActivityInterfaceFor,
  type LocalActivityInterfaceFor,
  type ActivityOptions,
  type LocalActivityOptions,
} from '@temporalio/workflow';
import { ApplicationFailure } from '@temporalio/common';
import {
  STREAMING_LOCAL_ACTIVITY_UNSUPPORTED,
  STREAMING_TOPIC_NOT_CONFIGURED,
  type ModelActivityOptions,
  type ModelSummaryProvider,
} from '../common/model-activity-options';
import {
  fromSerializedStreamEvent,
  type InvokeModelActivityInput,
  type InvokeModelStreamActivityInput,
  type JsonValue,
  type SerializedModelRequest,
  type SerializedModelResponse,
  type SerializedStreamEvent,
} from '../common/serialized-model';

export function toSerializedModelRequest(request: ModelRequest): SerializedModelRequest {
  return {
    systemInstructions: request.systemInstructions,
    input: request.input as JsonValue,
    modelSettings: request.modelSettings,
    tools: request.tools,
    toolsExplicitlyProvided: request.toolsExplicitlyProvided,
    outputType: request.outputType,
    handoffs: request.handoffs,
    prompt: request.prompt as JsonValue | undefined,
    previousResponseId: request.previousResponseId,
    conversationId: request.conversationId,
    tracing: request.tracing as JsonValue,
    overridePromptModel: request.overridePromptModel,
    reasoningEffortImplicit: (request as { _internal?: { reasoningEffortImplicit?: boolean } })._internal
      ?.reasoningEffortImplicit,
  };
}

/** What a model call needs from the run that makes it. */
export interface ModelCallRun {
  /** The agent whose model call it is. */
  agent(): Agent<any, any> | undefined;
  /** The run's own `modelSettings`. */
  modelSettings(): ModelSettings | undefined;
}

function fromSerializedModelResponse(wire: SerializedModelResponse): ModelResponse {
  return {
    usage: Usage.fromJSON(wire.usage as UsageInput),
    output: wire.output as unknown as AgentOutputItem[],
    responseId: wire.responseId,
    providerData: wire.providerData,
  };
}

interface ModelActivities {
  invokeModelActivity(input: InvokeModelActivityInput): Promise<SerializedModelResponse>;
  invokeModelStreamActivity(input: InvokeModelStreamActivityInput): Promise<SerializedStreamEvent[]>;
}

/**
 * Model implementation that delegates each call to a Temporal Activity, so all
 * LLM calls execute on the Activity Worker (where real ModelProviders live).
 */
export class ActivityBackedModel implements Model {
  private readonly activities: ActivityInterfaceFor<ModelActivities> | LocalActivityInterfaceFor<ModelActivities>;
  private readonly modelParams: ModelActivityOptions;

  constructor(
    private readonly modelName: string,
    modelParams: ModelActivityOptions,
    private readonly run?: ModelCallRun
  ) {
    this.modelParams = modelParams;

    if (modelParams.useLocalActivity) {
      const localOpts: LocalActivityOptions = {
        startToCloseTimeout: modelParams.startToCloseTimeout ?? '60s',
        scheduleToCloseTimeout: modelParams.scheduleToCloseTimeout,
        retry: modelParams.retry,
        cancellationType: modelParams.cancellationType,
        summary: typeof modelParams.summary === 'string' ? modelParams.summary : undefined,
      };
      this.activities = proxyLocalActivities<ModelActivities>(localOpts);
    } else {
      const { useLocalActivity: _useLocalActivity, summary, ...rest } = modelParams;
      const opts: ActivityOptions = {
        ...rest,
        startToCloseTimeout: modelParams.startToCloseTimeout ?? '60s',
        summary: typeof summary === 'string' ? summary : undefined,
      };
      this.activities = proxyActivities<ModelActivities>(opts);
    }
  }

  async getResponse(request: ModelRequest): Promise<ModelResponse> {
    // Wrap in a generation span so the trace tree has agent → generation → Activity.
    return withGenerationSpan(async (span) => {
      span.spanData.model = this.modelName;

      const agent = this.run?.agent();
      const input: InvokeModelActivityInput = {
        modelName: this.modelName,
        request: this.toWire(request, agent),
      };

      const modelSummary = this.modelParams.summary;
      if (modelSummary && typeof modelSummary !== 'string') {
        const provider = modelSummary as ModelSummaryProvider;
        const systemInstructions = request.systemInstructions;
        const summary = provider.provide(agent, systemInstructions, request.input);
        const wireResponse = await this.activities.invokeModelActivity.executeWithOptions({ summary }, [input]);
        return fromSerializedModelResponse(wireResponse);
      }

      return fromSerializedModelResponse(await this.activities.invokeModelActivity(input));
    });
  }

  async *getStreamedResponse(request: ModelRequest): AsyncIterable<StreamEvent> {
    const streamingTopic = this.modelParams.streamingTopic;
    if (streamingTopic === undefined) {
      throw ApplicationFailure.create({
        message: STREAMING_TOPIC_NOT_CONFIGURED.message,
        type: STREAMING_TOPIC_NOT_CONFIGURED.type,
        nonRetryable: true,
      });
    }
    // The runner checks its own streamed runs; this also covers nested `agent.asTool()` runs.
    if (this.modelParams.useLocalActivity === true) {
      throw ApplicationFailure.create({
        message: STREAMING_LOCAL_ACTIVITY_UNSUPPORTED.message,
        type: STREAMING_LOCAL_ACTIVITY_UNSUPPORTED.type,
        nonRetryable: true,
      });
    }

    const agent = this.run?.agent();
    const input: InvokeModelStreamActivityInput = {
      modelName: this.modelName,
      request: this.toWire(request, agent),
      streamingTopic,
      ...(this.modelParams.streamingBatchInterval !== undefined && {
        streamingBatchInterval:
          typeof this.modelParams.streamingBatchInterval === 'number'
            ? `${this.modelParams.streamingBatchInterval}ms`
            : this.modelParams.streamingBatchInterval,
      }),
    };

    const events = await withGenerationSpan(async (span) => {
      span.spanData.model = this.modelName;

      const modelSummary = this.modelParams.summary;
      if (modelSummary && typeof modelSummary !== 'string') {
        const provider = modelSummary as ModelSummaryProvider;
        const summary = provider.provide(agent, request.systemInstructions, request.input);
        return this.activities.invokeModelStreamActivity.executeWithOptions({ summary }, [input]);
      }

      return this.activities.invokeModelStreamActivity(input);
    });

    // The Activity collects and returns the full event list; yielding from that
    // return value (rather than polling the live stream) keeps replay deterministic.
    for (const event of events) {
      yield fromSerializedStreamEvent(event);
    }
  }

  /**
   * With no model named, upstream adds the model settings defaults for the Workflow's default model,
   * which the Worker may not use. If the agent set no model settings, send only the run's, and let
   * the Worker add the defaults for its own default model, as upstream does in one process.
   */
  private toWire(request: ModelRequest, agent: Agent<any, any> | undefined): SerializedModelRequest {
    const wire = toSerializedModelRequest(request);
    const hasModelSettings = (agent as { hasExplicitModelSettings?(): boolean } | undefined)?.hasExplicitModelSettings;
    if (this.modelName !== '' || agent === undefined || hasModelSettings?.call(agent) !== false) return wire;
    const modelSettings: ModelSettings = { ...this.run?.modelSettings() };
    // Upstream clears `toolChoice` after a tool call when the agent sets `resetToolChoice`.
    if (request.modelSettings.toolChoice === undefined) delete modelSettings.toolChoice;
    // A `SandboxAgent`'s capabilities add `providerData`, such as compaction, to the settings upstream
    // calls its model with. The defaults never set it.
    if (request.modelSettings.providerData !== undefined)
      modelSettings.providerData = request.modelSettings.providerData;
    // A run setting set to `undefined` clears a default upstream, but JSON would drop it.
    const cleared = (Object.keys(modelSettings) as (keyof ModelSettings)[]).filter(
      (key) => modelSettings[key] === undefined
    );
    return {
      ...wire,
      modelSettings,
      reasoningEffortImplicit: undefined,
      implicitModelSettings: true,
      ...(cleared.length > 0 && { clearedModelSettings: cleared }),
    };
  }
}

/**
 * Resolves model names to `ActivityBackedModel`s. The runner gives it to its inner `Runner`, and
 * upstream passes it on to nested `agent.asTool()` runs, so their model calls are Activities too.
 * The run check gives each run a copy bound to that run (see `installRunnerCheck`), so its model
 * calls know their agent and the run's model settings.
 */
export class ActivityBackedModelProvider implements ModelProvider {
  constructor(
    private readonly modelParams: ModelActivityOptions,
    private readonly run?: ModelCallRun
  ) {}

  forRun(run: ModelCallRun): ActivityBackedModelProvider {
    return new ActivityBackedModelProvider(this.modelParams, run);
  }

  getModel(modelName?: string): Model {
    return new ActivityBackedModel(modelName ?? '', this.modelParams, this.run);
  }
}
