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
  AsyncLocalStorage,
  proxyActivities,
  proxyLocalActivities,
  type ActivityInterfaceFor,
  type LocalActivityInterfaceFor,
  type LocalActivityOptions,
} from '@temporalio/workflow';
import { ApplicationFailure } from '@temporalio/common';
import {
  mergeModelOptions,
  STREAMING_LOCAL_ACTIVITY_UNSUPPORTED,
  STREAMING_TOPIC_NOT_CONFIGURED,
  type TemporalModelOptions,
  type ResolvedModelOptions,
  type ModelSummaryProvider,
} from '../common/model-options';
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
  /** The model fallback configured for this run. */
  model(): string | Model | undefined;
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

interface ModelCallContext {
  defaults: TemporalModelOptions;
  overrides: TemporalModelOptions;
  run: ModelCallRun;
  streaming: boolean;
}

// The Workflow SDK supplies deterministic async-local storage in its sandbox.
const modelCallContext = new AsyncLocalStorage<ModelCallContext>();

function currentModelCall(): ModelCallContext | undefined {
  return modelCallContext.getStore?.();
}

/** Validate before the SDK starts a streamed run or schedules a model Activity. */
function validateOptions(options: ResolvedModelOptions, streaming: boolean): void {
  if (options.activityOptions.summary !== undefined && options.summaryProvider !== undefined) {
    throw ApplicationFailure.create({
      message: 'Set either activityOptions.summary or summaryProvider for a model call.',
      type: 'ConflictingModelSummary',
      nonRetryable: true,
    });
  }
  if (streaming && options.streamingTopic === undefined) {
    throw ApplicationFailure.create({ ...STREAMING_TOPIC_NOT_CONFIGURED, nonRetryable: true });
  }
  if (streaming && options.useLocalActivity) {
    throw ApplicationFailure.create({ ...STREAMING_LOCAL_ACTIVITY_UNSUPPORTED, nonRetryable: true });
  }
  if (options.useLocalActivity) {
    const supported = new Set([
      'summary',
      'startToCloseTimeout',
      'scheduleToCloseTimeout',
      'scheduleToStartTimeout',
      'retry',
      'cancellationType',
      'eventGroups',
    ]);
    const unsupported = Object.keys(options.activityOptions).filter((key) => !supported.has(key));
    if (unsupported.length > 0) {
      throw ApplicationFailure.create({
        message: `Local model Activities do not support options: ${unsupported.sort().join(', ')}`,
        type: 'UnsupportedLocalActivityOptions',
        nonRetryable: true,
      });
    }
  }
}

function optionsForAgent(context: ModelCallContext, agent: Agent<any, any>): ResolvedModelOptions {
  // Follow upstream: an agent's own model wins over the run's fallback.
  const selected = agent.model || context.run.model();
  return mergeModelOptions(
    context.defaults,
    selected instanceof TemporalModel ? selected.options : undefined,
    context.overrides
  );
}

/** Recheck agents introduced or changed by handoffs, hooks, or nested tool runs. */
export function validateCurrentModelOptions(agent: Agent<any, any>): void {
  const context = currentModelCall();
  if (context !== undefined) validateOptions(optionsForAgent(context, agent), context.streaming);
}

/**
 * A Workflow-safe OpenAI Agents Model that invokes the provider configured on the Worker.
 * Use `Agent({ model: new TemporalModel(name, { activityOptions }) })`, or a runConfig.model fallback.
 * Model activityOptions override the client/runner defaults; run activityOptions wins per field.
 * Credentials and provider clients remain on the Worker.
 */
export class TemporalModel implements Model {
  readonly modelName: string;
  readonly options: Readonly<TemporalModelOptions>;

  constructor(modelName = '', options: TemporalModelOptions = {}) {
    if (typeof modelName !== 'string') throw new TypeError('TemporalModel modelName must be a string');
    this.modelName = modelName;
    this.options = { ...options, activityOptions: { ...options.activityOptions } };
  }

  /** Model identifier used by upstream sandbox capabilities. */
  get model(): string {
    return this.modelName;
  }

  private effectiveOptions(streaming: boolean): ResolvedModelOptions {
    const context = currentModelCall();
    const options = mergeModelOptions(context?.defaults, this.options, context?.overrides);
    validateOptions(options, streaming);
    return options;
  }

  private createActivities(
    options: ResolvedModelOptions
  ): ActivityInterfaceFor<ModelActivities> | LocalActivityInterfaceFor<ModelActivities> {
    return options.useLocalActivity
      ? proxyLocalActivities<ModelActivities>(options.activityOptions as LocalActivityOptions)
      : proxyActivities<ModelActivities>({ heartbeatTimeout: '10s', ...options.activityOptions });
  }

  async getResponse(request: ModelRequest): Promise<ModelResponse> {
    // Wrap in a generation span so the trace tree has agent → generation → Activity.
    return withGenerationSpan(async (span) => {
      span.spanData.model = this.modelName;

      const context = currentModelCall();
      const agent = context?.run.agent();
      const options = this.effectiveOptions(false);
      const activities = this.createActivities(options);
      const input: InvokeModelActivityInput = {
        modelName: this.modelName,
        request: this.toWire(request, agent, context?.run.modelSettings()),
      };

      const modelSummary = options.summaryProvider;
      if (modelSummary !== undefined) {
        const provider = modelSummary as ModelSummaryProvider;
        const systemInstructions = request.systemInstructions;
        const summary = provider.provide(agent, systemInstructions, request.input);
        const wireResponse = await activities.invokeModelActivity.executeWithOptions({ summary }, [input]);
        return fromSerializedModelResponse(wireResponse);
      }

      return fromSerializedModelResponse(await activities.invokeModelActivity(input));
    });
  }

  async *getStreamedResponse(request: ModelRequest): AsyncIterable<StreamEvent> {
    const options = this.effectiveOptions(true);
    const activities = this.createActivities(options);
    const streamingTopic = options.streamingTopic!;
    const context = currentModelCall();
    const agent = context?.run.agent();
    const input: InvokeModelStreamActivityInput = {
      modelName: this.modelName,
      request: this.toWire(request, agent, context?.run.modelSettings()),
      streamingTopic,
      ...(options.streamingBatchInterval !== undefined && {
        streamingBatchInterval:
          typeof options.streamingBatchInterval === 'number'
            ? `${options.streamingBatchInterval}ms`
            : options.streamingBatchInterval,
      }),
    };

    const events = await withGenerationSpan(async (span) => {
      span.spanData.model = this.modelName;

      const modelSummary = options.summaryProvider;
      if (modelSummary !== undefined) {
        const provider = modelSummary as ModelSummaryProvider;
        const summary = provider.provide(agent, request.systemInstructions, request.input);
        return activities.invokeModelStreamActivity.executeWithOptions({ summary }, [input]);
      }

      return activities.invokeModelStreamActivity(input);
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
   * the Worker add the defaults for its own default model, as upstream does in one process. A
   * `SandboxAgent` takes this path too, so the run's settings win over the defaults; upstream gives
   * its run copy the defaults as its own settings, which win. Explicit TemporalModel objects retain
   * the SDK's Model-object settings, even when their model name is omitted.
   */
  private toWire(
    request: ModelRequest,
    agent: Agent<any, any> | undefined,
    runModelSettings: ModelSettings | undefined
  ): SerializedModelRequest {
    // Upstream times the call out by aborting `request.signal`, which can't stop the Activity.
    if (request.modelSettings.timeoutMs !== undefined) {
      throw ApplicationFailure.create({
        message:
          "modelSettings.timeoutMs can't stop a model call that runs as an Activity. Set startToCloseTimeout " +
          'in activityOptions on the plugin, runner, TemporalModel, or run, instead.',
        type: 'AgentsWorkflowError',
        nonRetryable: true,
      });
    }
    const wire = toSerializedModelRequest(request);
    const selected = agent?.model || currentModelCall()?.run.model();
    const hasModelSettings = (agent as { hasExplicitModelSettings?(): boolean } | undefined)?.hasExplicitModelSettings;
    if (
      selected instanceof TemporalModel ||
      this.modelName !== '' ||
      agent === undefined ||
      hasModelSettings?.call(agent) !== false
    )
      return wire;
    const modelSettings: ModelSettings = { ...runModelSettings };
    // Upstream clears `toolChoice` after a tool call when the agent sets `resetToolChoice`.
    if (request.modelSettings.toolChoice === undefined) delete modelSettings.toolChoice;
    // A `SandboxAgent`'s capabilities add `providerData`, such as compaction, to the settings upstream
    // calls its model with. The defaults never set it.
    if (request.modelSettings.providerData !== undefined)
      modelSettings.providerData = request.modelSettings.providerData;
    // A run setting set to `undefined` clears a default upstream, but JSON would drop it. The defaults
    // also set keys of `reasoning` and `text`, which upstream merges key by key.
    const cleared: string[] = [];
    for (const [key, value] of Object.entries(modelSettings)) {
      if (value === undefined) cleared.push(key);
      else if (key === 'reasoning' || key === 'text') {
        for (const [nested, nestedValue] of Object.entries(value ?? {}))
          if (nestedValue === undefined) cleared.push(`${key}.${nested}`);
      }
    }
    return {
      ...wire,
      modelSettings,
      reasoningEffortImplicit: undefined,
      implicitModelSettings: true,
      ...(cleared.length > 0 && { clearedModelSettings: cleared }),
    };
  }
}

/** Resolves string names and carries Activity defaults through nested SDK runs. */
export class ActivityBackedModelProvider implements ModelProvider {
  constructor(
    private readonly options: TemporalModelOptions,
    private readonly overrides: TemporalModelOptions = {},
    private readonly run?: ModelCallRun
  ) {}

  forRun(run: ModelCallRun): ActivityBackedModelProvider {
    return new ActivityBackedModelProvider(this.options, this.overrides, run);
  }

  withContext<T>(callback: () => T, streaming: boolean): T {
    if (this.run === undefined) throw new Error('Model provider must be bound to an agent run');
    return modelCallContext.run(
      { defaults: this.options, overrides: this.overrides, run: this.run, streaming },
      callback
    );
  }

  validateAgent(agent: Agent<any, any>, fallback: string | Model | undefined, streaming: boolean): void {
    const selected = agent.model || fallback;
    validateOptions(
      mergeModelOptions(this.options, selected instanceof TemporalModel ? selected.options : undefined, this.overrides),
      streaming
    );
  }

  getModel(modelName?: string): Model {
    return new TemporalModel(modelName ?? '');
  }
}
