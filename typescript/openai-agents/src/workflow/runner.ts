import {
  type Agent,
  type AgentInputItem,
  Handoff,
  MemorySession,
  Runner,
  RunState,
  type AgentOutputType,
  type CallModelInputFilter,
  type HandoffInputData,
  type InputGuardrail,
  type ModelSettings,
  type OutputGuardrail,
  type RunResult,
  type Session,
  type SessionInputCallback,
  type StreamedRunResult,
  type TracingConfig,
} from '@openai/agents-core';
import { SandboxAgent, type SandboxRunConfig } from '@openai/agents-core/sandbox';
import { ApplicationFailure } from '@temporalio/common';
import {
  DEFAULT_MODEL_ACTIVITY_OPTIONS,
  STREAMING_LOCAL_ACTIVITY_UNSUPPORTED,
  STREAMING_TOPIC_NOT_CONFIGURED,
  type ModelActivityOptions,
  type SerializableModelActivityOptions,
} from '../common/model-activity-options';
import { unwrapTemporalFailure } from '../common/errors';
import { ActivityBackedModelProvider } from './activity-backed-model';
import { checkAgent } from './check-agent';
import { validateTemporalSandboxClient } from './sandbox-client';
import { ensureTracingProcessorRegistered } from './tracing';
import { flushOpenSpans } from './agent-sink-processor';
import { getCurrentPluginConfig } from './plugin-config-store';

export interface TemporalRunOptions<TContext = undefined> {
  /** Run context passed to agents and tools */
  context?: TContext;
  /** Maximum agent loop turns before aborting */
  maxTurns?: number;
  /** Previous OpenAI response ID for conversation continuity */
  previousResponseId?: string;
  /** OpenAI conversation ID for multi-turn persistence */
  conversationId?: string;
  /** Session state for conversation memory */
  session?: Session;
  /** Customize how session history merges with current turn input */
  sessionInputCallback?: SessionInputCallback;
  /** Edit system instructions or input items just before calling the model */
  callModelInputFilter?: CallModelInputFilter;
  /** Per-run tracing config override */
  tracing?: TracingConfig;
  /**
   * Stream incremental events as the model responds. Requires a `streamingTopic`
   * (set via the plugin's `modelParams` or the runner's `defaultModelParams`) and a
   * hosted `WorkflowStream` in the Workflow.
   *
   * @experimental Streaming support is experimental and may change without notice.
   */
  stream?: boolean;
  // signal intentionally omitted — use Temporal CancellationScope for Workflow cancellation

  /** Runner-level config overrides */
  runConfig?: {
    /** Model name override (string only — Model objects can't cross the Workflow/Activity boundary) */
    model?: string;
    /** Global model settings (temperature, topP, etc.). Non-null values override agent-specific settings. */
    modelSettings?: ModelSettings;
    /** Global handoff input filter. Agent-level inputFilter takes precedence. */
    handoffInputFilter?: (input: HandoffInputData) => HandoffInputData;
    /** Input guardrails run inline in the Workflow — callbacks must be deterministic */
    inputGuardrails?: InputGuardrail[];
    /** Output guardrails run inline in the Workflow — callbacks must be deterministic */
    outputGuardrails?: OutputGuardrail<AgentOutputType<unknown>>[];
    /** Disable tracing for this run */
    tracingDisabled?: boolean;
    /** Include sensitive data (tool I/O, LLM outputs) in trace spans */
    traceIncludeSensitiveData?: boolean;
    /** Logical name for the run, used in tracing */
    workflowName?: string;
    /** Custom trace ID */
    traceId?: string;
    /** Grouping ID for linking traces (e.g., chat thread ID) */
    groupId?: string;
    /** Additional metadata attached to the trace */
    traceMetadata?: Record<string, string>;
    /**
     * Sandbox runtime configuration used when execution reaches a `SandboxAgent`.
     * `client` must be created via `temporalSandboxClient(name)`.
     *
     * @experimental Sandbox support is experimental and may change without notice.
     */
    sandbox?: SandboxRunConfig;
  };
}

export interface TemporalOpenAIRunnerOptions {
  /**
   * Workflow-authored default Model Activity options, layered UNDER the client's
   * `modelParams` header: a client-set field wins, otherwise this default applies.
   * Because these options are constructed in the Workflow (not JSON-propagated),
   * they can carry the function form of `summary`. Use this to configure Workflows
   * started without the client interceptor, e.g. by a Schedule or the Temporal UI/CLI.
   */
  defaultModelParams?: ModelActivityOptions;
}

/**
 * Layers Model Activity options least→most significant:
 * package `DEFAULT_MODEL_ACTIVITY_OPTIONS` < constructor `defaultModelParams` <
 * client header `modelParams`. Undefined field values are dropped at each layer so
 * an absent option never clobbers a value set by a lower layer.
 */
export function layerModelParams(
  defaultModelParams: ModelActivityOptions | undefined,
  headerModelParams: SerializableModelActivityOptions | undefined
): ModelActivityOptions {
  return {
    ...DEFAULT_MODEL_ACTIVITY_OPTIONS,
    ...definedFields(defaultModelParams),
    ...definedFields(headerModelParams),
  };
}

function definedFields<T extends object>(obj: T | undefined): Partial<T> {
  if (obj === undefined) return {};
  const result: Partial<T> = {};
  for (const key of Object.keys(obj) as (keyof T)[]) {
    if (obj[key] !== undefined) result[key] = obj[key];
  }
  return result;
}

/** Whether a `SandboxAgent` is reachable from `agent` through its handoff graph. */
function hasSandboxAgent(agent: Agent<any, any>, seen: Set<Agent<any, any>> = new Set()): boolean {
  if (agent instanceof SandboxAgent) return true;
  if (seen.has(agent)) return false;
  seen.add(agent);
  for (const handoff of agent.handoffs ?? []) {
    const target = handoff instanceof Handoff ? handoff.agent : handoff;
    if (hasSandboxAgent(target, seen)) return true;
  }
  return false;
}

/** A reachable `SandboxAgent` needs `runConfig.sandbox`, with a Temporal sandbox client. */
function validateSandboxRunConfig(agent: Agent<any, any>, sandbox: SandboxRunConfig | undefined): void {
  if (!hasSandboxAgent(agent) && sandbox === undefined) return;
  if (sandbox === undefined) {
    throw ApplicationFailure.create({
      message:
        'A SandboxAgent was provided but runConfig.sandbox is not configured. ' +
        'Set runConfig.sandbox with a client created via temporalSandboxClient(name) ' +
        'from @temporalio/openai-agents/workflow.',
      type: 'SandboxConfigurationError',
      nonRetryable: true,
    });
  }
  validateTemporalSandboxClient(sandbox, 'runConfig.sandbox');
}

/**
 * A Temporal-aware agent runner that delegates model calls to Activities.
 *
 * Use `run(agent, input)` for request-response runs. Pass `{ stream: true }` to
 * stream incremental events as the model responds — the returned
 * `StreamedRunResult` is async-iterable. Streaming requires a `streamingTopic`
 * (set via the plugin's `modelParams` or the runner's `defaultModelParams`) and a
 * hosted `WorkflowStream` in the Workflow.
 */
export class TemporalOpenAIRunner {
  private readonly modelParams: ModelActivityOptions;

  constructor(options?: TemporalOpenAIRunnerOptions) {
    ensureTracingProcessorRegistered();
    const fromHeader = getCurrentPluginConfig();
    this.modelParams = layerModelParams(options?.defaultModelParams, fromHeader?.modelParams);
  }

  /**
   * `input` accepts a prompt string, structured `AgentInputItem[]`, or a
   * deserialized `RunState` to resume a previous run across `continueAsNew`.
   * Capture the current `RunState` via `result.state.toString()`.
   */
  run<TAgent extends Agent<any, any>, TContext = undefined>(
    agent: TAgent,
    input: string | AgentInputItem[] | RunState<TContext, TAgent>,
    options?: TemporalRunOptions<TContext> & { stream?: false }
  ): Promise<RunResult<any, TAgent>>;
  run<TAgent extends Agent<any, any>, TContext = undefined>(
    agent: TAgent,
    input: string | AgentInputItem[] | RunState<TContext, TAgent>,
    options: TemporalRunOptions<TContext> & { stream: true }
  ): Promise<StreamedRunResult<TContext, TAgent>>;
  async run<TAgent extends Agent<any, any>, TContext = undefined>(
    agent: TAgent,
    input: string | AgentInputItem[] | RunState<TContext, TAgent>,
    options?: TemporalRunOptions<TContext>
  ): Promise<RunResult<any, TAgent> | StreamedRunResult<TContext, TAgent>> {
    const session = options?.session;
    if (session instanceof MemorySession) {
      throw ApplicationFailure.create({
        message:
          'Pass WorkflowSafeMemorySession to TemporalOpenAIRunner.run({ session }); the upstream MemorySession is not safe inside a Workflow (heap-only state without Temporal Event History backing). See @temporalio/openai-agents/workflow.',
        type: 'UnsafeSessionError',
        nonRetryable: true,
      });
    }

    // Fail fast before the upstream streaming runner starts, so a missing topic
    // surfaces as a clean error rather than being captured into the streamed result.
    if (options?.stream === true && this.modelParams.streamingTopic === undefined) {
      throw ApplicationFailure.create({
        message: STREAMING_TOPIC_NOT_CONFIGURED.message,
        type: STREAMING_TOPIC_NOT_CONFIGURED.type,
        nonRetryable: true,
      });
    }

    if (options?.stream === true && this.modelParams.useLocalActivity === true) {
      throw ApplicationFailure.create({
        message: STREAMING_LOCAL_ACTIVITY_UNSUPPORTED.message,
        type: STREAMING_LOCAL_ACTIVITY_UNSUPPORTED.type,
        nonRetryable: true,
      });
    }

    validateSandboxRunConfig(agent, options?.runConfig?.sandbox);

    const { model: modelOverride, ...runnerConfigOverrides } = options?.runConfig ?? {};

    checkAgent(agent, modelOverride);

    let preparedInput: string | AgentInputItem[] | RunState<TContext, TAgent>;
    if (input instanceof RunState) {
      // Round-trip through fromString to drop upstream's live-session identity, which would otherwise
      // resolve manifest environment references on the Workflow thread rather than in the resume Activity.
      const restored = (await RunState.fromString(agent, input.toString())) as RunState<TContext, TAgent>;
      // Suppress upstream Runner.run's withTrace(state._trace, ...) branch; we want the current Workflow's trace.
      restored.clearTrace();
      preparedInput = restored;
    } else {
      preparedInput = input;
    }

    // The agent a ModelSummaryProvider gets. Model calls made during a tool call belong to a
    // nested `agent.asTool()` run, whose agent this runner doesn't see, so they get none.
    let runningAgent: Agent<any, any> | undefined =
      preparedInput instanceof RunState ? preparedInput._currentAgent : agent;
    let toolCalls = 0;

    // Every agent runs as written: the provider sends each model name to the model Activity.
    // Nested `agent.asTool()` runs take the model, provider and settings from this config, as upstream.
    const innerRunner = new Runner({
      ...runnerConfigOverrides,
      model: modelOverride,
      modelProvider: new ActivityBackedModelProvider(this.modelParams, () =>
        toolCalls === 0 ? runningAgent : undefined
      ),
    });
    innerRunner.on('agent_start', (_context, started) => (runningAgent = started));
    innerRunner.on('agent_handoff', (_context, _from, to) => (runningAgent = to));
    innerRunner.on('agent_tool_start', () => toolCalls++);
    innerRunner.on('agent_tool_end', () => toolCalls--);

    const sharedRunOptions = {
      maxTurns: options?.maxTurns,
      context: options?.context,
      previousResponseId: options?.previousResponseId,
      conversationId: options?.conversationId,
      session: options?.session,
      sessionInputCallback: options?.sessionInputCallback,
      callModelInputFilter: options?.callModelInputFilter,
      tracing: options?.tracing,
    };

    try {
      if (options?.stream === true) {
        return (await innerRunner.run(agent, preparedInput as any, {
          ...sharedRunOptions,
          stream: true,
        })) as StreamedRunResult<TContext, TAgent>;
      }
      return (await innerRunner.run(agent, preparedInput as any, sharedRunOptions)) as RunResult<any, TAgent>;
    } catch (error) {
      throw normalizeAgentsRunError(error);
    } finally {
      await flushOpenSpans();
    }
  }
}

/**
 * Returns the value the caller should throw: unwrapped TemporalFailure if
 * reachable via `cause` / `AggregateError.errors`, else an `AgentsWorkflowError`
 * ApplicationFailure wrapping a plain Error, else the original non-Error.
 */
export function normalizeAgentsRunError(error: unknown): unknown {
  const temporalFailure = unwrapTemporalFailure(error);
  if (temporalFailure) return temporalFailure;
  if (error instanceof Error) {
    return ApplicationFailure.create({
      message: `Agent workflow failed: ${error.message}`,
      type: 'AgentsWorkflowError',
      nonRetryable: true,
      cause: error,
    });
  }
  return error;
}
