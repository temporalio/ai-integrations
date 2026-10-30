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
import { convertAgent } from './convert-agent';
import { TemporalSandboxClient } from './sandbox-client';
import { ensureTracingProcessorRegistered } from './tracing';
import { flushOpenSpans } from './agent-sink-processor';
import { getCurrentPluginConfig } from './plugin-config-store';

const agentAsToolSandboxConfig = Symbol.for('@temporalio/openai-agents/agentAsToolSandboxConfig');

/** A `convertAgent` call, with the inputs that decide whether a later run would convert the same graph. */
interface Conversion {
  source: Agent<any, any>;
  converted: Agent<any, any>;
  modelParams: ModelActivityOptions;
  modelOverride: string | undefined;
  sandbox: SandboxRunConfig | undefined;
  /** Each source agent in the graph, with its fields when it was converted. */
  sourceFields: Map<Agent<any, any>, [string, unknown][]>;
}

/** Each converted agent, mapped to the conversion that produced it. */
const conversions = new WeakMap<Agent<any, any>, Conversion>();

/**
 * For each `RunContext`, the conversion its latest resume used for each source agent. A state rehydrated
 * onto the source agent doesn't point at a converted agent, so this is how its resume finds the graph that
 * already owns the context's approvals. It's keyed by context so Workflows that share module-level agents
 * under `reuseV8Context` never share a graph.
 */
const resumeConversions = new WeakMap<object, Map<Agent<any, any>, Conversion>>();

/**
 * The `SandboxAgent` fields that `SandboxAgent.clone` copies by value (`cloneManifest`, and a new user object
 * for `runAs`), so a change inside them doesn't reach a converted graph. `Agent.clone` and `Handoff.clone`
 * share every other field by reference.
 */
const sandboxAgentFieldsCopiedByValue = new Set(['defaultManifest', 'runAs']);

/** A deep copy of `value`'s contents that `sameValue` compares member by member. Functions stay by identity. */
function contentSnapshot(value: unknown, ancestors: object[] = []): unknown {
  if (typeof value !== 'object' || value === null) return value;
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
    const bytes = ArrayBuffer.isView(value)
      ? new Uint8Array(value.buffer, value.byteOffset, value.byteLength)
      : new Uint8Array(value);
    return { prototype: Object.getPrototypeOf(value), bytes: bytes.slice() };
  }
  const cycle = ancestors.indexOf(value);
  if (cycle !== -1) return { cycle };
  ancestors.push(value);
  const entries =
    value instanceof Map ? [...value] : value instanceof Set ? [...value].entries() : Object.entries(value);
  const snapshot = {
    prototype: Object.getPrototypeOf(value),
    members: [...entries].map(([key, member]) => [contentSnapshot(key, ancestors), contentSnapshot(member, ancestors)]),
  };
  ancestors.pop();
  return snapshot;
}

/** Snapshots an agent's fields, including those of each `Handoff`, which `convertAgent` clones. */
function agentFields(agent: Agent<any, any>): [string, unknown][] {
  return Object.entries(agent).map(([key, value]) => [
    key,
    agent instanceof SandboxAgent && sandboxAgentFieldsCopiedByValue.has(key)
      ? contentSnapshot(value)
      : Array.isArray(value)
        ? value.map((member) =>
            member instanceof Handoff ? { handoff: member, fields: Object.entries(member) } : member
          )
        : value,
  ]);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

/** Compares arrays, byte arrays and plain objects by their members, and any other value by identity. */
function sameValue(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (a instanceof Uint8Array && b instanceof Uint8Array) {
    return a.length === b.length && a.every((byte, index) => byte === b[index]);
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((value, index) => sameValue(value, b[index]));
  }
  if (isPlainObject(a) && isPlainObject(b)) {
    const keys = Object.keys(a);
    return keys.length === Object.keys(b).length && keys.every((key) => key in b && sameValue(a[key], b[key]));
  }
  return false;
}

function convertAgentGraph(
  agent: Agent<any, any>,
  modelParams: ModelActivityOptions,
  modelOverride: string | undefined,
  sandbox: SandboxRunConfig | undefined
): Agent<any, any> {
  const convertedBySource = new Map<Agent<any, any>, Agent<any, any>>();
  const converted = convertAgent(agent, modelParams, convertedBySource, modelOverride);
  if (sandbox !== undefined) {
    propagateAgentToolSandboxConfig(converted, sandbox);
  }
  const conversion: Conversion = {
    source: agent,
    converted,
    modelParams,
    modelOverride,
    sandbox,
    sourceFields: new Map([...convertedBySource.keys()].map((source) => [source, agentFields(source)])),
  };
  for (const convertedAgent of convertedBySource.values()) conversions.set(convertedAgent, conversion);
  return converted;
}

function convertsTheSame(
  conversion: Conversion | undefined,
  agent: Agent<any, any>,
  modelParams: ModelActivityOptions,
  modelOverride: string | undefined,
  sandbox: SandboxRunConfig | undefined
): conversion is Conversion {
  if (
    conversion === undefined ||
    conversion.source !== agent ||
    conversion.modelOverride !== modelOverride ||
    !sameValue(conversion.sandbox, sandbox) ||
    !sameValue(conversion.modelParams, modelParams)
  ) {
    return false;
  }
  for (const [source, fields] of conversion.sourceFields) {
    if (!sameValue(fields, agentFields(source))) return false;
  }
  return true;
}

/**
 * Returns the converted graph that `state` runs on, or else the one the latest resume on its context used,
 * if converting `agent` again would give the same graph.
 *
 * Function-tool approvals are keyed by agent object on the state's `RunContext`, which the resume keeps.
 * Rebuilding the state over a new graph would add a copy of its approvals under every resume's graph, and
 * the earlier copies can't be dropped, because saved states that still point at those agents read them.
 */
function reusableConversion(
  state: RunState<any, any>,
  agent: Agent<any, any>,
  modelParams: ModelActivityOptions,
  modelOverride: string | undefined,
  sandbox: SandboxRunConfig | undefined
): Agent<any, any> | undefined {
  const candidates = [
    conversions.get(state._currentAgent as Agent<any, any>),
    resumeConversions.get(state._context)?.get(agent),
  ];
  return candidates.find((conversion) => convertsTheSame(conversion, agent, modelParams, modelOverride, sandbox))
    ?.converted;
}

function recordResumeConversion(state: RunState<any, any>, agent: Agent<any, any>, converted: Agent<any, any>): void {
  const conversion = conversions.get(converted)!;
  let byAgent = resumeConversions.get(state._context);
  if (byAgent === undefined) {
    byAgent = new Map();
    resumeConversions.set(state._context, byAgent);
  }
  byAgent.set(agent, conversion);
}

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

/**
 * `runConfig.sandbox.client` must be a `TemporalSandboxClient` so every sandbox
 * operation is dispatched as an Activity rather than run inline in the Workflow.
 */
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
  if (sandbox.client == null) {
    throw ApplicationFailure.create({
      message:
        'runConfig.sandbox.client must be set to a Temporal sandbox client. ' +
        'Use temporalSandboxClient(name) from @temporalio/openai-agents/workflow.',
      type: 'SandboxConfigurationError',
      nonRetryable: true,
    });
  }
  if (!(sandbox.client instanceof TemporalSandboxClient)) {
    throw ApplicationFailure.create({
      message:
        'runConfig.sandbox.client must be created via temporalSandboxClient(name) ' +
        'from @temporalio/openai-agents/workflow. Do not pass a raw sandbox client directly.',
      type: 'SandboxConfigurationError',
      nonRetryable: true,
    });
  }
}

function propagateAgentToolSandboxConfig(
  agent: Agent<any, any>,
  sandbox: SandboxRunConfig,
  seen: Set<Agent<any, any>> = new Set()
): void {
  if (seen.has(agent)) return;
  seen.add(agent);
  agent.tools = agent.tools.map((tool) => {
    if (tool.type !== 'function') return tool;
    const configureSandbox = (
      tool as typeof tool & { [agentAsToolSandboxConfig]?: (sandbox: SandboxRunConfig) => typeof tool }
    )[agentAsToolSandboxConfig];
    if (typeof configureSandbox !== 'function') return tool;
    return configureSandbox(sandbox);
  });
  for (const handoff of agent.handoffs ?? []) {
    propagateAgentToolSandboxConfig(handoff instanceof Handoff ? handoff.agent : handoff, sandbox, seen);
  }
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

    const sandbox = options?.runConfig?.sandbox;
    const converted =
      (input instanceof RunState
        ? reusableConversion(input, agent, this.modelParams, modelOverride, sandbox)
        : undefined) ?? convertAgentGraph(agent, this.modelParams, modelOverride, sandbox);

    let preparedInput: string | AgentInputItem[] | RunState<TContext, TAgent>;
    if (input instanceof RunState) {
      // Record before restoring, since the restore merges the state's approvals under this graph even if
      // the resume then fails.
      recordResumeConversion(input, agent, converted);
      // Round-trip through fromString so the rehydrated state's agent graph carries our converted
      // agent (with ActivityBackedModel); setCurrentAgent only swaps the top-level ref, not nested ones.
      // It also drops upstream's live-session identity, which would otherwise resolve manifest
      // environment references on the Workflow thread rather than in the resume Activity.
      // Pass the state's own context, as the upstream Runner uses it, so resumed tools write the
      // caller's object rather than a JSON copy.
      const restored = (await RunState.fromStringWithContext(converted, input.toString(), input._context)) as RunState<
        TContext,
        TAgent
      >;
      // Suppress upstream Runner.run's withTrace(state._trace, ...) branch; we want the current Workflow's trace.
      restored.clearTrace();
      preparedInput = restored;
    } else {
      preparedInput = input;
    }

    const innerRunner = new Runner({ ...runnerConfigOverrides });

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
        return (await innerRunner.run(converted, preparedInput as any, {
          ...sharedRunOptions,
          stream: true,
        })) as StreamedRunResult<TContext, TAgent>;
      }
      return (await innerRunner.run(converted, preparedInput as any, sharedRunOptions)) as RunResult<any, TAgent>;
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
