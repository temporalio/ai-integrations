import {
  type Agent,
  type AgentInputItem,
  Handoff,
  MemorySession,
  RunContext,
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

/** A converted agent graph. */
interface Conversion {
  source: Agent<any, any>;
  /** Each source agent the graph has converted, mapped to its converted agent. */
  agents: Map<Agent<any, any>, Agent<any, any>>;
  /** Runs that are using the graph. A resume converts into the graph only while no run uses it. */
  activeRuns: number;
}

/** Each converted agent, mapped to the graph it belongs to. */
const conversions = new WeakMap<Agent<any, any>, Conversion>();

/**
 * For each `RunContext`, the graph its latest resume used for each source agent. A state rehydrated onto
 * the source agent doesn't point at a converted agent, so this is how its resume finds the graph that
 * already owns the context's approvals. It's keyed by context so Workflows that share module-level agents
 * under `reuseV8Context` never share a graph.
 */
const resumeConversions = new WeakMap<object, Map<Agent<any, any>, Conversion>>();

/**
 * Converts `agent`'s graph. With `into`, the conversion reuses those of `into`'s agent objects that
 * `convertAgent` can: each one gets the fields a fresh conversion would give it, and keeps its identity.
 * `into` then maps each source agent to its latest converted agent. It keeps them even if the resume
 * fails, since the restore may already have merged the state's approvals under them.
 */
function convertAgentGraph(
  agent: Agent<any, any>,
  modelParams: ModelActivityOptions,
  modelOverride: string | undefined,
  sandbox: SandboxRunConfig | undefined,
  into?: Conversion
): Conversion {
  const convertedBySource = new Map<Agent<any, any>, Agent<any, any>>();
  const converted = convertAgent(agent, modelParams, convertedBySource, modelOverride, into?.agents);
  if (sandbox !== undefined) {
    propagateAgentToolSandboxConfig(converted, sandbox);
  }
  const conversion = into ?? { source: agent, agents: new Map(), activeRuns: 0 };
  for (const [source, convertedAgent] of convertedBySource) {
    conversion.agents.set(source, convertedAgent);
    conversions.set(convertedAgent, conversion);
  }
  return conversion;
}

/**
 * Returns the converted graph that `state` runs on, or else the one the latest resume on its context used,
 * if it was converted from `agent`, no run is using it, and restoring the state into it keeps the state's
 * decisions and rejection messages.
 *
 * Function-tool approvals are keyed by agent object on the state's `RunContext`, which the resume keeps.
 * Rebuilding the state over new agent objects would add a copy of its approvals under every resume's
 * graph, and the earlier copies can't be dropped, because saved states that still point at those agents
 * read them.
 */
function reusableConversion(
  state: RunState<any, any>,
  agent: Agent<any, any>,
  serializedState: string
): Conversion | undefined {
  const candidates = [
    conversions.get(state._currentAgent as Agent<any, any>),
    resumeConversions.get(state._context)?.get(agent),
  ];
  return candidates.find(
    (conversion) =>
      conversion?.source === agent &&
      conversion.activeRuns === 0 &&
      !losesStateDecisions(conversion, serializedState, state._context)
  );
}

/** A function-tool approval record, as a serialized `RunState` and `RunContext` store it. */
interface ApprovalRecord {
  approved: boolean | string[];
  rejected: boolean | string[];
  messages?: Record<string, string>;
  stickyRejectMessage?: string;
}

interface FunctionApprovals {
  agentIdentity: string;
  approvals: Record<string, ApprovalRecord>;
}

/** The `RunContext` internals of @openai/agents-core that read and merge per-agent approvals. */
interface ApprovalContext {
  _toJSONForRunState(keys: ReadonlyMap<object, string>): { functionApprovals?: FunctionApprovals[] };
  _mergeFunctionApprovals(approvals: FunctionApprovals[], agentsByIdentity: ReadonlyMap<string, object>): void;
}

/**
 * Whether the restore would lose one of the serialized state's decisions or rejection messages if it merged
 * them into the decisions `conversion`'s agents already own. The merge keeps an approval for good over a
 * rejection, and an existing rejection message over the state's. So a state that always approved a tool
 * and then always rejected it would run it again, and a state whose rejection message changed after a
 * failed resume would send the model the earlier message. A fresh graph owns only the state's decisions.
 */
function losesStateDecisions(conversion: Conversion, serializedState: string, context: object): boolean {
  const stateApprovals: FunctionApprovals[] = JSON.parse(serializedState).context.functionApprovals ?? [];
  if (stateApprovals.length === 0) return false;
  const names = new Map<Agent<any, any>, string>();
  for (const [source, converted] of conversion.agents) names.set(converted, source.name);
  const { functionApprovals: owned = [] } = (context as ApprovalContext)._toJSONForRunState(names);
  // A state's agent identity is its name, followed by `#n` if other agents in the graph share it.
  return stateApprovals.some(({ agentIdentity, approvals }) =>
    owned.some(
      ({ agentIdentity: name, approvals: ownedApprovals }) =>
        (agentIdentity === name || agentIdentity.startsWith(`${name}#`)) &&
        Object.entries(approvals).some(
          ([toolName, record]) =>
            ownedApprovals[toolName] !== undefined &&
            !keepsDecisions(record, mergeApprovalRecords(ownedApprovals[toolName], record, toolName))
        )
    )
  );
}

/** Merges `incoming` into `current` as the restore does, in a scratch `RunContext`. */
function mergeApprovalRecords(current: ApprovalRecord, incoming: ApprovalRecord, toolName: string): ApprovalRecord {
  const owner = {};
  const scratch = new RunContext() as unknown as ApprovalContext;
  for (const record of [current, incoming]) {
    scratch._mergeFunctionApprovals(
      [{ agentIdentity: 'owner', approvals: { [toolName]: record } }],
      new Map([['owner', owner]])
    );
  }
  return scratch._toJSONForRunState(new Map([[owner, 'owner']])).functionApprovals![0]!.approvals[toolName]!;
}

/** Whether `merged` resolves every call that `record` decides, and its rejection messages, as `record` does. */
function keepsDecisions(record: ApprovalRecord, merged: ApprovalRecord): boolean {
  const callIds = new Set<string | undefined>([
    undefined,
    ...(Array.isArray(record.approved) ? record.approved : []),
    ...(Array.isArray(record.rejected) ? record.rejected : []),
    ...Object.keys(record.messages ?? {}),
  ]);
  return [...callIds].every((callId) => {
    const decision = approvalDecision(record, callId);
    if (decision === undefined) return true;
    if (approvalDecision(merged, callId) !== decision) return false;
    return decision || rejectionMessage(merged, callId) === rejectionMessage(record, callId);
  });
}

/** The decision a record gives a call, or a call it doesn't name if `callId` is undefined, as agents-core resolves it. */
function approvalDecision(record: ApprovalRecord, callId: string | undefined): boolean | undefined {
  if (record.approved === true) return true;
  if (record.rejected === true) return false;
  if (callId !== undefined && Array.isArray(record.approved) && record.approved.includes(callId)) return true;
  if (callId !== undefined && Array.isArray(record.rejected) && record.rejected.includes(callId)) return false;
  return undefined;
}

/** The rejection message a record gives a call, as agents-core resolves it. */
function rejectionMessage(record: ApprovalRecord, callId: string | undefined): string | undefined {
  if (callId !== undefined && typeof record.messages?.[callId] === 'string') return record.messages[callId];
  return record.rejected === true ? record.stickyRejectMessage : undefined;
}

function recordResumeConversion(state: RunState<any, any>, agent: Agent<any, any>, conversion: Conversion): void {
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
    // Serialize the state before converting, from the graph it was saved on, which the conversion changes.
    const serializedState = input instanceof RunState ? input.toString() : undefined;
    const conversion = convertAgentGraph(
      agent,
      this.modelParams,
      modelOverride,
      sandbox,
      input instanceof RunState ? reusableConversion(input, agent, serializedState!) : undefined
    );
    const converted = conversion.agents.get(agent)!;

    conversion.activeRuns++;
    // A streamed run keeps using the graph after `run` returns, until its result completes.
    let releasedOnCompletion = false;
    try {
      let preparedInput: string | AgentInputItem[] | RunState<TContext, TAgent>;
      if (input instanceof RunState) {
        // Record before restoring, since the restore merges the state's approvals under this graph even if
        // the resume then fails.
        recordResumeConversion(input, agent, conversion);
        // Round-trip through fromString so the rehydrated state's agent graph carries our converted
        // agent (with ActivityBackedModel); setCurrentAgent only swaps the top-level ref, not nested ones.
        // It also drops upstream's live-session identity, which would otherwise resolve manifest
        // environment references on the Workflow thread rather than in the resume Activity.
        // Pass the state's own context, as the upstream Runner uses it, so resumed tools write the
        // caller's object rather than a JSON copy.
        const restored = (await RunState.fromStringWithContext(
          converted,
          serializedState!,
          input._context
        )) as RunState<TContext, TAgent>;
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
          const result = (await innerRunner.run(converted, preparedInput as any, {
            ...sharedRunOptions,
            stream: true,
          })) as StreamedRunResult<TContext, TAgent>;
          const release = () => {
            conversion.activeRuns--;
          };
          result.completed.then(release, release);
          releasedOnCompletion = true;
          return result;
        }
        return (await innerRunner.run(converted, preparedInput as any, sharedRunOptions)) as RunResult<any, TAgent>;
      } catch (error) {
        throw normalizeAgentsRunError(error);
      } finally {
        await flushOpenSpans();
      }
    } finally {
      if (!releasedOnCompletion) conversion.activeRuns--;
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
