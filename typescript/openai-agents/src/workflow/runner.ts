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
import { CancellationScope } from '@temporalio/workflow';
import {
  DEFAULT_MODEL_ACTIVITY_OPTIONS,
  STREAMING_LOCAL_ACTIVITY_UNSUPPORTED,
  STREAMING_TOPIC_NOT_CONFIGURED,
  type ModelActivityOptions,
  type SerializableModelActivityOptions,
} from '../common/model-activity-options';
import { unwrapTemporalFailure } from '../common/errors';
import { ActivityBackedModelProvider } from './activity-backed-model';
import { getAgentToolRecord } from './agent-tools';
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
    /**
     * Model name for agents that set no model of their own, as upstream's code does (string only —
     * Model objects can't cross the Workflow/Activity boundary)
     */
    model?: string;
    /** Global model settings (temperature, topP, etc.). An agent's own `modelSettings` override them, as upstream. */
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

    // Normalized, so a malformed agent graph, such as an `undefined` tool, fails the run.
    let preparedInput: string | AgentInputItem[] | RunState<TContext, TAgent>;
    try {
      validateSandboxRunConfig(agent, options?.runConfig?.sandbox);
      checkAgent(agent);
      preparedInput = input instanceof RunState ? await prepareRunState(agent, input) : input;
    } catch (error) {
      throw normalizeAgentsRunError(error);
    }

    const { model: modelOverride, ...runnerConfigOverrides } = options?.runConfig ?? {};

    // Every agent runs as written: the provider sends each model name to the model Activity.
    // Nested `agent.asTool()` runs take the model, provider and settings from this config, as upstream.
    const innerRunner = new Runner({
      ...runnerConfigOverrides,
      model: modelOverride,
      modelProvider: new ActivityBackedModelProvider(this.modelParams),
    });

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

    // A tool can turn a CancelledFailure into its output: upstream `asTool()` returns a nested run's
    // failure to the parent model. A run that then ends on that output, for example with
    // `toolUseBehavior: 'stop_on_first_tool'`, or fails another way, ends with the cancellation.
    const scope = CancellationScope.current();
    try {
      if (options?.stream === true) {
        const streamed = (await innerRunner.run(agent, preparedInput as any, {
          ...sharedRunOptions,
          stream: true,
        })) as StreamedRunResult<TContext, TAgent>;
        // Upstream's stream loop ends through these internal methods, which its typings omit.
        const completion = streamed as unknown as { _done(): void; _raiseError(error: unknown): void };
        const done = completion._done.bind(streamed);
        const raiseError = completion._raiseError.bind(streamed);
        completion._done = () => {
          if (!scope.consideredCancelled) return done();
          scope.cancelRequested.then(done, raiseError);
        };
        completion._raiseError = (error: unknown) => {
          if (!scope.consideredCancelled) return raiseError(error);
          scope.cancelRequested.then(() => raiseError(error), raiseError);
        };
        return streamed;
      }
      const result = (await innerRunner.run(agent, preparedInput as any, sharedRunOptions)) as RunResult<any, TAgent>;
      if (scope.consideredCancelled) await scope.cancelRequested;
      return result;
    } catch (error) {
      if (scope.consideredCancelled) await scope.cancelRequested;
      throw normalizeAgentsRunError(error);
    } finally {
      await flushOpenSpans();
    }
  }
}

/**
 * Fails the run if restoring `input` would move one of its approvals to another agent. Restoring finds
 * an agent through handoffs and the tools `agent.asTool()` returned, so the approval of an agent reached
 * only through a copy of such a tool goes to the parent or to another agent of the same name, and
 * resumed it would stay pending. Checks agent objects, not names, from both `agent` and the state's
 * current agent, so a state whose agents were built again for this run is checked too.
 */
function checkRestoredInterruptions(agent: Agent<any, any>, input: RunState<any, any>): void {
  const roots = [agent, input._currentAgent];
  const restorable = collectAgents(roots, false);
  const reachable = collectAgents(roots, true);
  for (const interruption of input.getInterruptions()) {
    if (restorable.has(interruption.agent) || !reachable.has(interruption.agent)) continue;
    throw ApplicationFailure.create({
      message:
        `The approval for '${interruption.name}' of agent '${interruption.agent.name}' can't be resumed: ` +
        "this run's sandbox state is restored from its serialized form, which finds a nested agent only " +
        'through the tool its agent.asTool() returned, not a copy such as a toolNamespace() one. Give the ' +
        'parent agent the tool agent.asTool() returned.',
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
}

/**
 * The agents reached from `roots` through handoffs and `asTool()` tools: all of them, or with
 * `throughCopies` false, only the tools `asTool()` returned, as upstream's `RunState` does.
 */
function collectAgents(roots: Agent<any, any>[], throughCopies: boolean): Set<Agent<any, any>> {
  const seen = new Set<Agent<any, any>>();
  const queue = [...roots];
  for (let next = queue.shift(); next !== undefined; next = queue.shift()) {
    if (seen.has(next)) continue;
    seen.add(next);
    for (const h of next.handoffs ?? []) queue.push(h instanceof Handoff ? h.agent : h);
    for (const tool of next.tools ?? []) {
      const record = getAgentToolRecord(tool);
      if (record !== undefined && (throughCopies || record.tool === tool)) queue.push(record.agent);
    }
  }
  return seen;
}

/**
 * A state with sandbox sessions holds upstream's live-session identity, which would resolve manifest
 * environment references on the Workflow thread rather than in the resume Activity, so it's rebuilt
 * with `fromString`, which also lists tools and so runs MCP tool filters. Other states resume as they
 * are, as upstream: restoring finds an `asTool()` agent only through the tool object `asTool()`
 * returned, and not an agent that only a custom `Handoff`'s callback picked.
 */
async function prepareRunState<TContext, TAgent extends Agent<any, any>>(
  agent: TAgent,
  state: RunState<TContext, TAgent>
): Promise<RunState<TContext, TAgent>> {
  let prepared = state;
  if (state._sandbox !== undefined) {
    checkRestoredInterruptions(agent, state);
    prepared = (await RunState.fromString(agent, state.toString())) as RunState<TContext, TAgent>;
  }
  // Suppress upstream Runner.run's withTrace(state._trace, ...) branch; we want the current Workflow's trace.
  prepared.clearTrace();
  return prepared;
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
