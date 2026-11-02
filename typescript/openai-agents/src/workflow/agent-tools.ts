import {
  Agent,
  MemorySession,
  type AgentTool,
  type AgentToolOptions,
  type FunctionTool,
  type RunResult,
  type Tool,
} from '@openai/agents-core';
import type { SandboxRunConfig } from '@openai/agents-core/sandbox';
import { ApplicationFailure } from '@temporalio/common';
import { validateTemporalSandboxClient } from './sandbox-client';

type AnyAgentTool = AgentTool<any, any, any>;
type AnyAgentToolOptions = AgentToolOptions<any, any, any>;

/** The agent behind an `agent.asTool()` tool, and the run options the tool took when it was made. */
export interface AgentToolRecord {
  agent: Agent<any, any>;
  /** The tool `asTool()` returned. Its copies share the record, but upstream's `RunState` doesn't know them. */
  tool: AnyAgentTool;
  runConfig: AnyAgentToolOptions['runConfig'];
  runOptions: AnyAgentToolOptions['runOptions'];
}

interface InterceptionState {
  original: (this: Agent<any, any>, options: AnyAgentToolOptions) => AnyAgentTool;
  // Keyed by the tool's `on` closure, which copies keep, such as the ones `toolNamespace()` makes.
  records: WeakMap<object, AgentToolRecord>;
}

// Kept on the prototype under a registry symbol, so a second copy of this module (or a second
// install) reuses the same wrapper and records instead of wrapping `asTool` twice.
const interceptionState = Symbol.for('@temporalio/openai-agents/agentToolInterception');

function getState(): InterceptionState | undefined {
  return (Agent.prototype as unknown as Record<symbol, InterceptionState | undefined>)[interceptionState];
}

/**
 * Wraps `Agent.prototype.asTool` to record each tool's agent. Upstream keeps that link private, and
 * the runner needs it to check nested agents before a run. The plugin's Workflow interceptor module
 * installs it before the Workflow module loads; loading this module, which the runner imports,
 * installs it too.
 *
 * Upstream keeps the `runConfig` and `runOptions` objects and reads them only when the tool runs, so
 * a listener, `inputBuilder` or schema can change them after these checks. The nested `Runner`'s run
 * checks what it actually uses (see `installRunnerCheck`).
 */
export function installAgentToolInterception(): void {
  if (getState() !== undefined) return;
  const state: InterceptionState = { original: Agent.prototype.asTool as any, records: new WeakMap() };
  function asTool(this: Agent<any, any>, options: AnyAgentToolOptions): AnyAgentTool {
    // Upstream reads its options when the tool is made, so record them then too.
    const tool = state.original.call(this, options);
    state.records.set(tool.on, { agent: this, tool, runConfig: options.runConfig, runOptions: options.runOptions });
    return tool;
  }
  Object.defineProperty(Agent.prototype, interceptionState, { value: state });
  Agent.prototype.asTool = asTool as typeof Agent.prototype.asTool;
}

installAgentToolInterception();

export function getAgentToolRecord(tool: Tool<any>): AgentToolRecord | undefined {
  const on = (tool as Partial<AnyAgentTool>).on;
  return typeof on === 'function' ? getState()?.records.get(on) : undefined;
}

/**
 * Whether `tool` has the shape of an `agent.asTool()` tool that was not recorded: created before
 * the interception was installed, or made by another copy of `@openai/agents-core`.
 */
export function isUnrecordedAgentTool(tool: Tool<any>): boolean {
  return (
    tool.type === 'function' &&
    typeof (tool as Partial<AnyAgentTool>).on === 'function' &&
    getAgentToolRecord(tool) === undefined
  );
}

/** Rejects `asTool()` options that would run model calls or sandbox operations in the Workflow. */
export function checkAgentToolOptions({ agent, runConfig, runOptions }: AgentToolRecord, toolName: string): void {
  const where = `Agent '${agent.name}' asTool('${toolName}')`;
  // Upstream spreads runConfig over the inherited config, so even `modelProvider: undefined`
  // replaces the runner's provider.
  if (runConfig !== undefined && Object.prototype.hasOwnProperty.call(runConfig, 'modelProvider')) {
    throw ApplicationFailure.create({
      message:
        `${where} sets runConfig.modelProvider. Nested runs must use the runner's provider, which ` +
        'sends model calls to Activities; set a string runConfig.model instead.',
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
  if (runConfig?.model !== undefined && typeof runConfig.model !== 'string') {
    throw ApplicationFailure.create({
      message:
        `${where} sets runConfig.model to a Model object. In Temporal workflows, all models must be ` +
        'specified as strings.',
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
  for (const [sandbox, field] of [
    [runConfig?.sandbox, 'runConfig.sandbox'],
    [runOptions?.sandbox, 'runOptions.sandbox'],
  ] as [SandboxRunConfig | undefined, string][]) {
    if (sandbox !== undefined) validateTemporalSandboxClient(sandbox, `${where} ${field}`);
  }
  if (runOptions?.session instanceof MemorySession) {
    throw ApplicationFailure.create({
      message:
        `${where} sets runOptions.session to a MemorySession, which is not safe inside a Workflow. ` +
        'Pass WorkflowSafeMemorySession from @temporalio/openai-agents/workflow.',
      type: 'UnsafeSessionError',
      nonRetryable: true,
    });
  }
}

/** @deprecated Use the options of `agent.asTool()`. */
export interface AgentAsToolOptions<TAgent extends Agent<any, any>> {
  toolName: string;
  toolDescription?: string;
  customOutputExtractor?: (result: RunResult<any, TAgent>) => string | Promise<string>;
}

/**
 * @deprecated Use `agent.asTool({ toolName, toolDescription, customOutputExtractor })`. This returns
 * that tool, and will be removed in a later release.
 */
export function agentAsTool<TAgent extends Agent<any, any>>(
  agent: TAgent,
  // eslint-disable-next-line @typescript-eslint/no-deprecated -- the alias takes its own deprecated options
  options: AgentAsToolOptions<TAgent>
): FunctionTool<any, any, any> {
  return agent.asTool({
    toolName: options.toolName,
    toolDescription: options.toolDescription,
    customOutputExtractor: options.customOutputExtractor as AnyAgentToolOptions['customOutputExtractor'],
  });
}
