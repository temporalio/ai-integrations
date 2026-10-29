import { Agent, type AgentTool, type AgentToolOptions, type Tool } from '@openai/agents-core';

type AnyAgentTool = AgentTool<any, any, any>;
type AnyAgentToolOptions = AgentToolOptions<any, any, any>;

/** The agent behind an `agent.asTool()` tool, and the run options the tool took when it was made. */
export interface AgentToolRecord {
  agent: Agent<any, any>;
  runConfig: AnyAgentToolOptions['runConfig'];
  runOptions: AnyAgentToolOptions['runOptions'];
}

interface InterceptionState {
  original: (this: Agent<any, any>, options: AnyAgentToolOptions) => AnyAgentTool;
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
 */
export function installAgentToolInterception(): void {
  if (getState() !== undefined) return;
  const state: InterceptionState = { original: Agent.prototype.asTool as any, records: new WeakMap() };
  function asTool(this: Agent<any, any>, options: AnyAgentToolOptions): AnyAgentTool {
    const tool = state.original.call(this, options);
    // Upstream reads its options when the tool is made, so record them then too.
    state.records.set(tool, { agent: this, runConfig: options.runConfig, runOptions: options.runOptions });
    return tool;
  }
  Object.defineProperty(Agent.prototype, interceptionState, { value: state });
  Agent.prototype.asTool = asTool as typeof Agent.prototype.asTool;
}

installAgentToolInterception();

export function getAgentToolRecord(tool: Tool<any>): AgentToolRecord | undefined {
  return getState()?.records.get(tool);
}

/**
 * Whether `tool` has the shape of an `agent.asTool()` tool that was not recorded: created before
 * the interception was installed, copied, or made by another copy of `@openai/agents-core`.
 */
export function isUnrecordedAgentTool(tool: Tool<any>): boolean {
  return (
    tool.type === 'function' &&
    typeof (tool as Partial<AnyAgentTool>).on === 'function' &&
    getAgentToolRecord(tool) === undefined
  );
}
