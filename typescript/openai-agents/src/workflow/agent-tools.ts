import { Agent, type AgentTool, type AgentToolOptions, type FunctionTool, type Tool } from '@openai/agents-core';

type AnyAgentTool = AgentTool<any, any, any>;
type AgentToolOn = AnyAgentTool['on'];

/** The agent, options and `on()` handlers behind an `agent.asTool()` tool. */
export interface AgentToolRecord {
  agent: Agent<any, any>;
  options: AgentToolOptions<any, any, any>;
  handlers: Parameters<AgentToolOn>[];
}

interface InterceptionState {
  original: (this: Agent<any, any>, options: AgentToolOptions<any, any, any>) => AnyAgentTool;
  records: WeakMap<object, AgentToolRecord>;
}

// Kept on the prototype under a registry symbol, so a second copy of this module (or a second
// install) reuses the same wrapper and records instead of wrapping `asTool` twice.
const interceptionState = Symbol.for('@temporalio/openai-agents/agentToolInterception');

function getState(): InterceptionState | undefined {
  return (Agent.prototype as unknown as Record<symbol, InterceptionState | undefined>)[interceptionState];
}

/**
 * Wraps `Agent.prototype.asTool` to record each tool's agent and options. Upstream keeps that link
 * private, and the runner needs it to check nested agents and rebuild their tools on the converted
 * agents. The plugin's Workflow interceptor module installs it before the Workflow module loads;
 * loading this module, which the runner imports, installs it too.
 */
export function installAgentToolInterception(): void {
  if (getState() !== undefined) return;
  const state: InterceptionState = { original: Agent.prototype.asTool as any, records: new WeakMap() };
  function asTool(this: Agent<any, any>, options: AgentToolOptions<any, any, any>): AnyAgentTool {
    const tool = state.original.call(this, options);
    const record: AgentToolRecord = { agent: this, options, handlers: [] };
    state.records.set(tool, record);
    const on = tool.on;
    tool.on = (name, handler) => {
      record.handlers.push([name, handler]);
      on(name, handler);
      return tool;
    };
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

/**
 * Builds the tool again with upstream `asTool()` on `agent`, so upstream links it to `agent` and
 * runs `agent`. The rebuilt tool is not recorded: it belongs to a converted graph.
 */
export function rebuildAgentTool(record: AgentToolRecord, agent: Agent<any, any>): FunctionTool<any, any> {
  const tool = getState()!.original.call(agent, record.options);
  for (const [name, handler] of record.handlers) tool.on(name, handler);
  return tool;
}
