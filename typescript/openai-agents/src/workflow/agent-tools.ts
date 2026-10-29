import { Agent, type AgentTool, type AgentToolOptions, type FunctionTool, type Tool } from '@openai/agents-core';

type AnyAgentTool = AgentTool<any, any, any>;
type AgentToolOn = AnyAgentTool['on'];

/** The agent, options and `on()` handlers behind an `agent.asTool()` tool. */
export interface AgentToolRecord {
  agent: Agent<any, any>;
  /**
   * The options as `asTool()` read them: a copy taken when the tool was created, with the tool name
   * it resolved. Later changes to the caller's options object or the agent's name don't apply, as upstream.
   */
  options: AgentToolOptions<any, any, any>;
  handlers: Parameters<AgentToolOn>[];
  /** Rebuilt tools with a nested run in progress, by number of runs, to pass them later `on()` handlers. */
  running: Map<AnyAgentTool, number>;
}

interface InterceptionState {
  original: (this: Agent<any, any>, options: AgentToolOptions<any, any, any>) => AnyAgentTool;
  records: WeakMap<object, AgentToolRecord>;
  /** The record of each rebuilt tool. */
  rebuilt: WeakMap<object, AgentToolRecord>;
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
  const state: InterceptionState = {
    original: Agent.prototype.asTool as any,
    records: new WeakMap(),
    rebuilt: new WeakMap(),
  };
  function asTool(this: Agent<any, any>, options: AgentToolOptions<any, any, any>): AnyAgentTool {
    const tool = state.original.call(this, options);
    const record: AgentToolRecord = {
      agent: this,
      options: { ...options, toolName: tool.name },
      handlers: [],
      running: new Map(),
    };
    state.records.set(tool, record);
    const on = tool.on;
    tool.on = (name, handler) => {
      record.handlers.push([name, handler]);
      on(name, handler);
      for (const rebuilt of record.running.keys()) rebuilt.on(name, handler);
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
 * runs `agent`. The rebuilt tool is not recorded: it belongs to a converted graph. The converted
 * agent that lists it must be passed to {@link forwardAgentToolHandlers}.
 */
export function rebuildAgentTool(
  record: AgentToolRecord,
  options: AgentToolOptions<any, any, any>,
  agent: Agent<any, any>
): FunctionTool<any, any> {
  const state = getState()!;
  const tool = state.original.call(agent, options);
  state.rebuilt.set(tool, record);
  for (const [name, handler] of record.handlers) tool.on(name, handler);
  return tool;
}

/**
 * Passes `on()` handlers registered on the declared tool after conversion to its rebuilt tools in
 * `owner`: at each call's `agent_tool_start`, before upstream decides whether the nested run streams,
 * and while the nested run is in progress. Upstream's handler sets ignore handlers added twice.
 */
export function forwardAgentToolHandlers(owner: Agent<any, any>): void {
  const rebuilt = getState()!.rebuilt;
  owner.on('agent_tool_start', (_context, tool) => {
    const record = rebuilt.get(tool);
    if (record === undefined) return;
    const agentTool = tool as AnyAgentTool;
    for (const [name, handler] of record.handlers) agentTool.on(name, handler);
    record.running.set(agentTool, (record.running.get(agentTool) ?? 0) + 1);
  });
  owner.on('agent_tool_end', (_context, tool) => {
    const record = rebuilt.get(tool);
    const count = record?.running.get(tool as AnyAgentTool);
    if (record === undefined || count === undefined) return;
    if (count > 1) record.running.set(tool as AnyAgentTool, count - 1);
    else record.running.delete(tool as AnyAgentTool);
  });
}
