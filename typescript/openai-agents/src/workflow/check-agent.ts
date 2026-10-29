import { Agent, Handoff, type RunContext, type Tool } from '@openai/agents-core';
import { ApplicationFailure } from '@temporalio/common';
import { inWorkflowContext } from '@temporalio/workflow';
import { type AgentToolRecord, checkAgentToolOptions, getAgentToolRecord, isUnrecordedAgentTool } from './agent-tools';

/**
 * Checks an agent graph before a run, without changing it: the agents reached through handoffs and
 * through `agent.asTool()` tools, and those tools' options. Rejects raw functions in `tools`, an agent
 * with no model when the run sets none, and setups that would run model calls or sandbox operations
 * in the Workflow. Agents then run as written: the runner's model provider sends every model name to
 * the model Activity. Uses a seen-set for circular references.
 */
export function checkAgent(agent: Agent<any, any>, modelOverride?: string, seen = new Set<Agent<any, any>>()): void {
  if (seen.has(agent)) return;
  seen.add(agent);

  rejectModelObject(agent);
  if (typeof agent.model !== 'string' && modelOverride === undefined) {
    throw ApplicationFailure.create({
      message:
        `Agent '${agent.name}' has no model declared and no runConfig.model override given. ` +
        'Declare a model on the agent or pass runConfig.model to runner.run().',
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
  for (const tool of agent.tools ?? []) checkTool(tool, agent, modelOverride, seen);
  for (const h of agent.handoffs ?? []) checkAgent(h instanceof Handoff ? h.agent : h, modelOverride, seen);
}

function checkTool(
  tool: Tool<any>,
  owner: Agent<any, any>,
  modelOverride: string | undefined,
  seen: Set<Agent<any, any>>
): void {
  const record = checkToolOptions(tool, owner);
  if (record === undefined) return;
  const nestedModel = typeof record.runConfig?.model === 'string' ? record.runConfig.model : modelOverride;
  checkAgent(record.agent, nestedModel, seen);
}

interface RunningAgentCheckState {
  original: Agent<any, any>['getSystemPrompt'];
}

const runningAgentCheckState = Symbol.for('@temporalio/openai-agents/runningAgentCheck');

/**
 * Checks each agent again just before it calls its model, in the Workflow: an agent that a custom
 * `Handoff`'s callback returns, or that an `agent.on(...)` listener changes, isn't in the graph
 * `checkAgent` sees before the run. Upstream calls `getSystemPrompt` once per model call, after it
 * picks the model and before it calls it, so the wrapper rejects a `Model` object and unsafe
 * `asTool()` tools there. The agents behind those tools are checked when they run, and the tools'
 * options again when they are invoked.
 */
export function installRunningAgentCheck(): void {
  const prototype = Agent.prototype as unknown as Record<symbol, RunningAgentCheckState | undefined>;
  if (prototype[runningAgentCheckState] !== undefined) return;
  const state: RunningAgentCheckState = { original: Agent.prototype.getSystemPrompt };
  function getSystemPrompt(this: Agent<any, any>, runContext: RunContext<any>): Promise<string | undefined> {
    if (inWorkflowContext()) {
      rejectModelObject(this);
      for (const tool of this.tools ?? []) checkToolOptions(tool, this);
    }
    return state.original.call(this, runContext);
  }
  Object.defineProperty(Agent.prototype, runningAgentCheckState, { value: state });
  Agent.prototype.getSystemPrompt = getSystemPrompt;
}

installRunningAgentCheck();

/**
 * Rejects a raw function, an unrecorded `asTool()` tool, and `asTool()` options that would run in the
 * Workflow. Returns the record of an `asTool()` tool, so the caller can check its agent.
 */
function checkToolOptions(tool: Tool<any>, owner: Agent<any, any>): AgentToolRecord | undefined {
  // Reject raw functions: authoring mistake — use tool() or activityAsTool().
  if (typeof tool === 'function') {
    throw ApplicationFailure.create({
      message:
        `Agent '${owner.name}': Provided tool is a raw function, not a tool object. ` +
        'Did you mean to use tool() or activityAsTool()?',
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
  const record = getAgentToolRecord(tool);
  if (record === undefined) {
    if (isUnrecordedAgentTool(tool)) {
      throw ApplicationFailure.create({
        message:
          `Agent '${owner.name}': tool '${tool.name}' looks like an agent.asTool() tool that ` +
          '@temporalio/openai-agents did not record, so its nested agent cannot be checked. ' +
          'Create it with agent.asTool() in Workflow code, after @temporalio/openai-agents/workflow ' +
          'is loaded and with the same @openai/agents-core copy, and pass it as is.',
        type: 'AgentsWorkflowError',
        nonRetryable: true,
      });
    }
    return undefined;
  }
  checkAgentToolOptions(record, tool.name);
  return record;
}

function rejectModelObject(agent: Agent<any, any>): void {
  const rawModel = agent.model;
  if (rawModel !== undefined && rawModel !== null && typeof rawModel !== 'string') {
    throw ApplicationFailure.create({
      message:
        `Agent '${agent.name}' has a Model object instead of a string model name. ` +
        'In Temporal workflows, all models must be specified as strings — use ' +
        'runConfig.model to override, or declare a string model on the agent.',
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
}
