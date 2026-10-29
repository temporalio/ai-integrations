import { type Agent, Handoff, MemorySession, type Tool } from '@openai/agents-core';
import type { SandboxRunConfig } from '@openai/agents-core/sandbox';
import { ApplicationFailure } from '@temporalio/common';
import type { ModelActivityOptions } from '../common/model-activity-options';
import { ActivityBackedModel } from './activity-backed-model';
import { type AgentToolRecord, getAgentToolRecord, isUnrecordedAgentTool, rebuildAgentTool } from './agent-tools';
import { validateTemporalSandboxClient } from './sandbox-client';

interface Conversion {
  modelParams: ModelActivityOptions;
  modelNameOverride?: string;
  seen: Map<Agent<any, any>, Agent<any, any>>;
  /** Handoff-graph agents by name, to reject a different `asTool()` agent with the same name. */
  handoffAgents: Map<string, Agent<any, any>>;
  /** Original and converted agents whose tools still need converting. */
  pendingTools: [Agent<any, any>, Agent<any, any>][];
}

/**
 * Converts an agent graph: replace each handoff-graph agent's model with an ActivityBackedModel
 * and clone handoffs, then check every tool and rebuild `agent.asTool()` tools on converted agents.
 * Uses a seen-map to handle circular references, so each declared agent has one converted agent.
 */
export function convertAgent(
  agent: Agent<any, any>,
  modelParams: ModelActivityOptions,
  seen?: Map<Agent<any, any>, Agent<any, any>>,
  modelNameOverride?: string
): Agent<any, any> {
  const conversion: Conversion = {
    modelParams,
    modelNameOverride,
    seen: seen ?? new Map(),
    handoffAgents: new Map(),
    pendingTools: [],
  };
  const converted = convertHandoffAgent(agent, conversion);
  // Tools are converted after the whole handoff graph, so an agent that is both a handoff target and
  // an `asTool()` source resolves to its handoff conversion: one agent, and one identity in RunState.
  for (let i = 0; i < conversion.pendingTools.length; i++) {
    const [original, convertedAgent] = conversion.pendingTools[i]!;
    convertedAgent.tools = (original.tools ?? []).map((t) => convertTool(t, original, conversion));
  }
  return converted;
}

function convertHandoffAgent(agent: Agent<any, any>, conversion: Conversion): Agent<any, any> {
  if (conversion.seen.has(agent)) return conversion.seen.get(agent)!;

  rejectModelObject(agent);
  const rawModel = agent.model;
  const modelName = conversion.modelNameOverride ?? (typeof rawModel === 'string' ? rawModel : undefined);
  if (modelName === undefined) {
    throw ApplicationFailure.create({
      message:
        `Agent '${agent.name}' has no model declared and no runConfig.model override given. ` +
        'Declare a model on the agent or pass runConfig.model to runner.run().',
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
  const activityBackedModel = new ActivityBackedModel(modelName, conversion.modelParams);
  // Bind to the original (pre-clone) agent so the summary provider sees the
  // user-declared name and instructions.
  activityBackedModel.setAgent(agent);

  const converted = agent.clone({ model: activityBackedModel });
  conversion.seen.set(agent, converted);
  conversion.handoffAgents.set(agent.name, agent);
  conversion.pendingTools.push([agent, converted]);

  converted.handoffs = (agent.handoffs ?? []).map((h) =>
    h instanceof Handoff
      ? h.clone({ agent: convertHandoffAgent(h.agent, conversion) })
      : convertHandoffAgent(h, conversion)
  );
  return converted;
}

/**
 * Converts an agent that runs only inside nested `asTool()` runs, and the agents it hands off to.
 * Their string models stay, as upstream: the inherited Activity-backed provider resolves them.
 */
function convertToolAgent(agent: Agent<any, any>, conversion: Conversion): Agent<any, any> {
  if (conversion.seen.has(agent)) return conversion.seen.get(agent)!;

  rejectModelObject(agent);
  if (conversion.handoffAgents.has(agent.name)) {
    // Upstream numbers agents that share a name by comparing them, models included, and converted
    // handoff agents carry an ActivityBackedModel, so saved state could resolve to the other agent.
    throw ApplicationFailure.create({
      message:
        `Agent '${agent.name}' is used through agent.asTool() and a different agent with the same name ` +
        'is a handoff target. Give the agents different names, or use the same Agent object for both.',
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
  const converted = agent.clone({});
  conversion.seen.set(agent, converted);
  conversion.pendingTools.push([agent, converted]);

  converted.handoffs = (agent.handoffs ?? []).map((h) =>
    h instanceof Handoff ? h.clone({ agent: convertToolAgent(h.agent, conversion) }) : convertToolAgent(h, conversion)
  );
  return converted;
}

function convertTool(tool: Tool<any>, owner: Agent<any, any>, conversion: Conversion): Tool<any> {
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
    return tool;
  }
  validateAgentToolOptions(record, tool.name);
  return rebuildAgentTool(record, convertToolAgent(record.agent, conversion));
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

/** Rejects `asTool()` options that would run model calls or sandbox operations in the Workflow. */
function validateAgentToolOptions({ agent, options }: AgentToolRecord, toolName: string): void {
  const where = `Agent '${agent.name}' asTool('${toolName}')`;
  const runConfig = options.runConfig;
  if (runConfig?.modelProvider !== undefined) {
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
    [options.runOptions?.sandbox, 'runOptions.sandbox'],
  ] as [SandboxRunConfig | undefined, string][]) {
    if (sandbox !== undefined) validateTemporalSandboxClient(sandbox, `${where} ${field}`);
  }
  if (options.runOptions?.session instanceof MemorySession) {
    throw ApplicationFailure.create({
      message:
        `${where} sets runOptions.session to a MemorySession, which is not safe inside a Workflow. ` +
        'Pass WorkflowSafeMemorySession from @temporalio/openai-agents/workflow.',
      type: 'UnsafeSessionError',
      nonRetryable: true,
    });
  }
}
