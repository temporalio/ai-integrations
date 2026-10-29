import { type Agent, Handoff, MemorySession, type Tool } from '@openai/agents-core';
import type { SandboxRunConfig } from '@openai/agents-core/sandbox';
import { ApplicationFailure } from '@temporalio/common';
import { type AgentToolRecord, getAgentToolRecord, isUnrecordedAgentTool } from './agent-tools';
import { validateTemporalSandboxClient } from './sandbox-client';

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
    return;
  }
  checkAgentToolOptions(record, tool.name);
  const nestedModel = typeof record.runConfig?.model === 'string' ? record.runConfig.model : modelOverride;
  checkAgent(record.agent, nestedModel, seen);
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
function checkAgentToolOptions({ agent, runConfig, runOptions }: AgentToolRecord, toolName: string): void {
  const where = `Agent '${agent.name}' asTool('${toolName}')`;
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
