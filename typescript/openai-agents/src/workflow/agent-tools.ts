import { Agent, MemorySession, type AgentTool, type AgentToolOptions, type Tool } from '@openai/agents-core';
import type { SandboxRunConfig } from '@openai/agents-core/sandbox';
import { ApplicationFailure } from '@temporalio/common';
import { inWorkflowContext } from '@temporalio/workflow';
import { validateTemporalSandboxClient } from './sandbox-client';

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
 *
 * Upstream keeps the `runConfig` and `runOptions` objects and reads them only when the tool runs,
 * so a listener or `inputBuilder` can change them after the run's checks. The tool checks them
 * again when it is invoked, and after `inputBuilder`, the last caller code before upstream reads them.
 */
export function installAgentToolInterception(): void {
  if (getState() !== undefined) return;
  const state: InterceptionState = { original: Agent.prototype.asTool as any, records: new WeakMap() };
  function asTool(this: Agent<any, any>, options: AnyAgentToolOptions): AnyAgentTool {
    // Upstream reads its options when the tool is made, so record them then too.
    const record: AgentToolRecord = { agent: this, runConfig: options.runConfig, runOptions: options.runOptions };
    const { inputBuilder } = options;
    const tool = state.original.call(
      this,
      inputBuilder === undefined
        ? options
        : {
            ...options,
            inputBuilder: async (...args: Parameters<typeof inputBuilder>) => {
              const input = await inputBuilder(...args);
              checkInWorkflow(record, tool.name);
              return input;
            },
          }
    );
    const invoke = tool.invoke;
    tool.invoke = function (this: unknown, ...args: Parameters<typeof invoke>) {
      checkInWorkflow(record, tool.name);
      return invoke.apply(this, args);
    };
    state.records.set(tool, record);
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

function checkInWorkflow(record: AgentToolRecord, toolName: string): void {
  if (inWorkflowContext()) checkAgentToolOptions(record, toolName);
}

/** Rejects `asTool()` options that would run model calls or sandbox operations in the Workflow. */
export function checkAgentToolOptions({ agent, runConfig, runOptions }: AgentToolRecord, toolName: string): void {
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
