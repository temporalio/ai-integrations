import type { Agent, FunctionTool, RunContext, RunResult } from '@openai/agents-core';
import { ApplicationFailure } from '@temporalio/common';
import { patched } from '@temporalio/workflow';
import { agentToolRunner, type AgentToolParentRun, type TemporalRunOptions } from './runner';
import { ToolSerializationError } from './tools';

export interface AgentAsToolOptions<TAgent extends Agent<any, any>> {
  toolName: string;
  toolDescription?: string;
  customOutputExtractor?: (result: RunResult<any, TAgent>) => string | Promise<string>;
}

const agentAsToolParentRun = Symbol.for('@temporalio/openai-agents/agentAsToolParentRun');

// Nested runs recorded before agentAsTool took the run's model, model settings and runner defaults
// replay without them.
const AGENT_TOOL_RUN_CONFIG_PATCH = 'openai-agents.agent-tool-run-config';

type TemporalAgentTool = FunctionTool & {
  [agentAsToolParentRun]: (parent: AgentToolParentRun) => TemporalAgentTool;
};

/**
 * The nested run takes the calling run's sandbox, its `runConfig.model` (for
 * agents that set no model), its `runConfig.modelSettings` (without `toolChoice`
 * and `parallelToolCalls`) and its runner's `defaultModelParams`, as upstream
 * `agent.asTool()` does.
 *
 * Nested approval interruptions are not supported: `invoke` throws
 * `ApplicationFailure` of type `NestedAgentInterruption`. Handle approvals
 * at the top-level runner instead.
 */
export function agentAsTool<TAgent extends Agent<any, any>>(
  agent: TAgent,
  options: AgentAsToolOptions<TAgent>
): FunctionTool {
  const createTool = (parent?: AgentToolParentRun): TemporalAgentTool =>
    ({
      type: 'function',
      name: options.toolName,
      description: options.toolDescription ?? '',
      parameters: {
        type: 'object',
        properties: { input: { type: 'string' } },
        required: ['input'],
        additionalProperties: false,
      },
      strict: true,
      async invoke(_runContext: RunContext<any>, input: string): Promise<string> {
        let parsed: { input: string };
        try {
          parsed = JSON.parse(input);
        } catch (e) {
          throw new ToolSerializationError(`Failed to parse tool input for '${options.toolName}': ${e}`);
        }
        const inherited =
          parent !== undefined &&
          (parent.model !== undefined ||
            parent.modelSettings !== undefined ||
            parent.defaultModelParams !== undefined) &&
          patched(AGENT_TOOL_RUN_CONFIG_PATCH)
            ? parent
            : undefined;
        const runConfig: NonNullable<TemporalRunOptions['runConfig']> = {};
        if (parent?.sandbox !== undefined) runConfig.sandbox = parent.sandbox;
        if (inherited?.modelSettings !== undefined) runConfig.modelSettings = inherited.modelSettings;
        const runner = agentToolRunner(inherited?.defaultModelParams, inherited?.model);
        const result = await runner.run(
          agent,
          parsed.input,
          Object.keys(runConfig).length > 0 ? { runConfig } : undefined
        );
        return extractToolOutput(result, agent.name, options);
      },
      needsApproval: async () => false,
      isEnabled: async () => true,
      [agentAsToolParentRun]: createTool,
    }) as TemporalAgentTool;

  return createTool();
}

export async function extractToolOutput<TAgent extends Agent<any, any>>(
  result: RunResult<any, TAgent>,
  agentName: string,
  options: AgentAsToolOptions<TAgent>
): Promise<string> {
  if (result.interruptions.length > 0) {
    throw ApplicationFailure.create({
      type: 'NestedAgentInterruption',
      nonRetryable: true,
      message:
        `Nested agent '${agentName}' invoked via agentAsTool('${options.toolName}') paused for ` +
        `${result.interruptions.length} approval interruption(s). agentAsTool does not currently ` +
        `bridge nested approval interruptions to the parent runner; handle approvals at the top-level runner instead.`,
    });
  }
  if (options.customOutputExtractor) {
    return options.customOutputExtractor(result);
  }
  if (result.finalOutput === undefined) return '';
  return typeof result.finalOutput === 'string' ? result.finalOutput : JSON.stringify(result.finalOutput);
}
