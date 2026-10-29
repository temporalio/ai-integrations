import { Agent, RunState, tool, type Model, type Tool } from '@openai/agents-core';
import { SandboxAgent } from '@openai/agents-core/sandbox';
import { ApplicationFailure, continueAsNew } from '@temporalio/workflow';
import { WorkflowStream } from '@temporalio/workflow-streams/workflow';
import { TemporalOpenAIRunner, statelessMcpServer, statefulMcpServer, temporalSandboxClient } from '../../workflow';

export interface StreamingAgentResult {
  finalOutput: string;
  /** `output_text_delta` deltas observed in-workflow, in order. */
  deltas: string[];
}

export async function streamingAgentWorkflow(prompt: string): Promise<StreamingAgentResult> {
  new WorkflowStream();
  const agent = new Agent({
    name: 'StreamingAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });
  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt, { stream: true });

  const deltas: string[] = [];
  for await (const event of result) {
    if (event.type === 'raw_model_stream_event' && event.data.type === 'output_text_delta') {
      deltas.push(event.data.delta);
    }
  }
  await result.completed;
  return { finalOutput: result.finalOutput ?? '', deltas };
}

export async function streamingNoTopicWorkflow(prompt: string): Promise<string> {
  new WorkflowStream();
  const agent = new Agent({
    name: 'StreamingAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });
  const runner = new TemporalOpenAIRunner();
  try {
    const result = await runner.run(agent, prompt, { stream: true });
    for await (const _event of result);
    await result.completed;
    return 'unexpected-success';
  } catch (err: unknown) {
    if (err instanceof ApplicationFailure) {
      return `${err.type}`;
    }
    throw err;
  }
}

export async function streamingLocalActivityWorkflow(prompt: string): Promise<string> {
  new WorkflowStream();
  const agent = new Agent({
    name: 'StreamingAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });
  const runner = new TemporalOpenAIRunner({
    defaultModelParams: { streamingTopic: 'events', useLocalActivity: true },
  });
  try {
    const result = await runner.run(agent, prompt, { stream: true });
    for await (const _event of result);
    await result.completed;
    return 'unexpected-success';
  } catch (err: unknown) {
    if (err instanceof ApplicationFailure) {
      return `${err.type}`;
    }
    throw err;
  }
}

export async function localActivityAgentWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'LocalActivityAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });
  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

export async function mcpFactoryArgWorkflow(prompt: string): Promise<string> {
  const mcpServer = statelessMcpServer('testMcp', {
    factoryArgument: { tenantId: 'tenant-42' },
  });
  const agent = new Agent({
    name: 'McpFactoryArgAgent',
    instructions: 'You have access to MCP tools.',
    model: 'gpt-4o-mini',
    mcpServers: [mcpServer],
  });
  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt, { maxTurns: 5 });
  return result.finalOutput ?? '';
}

export async function summaryOverrideStringWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'SummaryAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });
  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

export async function agentToolRunConfigWorkflow(prompt: string): Promise<string> {
  const expert = new Agent({ name: 'Expert', instructions: 'Answer as the expert.' });
  const specialist = new Agent({ name: 'Specialist', instructions: 'Answer as the specialist.', model: 'own-model' });
  const lead = new Agent({
    name: 'Lead',
    instructions: 'Use the tools.',
    tools: [expert.asTool({ toolName: 'ask_expert' }), specialist.asTool({ toolName: 'ask_specialist' })],
  });
  const runner = new TemporalOpenAIRunner({
    defaultModelParams: { startToCloseTimeout: '17 seconds', summary: 'runner default' },
  });
  const result = await runner.run(lead, prompt, {
    runConfig: { model: 'run-model', modelSettings: { temperature: 0.33, toolChoice: 'auto' } },
  });
  return result.finalOutput ?? '';
}

export type AgentToolRejection =
  | 'model-object'
  | 'nested-handoff-model-object'
  | 'model-provider'
  | 'raw-sandbox'
  | 'raw-sandbox-run-options'
  | 'unrecorded'
  | 'name-collision';

/** Returns the failure for an `asTool()` setup that would run a model or sandbox in the Workflow. */
export async function agentToolRejectedWorkflow(kind: AgentToolRejection): Promise<string> {
  const inlineModel: Model = {
    async getResponse() {
      throw new Error('the inline model ran in the Workflow');
    },
    // eslint-disable-next-line require-yield
    async *getStreamedResponse() {
      throw new Error('the inline model ran in the Workflow');
    },
  };
  const rawSandbox = { client: { backendId: 'raw' } as any };
  const nested = new Agent({ name: 'Nested', instructions: 'Answer.', model: 'nested-model' });
  const tools: Record<AgentToolRejection, () => Tool<any>> = {
    'model-object': () => new Agent({ name: 'Inline', model: inlineModel }).asTool({ toolName: 'ask_nested' }),
    'nested-handoff-model-object': () =>
      new Agent({
        name: 'Router',
        model: 'router-model',
        handoffs: [new Agent({ name: 'Inline', model: inlineModel })],
      }).asTool({ toolName: 'ask_nested' }),
    'model-provider': () =>
      nested.asTool({ toolName: 'ask_nested', runConfig: { modelProvider: { getModel: () => inlineModel } } }),
    'raw-sandbox': () =>
      new SandboxAgent({ name: 'Sandboxed', model: 'nested-model' }).asTool({
        toolName: 'ask_nested',
        runConfig: { sandbox: rawSandbox },
      }),
    'raw-sandbox-run-options': () =>
      new SandboxAgent({ name: 'Sandboxed', model: 'nested-model' }).asTool({
        toolName: 'ask_nested',
        runOptions: { sandbox: rawSandbox },
      }),
    unrecorded: () => ({ ...nested.asTool({ toolName: 'ask_nested' }) }),
    'name-collision': () => new Agent({ name: 'Nested', model: 'other-model' }).asTool({ toolName: 'ask_nested' }),
  };
  const lead = new Agent({
    name: 'Lead',
    instructions: 'Use the tool.',
    model: 'lead-model',
    tools: [tools[kind]()],
    handoffs: kind === 'name-collision' ? [nested] : [],
  });
  try {
    await new TemporalOpenAIRunner().run(lead, 'go', {
      runConfig: { sandbox: { client: temporalSandboxClient('unused') } },
    });
    return 'unexpected-success';
  } catch (err: unknown) {
    if (err instanceof ApplicationFailure) return `${err.type}: ${err.message}`;
    throw err;
  }
}

/**
 * `Nested` is both a handoff target and an `asTool()` source of `Parent`, and pauses on an approval.
 * `legacyTool` stands in for the plugin's former `agentAsTool`, a function tool not linked to an agent.
 */
function sharedAgentDeclarations(legacyTool: boolean): { parent: Agent<any, any>; executions: () => number } {
  let executions = 0;
  const confirmAction = tool({
    name: 'confirmAction',
    description: 'Confirm the action.',
    parameters: {
      type: 'object' as const,
      properties: { reason: { type: 'string' } },
      required: ['reason'] as const,
      additionalProperties: false as const,
    },
    needsApproval: true,
    execute: async (args) => {
      executions++;
      return `confirmed:${(args as { reason: string }).reason}`;
    },
  });
  const nested = new Agent({
    name: 'Nested',
    instructions: 'Answer as nested.',
    model: 'own-model',
    tools: [confirmAction],
  });
  const askNested = legacyTool
    ? tool({
        name: 'ask_nested',
        description: '',
        parameters: {
          type: 'object' as const,
          properties: { input: { type: 'string' } },
          required: ['input'] as const,
          additionalProperties: false as const,
        },
        execute: async () => '',
      })
    : nested.asTool({ toolName: 'ask_nested' });
  const parent = new Agent({
    name: 'Parent',
    instructions: 'Reach nested.',
    model: 'parent-model',
    handoffs: [nested],
    tools: [askNested],
  });
  return { parent, executions: () => executions };
}

export interface SharedAgentApprovalInput {
  legacyFirstRun?: boolean;
  savedState?: string;
}

export interface SharedAgentApprovalResult {
  savedIdentity: string;
  output: string;
  executions: number;
}

/** Pauses on the approval, continues as new with the saved state, then approves and resumes. */
export async function sharedAgentApprovalWorkflow(input: SharedAgentApprovalInput): Promise<SharedAgentApprovalResult> {
  const runConfig = { model: 'run-model' };
  if (input.savedState === undefined) {
    const { parent } = sharedAgentDeclarations(input.legacyFirstRun === true);
    const result = await new TemporalOpenAIRunner().run(parent, 'go', { runConfig });
    if (result.interruptions.length !== 1) {
      throw ApplicationFailure.nonRetryable(`expected one approval, got ${result.interruptions.length}`);
    }
    return await continueAsNew<typeof sharedAgentApprovalWorkflow>({ savedState: result.state.toString() });
  }
  const { parent, executions } = sharedAgentDeclarations(false);
  const state = await RunState.fromString(parent, input.savedState);
  for (const interruption of state.getInterruptions()) state.approve(interruption);
  const result = await new TemporalOpenAIRunner().run(parent, state, { runConfig });
  const saved = JSON.parse(input.savedState);
  return {
    savedIdentity: saved.currentAgent.identity ?? saved.currentAgent.name,
    output: result.finalOutput ?? '',
    executions: executions(),
  };
}

export interface AgentToolStreamResult {
  output: string;
  onStream: string[];
  on: string[];
}

/** A nested `asTool()` run streams to its `onStream` option and `on()` handlers. */
export async function agentToolStreamWorkflow(prompt: string): Promise<AgentToolStreamResult> {
  new WorkflowStream();
  const onStream: string[] = [];
  const on: string[] = [];
  const streamer = new Agent({ name: 'Streamer', instructions: 'Stream the answer.', model: 'stream-model' });
  const askStreamer = streamer.asTool({
    toolName: 'ask_streamer',
    onStream: ({ event }) => {
      if (event.type === 'raw_model_stream_event' && event.data.type === 'output_text_delta') {
        onStream.push(event.data.delta);
      }
    },
  });
  askStreamer.on('raw_model_stream_event', ({ event }) => {
    if (event.type === 'raw_model_stream_event' && event.data.type === 'output_text_delta') {
      on.push(event.data.delta);
    }
  });
  const lead = new Agent({ name: 'Lead', instructions: 'Use the tool.', model: 'lead-model', tools: [askStreamer] });
  const runner = new TemporalOpenAIRunner({ defaultModelParams: { streamingTopic: 'events' } });
  const result = await runner.run(lead, prompt);
  return { output: result.finalOutput ?? '', onStream, on };
}

export async function statefulMcpNotConnectedWorkflow(): Promise<string> {
  const server = statefulMcpServer('testStateful');
  try {
    await server.listTools();
    return 'unexpected-success';
  } catch (err: unknown) {
    if (err instanceof ApplicationFailure) {
      return err.message;
    }
    throw err;
  }
}

export async function statefulMcpIsolationWorkflow(): Promise<string> {
  const server = statefulMcpServer('isolationTest');
  await server.connect();
  try {
    const tools = await server.listTools();
    if (tools.length === 0) return 'no-tools';
    const result = await server.callTool(tools[0]!.name, null);
    return JSON.stringify(result);
  } finally {
    await server.cleanup();
  }
}

export async function statefulMcpHeartbeatTimeoutWorkflow(): Promise<string> {
  const server = statefulMcpServer('heartbeatTest', {
    config: {
      startToCloseTimeout: '30 seconds',
      heartbeatTimeout: '1 second',
      retryPolicy: { maximumAttempts: 1 },
    },
  });
  await server.connect();
  try {
    await server.listTools();
    return 'unexpected-success';
  } catch (err: unknown) {
    if (err instanceof ApplicationFailure) {
      return `${err.type}: ${err.message}`;
    }
    throw err;
  } finally {
    await server.cleanup();
  }
}

export async function statefulMcpSlowConnectHeartbeatWorkflow(): Promise<string> {
  const server = statefulMcpServer('slowConnectTest', {
    serverSessionConfig: {
      heartbeatTimeout: '3 seconds',
      startToCloseTimeout: '30 seconds',
    },
    config: {
      startToCloseTimeout: '10 seconds',
    },
  });
  await server.connect();
  try {
    const tools = await server.listTools();
    return `connected:${tools.length}`;
  } finally {
    await server.cleanup();
  }
}
