import { Agent, type MCPToolMetaContext } from '@openai/agents-core';
import { ActivityFailure, ApplicationFailure, proxyActivities } from '@temporalio/workflow';
import { WorkflowStream } from '@temporalio/workflow-streams/workflow';
import { TemporalOpenAIRunner, statelessMcpServer, statefulMcpServer } from '../../workflow';

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

export async function mcpToolMetaWorkflow(stateful: boolean): Promise<string> {
  const toolMetaResolver = ({ runContext, toolName, serverName }: MCPToolMetaContext) => ({
    tenant: (runContext.context as { tenant: string }).tenant,
    tool: toolName,
    server: serverName,
  });
  const server = stateful
    ? statefulMcpServer('metaStateful', { toolMetaResolver })
    : statelessMcpServer('metaStateless', { toolMetaResolver });
  await server.connect();
  try {
    const agent = new Agent({
      name: 'McpToolMetaAgent',
      instructions: 'You have access to MCP tools.',
      model: 'gpt-4o-mini',
      mcpServers: [server],
    });
    const result = await new TemporalOpenAIRunner().run(agent, 'Hello', { context: { tenant: 'tenant-7' } });
    return result.finalOutput ?? '';
  } finally {
    await server.close();
  }
}

const { orderingGate, orderingKick, orderingAfterMcp, orderingAfterGate } = proxyActivities<
  Record<'orderingGate' | 'orderingKick' | 'orderingAfterMcp' | 'orderingAfterGate', () => Promise<void>>
>({ startToCloseTimeout: '10 seconds' });

/**
 * A stateful callTool() and a gate Activity complete while a Workflow Task spins, so both
 * results reach the same activation. The order of the follow-up Activities depends on how many
 * promise steps callTool() takes after its Activity completes.
 */
export async function statefulMcpOrderingWorkflow(spin: number): Promise<void> {
  const server = statefulMcpServer('ordering');
  await server.connect();
  await server.listTools();
  const mcp = async () => {
    await server.callTool('slow', {});
    await orderingAfterMcp();
  };
  const gate = async () => {
    await orderingGate();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    await orderingAfterGate();
  };
  const kick = async () => {
    await orderingKick();
    let x = 0;
    for (let i = 0; i < spin; i++) x += i;
    return x;
  };
  await Promise.all([mcp(), gate(), kick()]);
  await server.cleanup();
}

function failureType(err: unknown): string {
  const failure = err instanceof ActivityFailure ? err.cause : err;
  return failure instanceof ApplicationFailure ? `${failure.type}:${failure.nonRetryable}` : String(err);
}

export async function statefulMcpFactoryResolverWorkflow(): Promise<string[]> {
  // listTools() is scheduled before the session fails, so it gets the refusal at its schedule-to-start timeout;
  // callTool() starts after the session failed, so it throws the refusal without scheduling an Activity.
  const server = statefulMcpServer('factoryResolver', { config: { scheduleToStartTimeout: '3 seconds' } });
  await server.connect();
  const results: string[] = [];
  for (const operation of [() => server.listTools(), () => server.callTool('echo', {})]) {
    try {
      await operation();
      results.push('unexpected-success');
    } catch (err: unknown) {
      results.push(failureType(err));
    }
  }
  try {
    await server.cleanup();
    results.push('cleanup-success');
  } catch (err: unknown) {
    results.push(failureType(err));
  }
  return results;
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

export async function tracingApiKeyWorkflow(apiKey?: string): Promise<string> {
  const agent = new Agent({
    name: 'TracingKeyAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });
  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, 'Hello', apiKey ? { tracing: { apiKey } } : undefined);
  return result.finalOutput ?? '';
}
