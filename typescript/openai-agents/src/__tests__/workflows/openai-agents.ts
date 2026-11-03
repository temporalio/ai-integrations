import {
  Agent,
  getGlobalTraceProvider,
  type MCPToolFilterContext,
  withTrace,
  withTraceContext,
} from '@openai/agents-core';
import {
  ApplicationFailure,
  CancellationScope,
  condition,
  defineSignal,
  isCancellation,
  proxyActivities,
  setHandler,
  sleep,
} from '@temporalio/workflow';
import { WorkflowStream } from '@temporalio/workflow-streams/workflow';
import { TemporalOpenAIRunner, statelessMcpServer, statefulMcpServer } from '../../workflow';

export {
  statefulMcpCallerCancelWorkflow,
  statefulMcpCancelWorkflow,
  statefulMcpGatherWorkflow,
} from './stateful-mcp-replay';

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

export async function mcpToolFilterWorkflow(stateful: boolean): Promise<string[]> {
  const name = stateful ? 'filterStateful' : 'filterStateless';
  // The factory's server blocks `danger_tool` with a static filter; this filter blocks `extra_tool`
  // unless the run's context allows it.
  const toolFilter = async ({ runContext, serverName }: MCPToolFilterContext, tool: { name: string }) =>
    serverName === name && (tool.name !== 'extra_tool' || (runContext.context as { admin: boolean }).admin);
  const server = stateful ? statefulMcpServer(name, { toolFilter }) : statelessMcpServer(name, { toolFilter });
  await server.connect();
  try {
    const outputs: string[] = [];
    // Two runs of one agent with different contexts, then a second agent on the same server.
    for (const [agentName, admin] of [
      ['FirstAgent', true],
      ['FirstAgent', false],
      ['SecondAgent', false],
    ] as const) {
      const agent = new Agent({
        name: agentName,
        instructions: 'You have access to MCP tools.',
        model: 'gpt-4o-mini',
        mcpServers: [server],
      });
      const result = await new TemporalOpenAIRunner().run(agent, 'Hello', { context: { admin } });
      outputs.push(result.finalOutput ?? '');
    }
    return outputs;
  } finally {
    await server.close();
  }
}

class FilterSession {
  isAdmin(): boolean {
    return true;
  }
}

export async function mcpToolFilterResumeWorkflow(): Promise<string> {
  // Resuming from a RunState rebuilds the context from JSON, so `session` loses `isAdmin()`.
  const toolFilter = async ({ runContext }: MCPToolFilterContext) =>
    (runContext.context as { session: FilterSession }).session.isAdmin();
  const agent = new Agent({
    name: 'FirstAgent',
    instructions: 'You have access to MCP tools.',
    model: 'gpt-4o-mini',
    mcpServers: [statelessMcpServer('filterStateless', { toolFilter })],
  });
  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, 'Hello', { context: { session: new FilterSession() } });
  await runner.run(agent, result.state);
  return 'resumed';
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

const text = (content: unknown): string =>
  (content as { text?: string }[]).map((c) => c.text ?? '').join('');

export async function statefulMcpRetriedWorkflow(): Promise<string[]> {
  const server = statefulMcpServer('retried', {
    serverSessionConfig: { heartbeatTimeout: '2 seconds' },
    config: { scheduleToStartTimeout: '5 seconds' },
  });
  await server.connect();
  const out: string[] = [];
  try {
    out.push(text(await server.callTool('put', { value: 'V' })));
    // The session's host is lost here, and Temporal retries the session Activity.
    await sleep('10 seconds');
    try {
      out.push(text(await server.callTool('get', {})));
    } catch (err: unknown) {
      // Leaving the block reports the session Activity's failure.
      out.push(String(err));
    }
  } finally {
    await server.cleanup();
  }
  return out;
}

export const resumeSignal = defineSignal('resume');

export async function statefulMcpSessionEndedWorkflow(cancelQueued: boolean): Promise<string[]> {
  // No schedule-to-start timeout: only the session's end stops an operation
  // waiting on the session's Task Queue.
  const server = statefulMcpServer('sessionEnded', { config: { scheduleToStartTimeout: 0 } });
  let resumed = false;
  setHandler(resumeSignal, () => {
    resumed = true;
  });
  await server.connect();
  const out: string[] = [];
  try {
    if (cancelQueued) {
      // Cancelled while queued, before the session's Worker polls.
      const scope = new CancellationScope();
      const call = scope.run(() => server.callTool('put', { value: 'cancelled' }));
      await sleep('1 second');
      scope.cancel();
      try {
        await call;
      } catch (err: unknown) {
        out.push(isCancellation(err) ? 'cancelled' : String(err));
      }
      out.push(text(await server.callTool('get', {})));
      return out;
    }
    out.push(text(await server.callTool('put', { value: 'V' })));
    await condition(() => resumed);
    try {
      out.push(text(await server.callTool('get', {})));
    } catch (err: unknown) {
      out.push(err instanceof ApplicationFailure ? `${err.type}` : String(err));
    }
  } finally {
    await server.cleanup();
  }
  return out;
}

export async function statefulMcpOperationWaitingWorkflow(): Promise<string[]> {
  // No schedule-to-start timeout: only the session's end stops "get" waiting.
  const server = statefulMcpServer('sessionEnded', { config: { scheduleToStartTimeout: 0 } });
  let resumed = false;
  setHandler(resumeSignal, () => {
    resumed = true;
  });
  await server.connect();
  try {
    // "hold" takes the session Worker's only slot, so "get" waits behind it.
    const hold = server.callTool('hold', {}).catch(() => undefined);
    await condition(() => resumed);
    const get = await server.callTool('get', {}).then(
      (content) => text(content),
      (err: unknown) => (err instanceof ApplicationFailure ? `${err.type}` : String(err))
    );
    await hold;
    return [get];
  } finally {
    await server.cleanup();
  }
}

export async function statefulMcpCancelledCallerWorkflow(): Promise<string> {
  const server = statefulMcpServer('sessionEnded', { config: { scheduleToStartTimeout: 0 } });
  await server.connect();
  try {
    try {
      await server.callTool('hold', {});
    } catch (err: unknown) {
      if (!isCancellation(err)) throw err;
    }
    // The Workflow's cancellation also ended the session; this call must still
    // reject with the cancellation, not with DedicatedWorkerFailure.
    await server.listTools();
    return 'listed';
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

export type TracingKeyEntryPoint =
  | 'run'
  | 'stream'
  | 'runConfig'
  | 'withTrace'
  | 'withTraceOnly'
  | 'traceContext'
  | 'activity';

export async function tracingKeyRefusedWorkflow(entryPoint: TracingKeyEntryPoint): Promise<string> {
  const agent = new Agent({
    name: 'TracingKeyAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });
  const runner = new TemporalOpenAIRunner();
  // A keyed trace that is never started, so only the checks in run and at each header write see its key.
  const keyedTrace = () =>
    getGlobalTraceProvider().createTrace({ name: 'workflow-trace', tracingApiKey: 'sk-workflow' });
  const { notRegistered } = proxyActivities<{ notRegistered(): Promise<void> }>({
    startToCloseTimeout: '5 seconds',
    retry: { maximumAttempts: 1 },
  });
  try {
    switch (entryPoint) {
      case 'run':
        await runner.run(agent, 'Hello', { tracing: { apiKey: 'sk-run' } });
        break;
      case 'stream':
        await runner.run(agent, 'Hello', { stream: true, tracing: { apiKey: 'sk-run' } });
        break;
      case 'runConfig': {
        // runConfig has no tracing field, but a config object shared through a variable passes the type check.
        const runConfig = { workflowName: 'support', tracing: { apiKey: 'sk-run' } };
        await runner.run(agent, 'Hello', { runConfig });
        break;
      }
      case 'withTrace':
        await withTrace('workflow-trace', () => runner.run(agent, 'Hello'), { tracingApiKey: 'sk-workflow' });
        break;
      case 'withTraceOnly':
        // No run and no Temporal call inside, so only the trace processor's check sees the key.
        await withTrace('workflow-trace', async () => {}, { tracingApiKey: 'sk-workflow' });
        break;
      case 'traceContext':
        await withTraceContext({ trace: keyedTrace() }, () => runner.run(agent, 'Hello'));
        break;
      case 'activity':
        await withTraceContext({ trace: keyedTrace() }, () => notRegistered());
        break;
    }
  } catch (e) {
    return `refused: ${(e as Error).message}`;
  }
  return 'not refused';
}
