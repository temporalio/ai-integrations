import { Agent, RunState, tool } from '@openai/agents-core';
import { ApplicationFailure, condition } from '@temporalio/workflow';
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

interface ResumeContext {
  original: string;
  after?: string;
  writes: string[];
}

export interface ApprovalResumeContextResult {
  callerAfter: string | undefined;
  resumedUsesCallerObject: boolean;
  writes: string[];
  approvalRounds: number;
  failedResumeThrew: boolean;
  approvalOwners: number;
  approvedCallIds: number;
  sharedContextWrites: string[];
  sharedContextInterruptions: number;
  retryThrew: boolean;
  retryAttempts: number;
  retryWrites: string[];
  retryInterruptions: number;
  concurrentFailedResumeThrew: boolean;
  concurrentWrites: string[];
  concurrentInterruptions: number;
}

/** Serialized approval record, as the `RunContext` internal below returns it. */
interface ApprovalRecord {
  approved: boolean | string[];
}

/** Counts the agents that own function approvals in `context`, and the call IDs they approve. */
function countApprovalOwners(context: unknown): { approvalOwners: number; approvedCallIds: number } {
  let approvalOwners = 0;
  // _toJSONForRunState is internal to @openai/agents-core; it looks up each owner in the map it's given.
  const json = (
    context as {
      _toJSONForRunState(keys: { get(agent: unknown): string }): {
        functionApprovals?: Array<{ approvals: Record<string, ApprovalRecord> }>;
      };
    }
  )._toJSONForRunState({ get: () => `owner-${++approvalOwners}` });
  let approvedCallIds = 0;
  for (const { approvals } of json.functionApprovals ?? []) {
    for (const record of Object.values(approvals)) {
      approvedCallIds += Array.isArray(record.approved) ? record.approved.length : 0;
    }
  }
  return { approvalOwners, approvedCallIds };
}

/**
 * Approves a tool that writes the run context and resumes the same `RunState` until the run
 * finishes, then reads the caller's context object. The first resume uses an agent the state
 * doesn't match, which must fail without losing the approval. Then three paused states share one
 * `RunContext`, all are approved, and resuming one must keep the others' approvals. Last, an approved
 * tool throws once, and resuming the same state again must run it without asking again. Then a
 * resume fails while another state on its context advances and is approved again.
 */
export async function approvalResumeContextWorkflow(): Promise<ApprovalResumeContextResult> {
  const setAfter = tool({
    name: 'set_after',
    description: 'Set a value after approval.',
    parameters: {
      type: 'object' as const,
      properties: { value: { type: 'string' } },
      required: ['value'] as const,
      additionalProperties: false as const,
    },
    needsApproval: true,
    execute: async (args, runContext) => {
      const context = runContext!.context as ResumeContext;
      context.after = (args as { value: string }).value;
      context.writes.push(context.after);
      return 'ok';
    },
  });
  const agent = new Agent<ResumeContext>({
    name: 'ResumeContextAgent',
    instructions: 'Set the value.',
    model: 'gpt-4o-mini',
    tools: [setAfter],
  });
  const otherAgent = new Agent<ResumeContext>({ name: 'OtherAgent', model: 'gpt-4o-mini', tools: [setAfter] });
  const runner = new TemporalOpenAIRunner();
  const ctx: ResumeContext = { original: 'orig', writes: [] };
  let result = await runner.run(agent, 'go', { context: ctx });
  let approvalRounds = 0;
  let failedResumeThrew = false;
  while (result.interruptions.length > 0 && approvalRounds < 10) {
    result.state.approve(result.interruptions[0]!);
    if (approvalRounds++ === 0) {
      try {
        await runner.run(otherAgent, result.state);
      } catch {
        failedResumeThrew = true;
      }
    }
    result = await runner.run(agent, result.state);
  }
  const owners = countApprovalOwners(result.runContext);

  const sharedAgentA = new Agent<ResumeContext>({ name: 'SharedAgentA', model: 'gpt-4o-mini', tools: [setAfter] });
  const sharedAgentB = new Agent<ResumeContext>({ name: 'SharedAgentB', model: 'gpt-4o-mini', tools: [setAfter] });
  const sharedCtx: ResumeContext = { original: 'shared', writes: [] };
  const pausedA = await runner.run(sharedAgentA, 'go', { context: sharedCtx });
  const pausedB = await runner.run(sharedAgentB, 'go', { context: sharedCtx });
  const pausedC = await runner.run(sharedAgentA, 'go', { context: sharedCtx });
  // State A keeps the converted graph its run produced. B is rehydrated onto A's context, and C onto
  // A's context and A's converted graph, so A and C share their approval owners.
  const stateA = pausedA.state;
  const stateB = await RunState.fromStringWithContext(sharedAgentB, pausedB.state.toString(), stateA._context);
  const stateC = await RunState.fromStringWithContext(pausedA.lastAgent!, pausedC.state.toString(), stateA._context);
  for (const state of [stateA, stateB, stateC]) state.approve(state.getInterruptions()[0]!);
  const resumed = [
    await runner.run(sharedAgentA, stateA),
    await runner.run(sharedAgentB, stateB),
    await runner.run(sharedAgentA, stateC),
  ];

  const flaky = tool({
    name: 'flaky',
    description: 'Fail once after approval.',
    parameters: {
      type: 'object' as const,
      properties: { value: { type: 'string' } },
      required: ['value'] as const,
      additionalProperties: false as const,
    },
    needsApproval: true,
    errorFunction: null,
    execute: async (args, runContext) => {
      const context = runContext!.context as ResumeContext & { attempts: number };
      if (context.attempts++ === 0) throw new Error('flaky failed');
      context.writes.push((args as { value: string }).value);
      return 'ok';
    },
  });
  const retryAgent = new Agent<ResumeContext>({ name: 'RetryAgent', model: 'gpt-4o-mini', tools: [flaky] });
  const retryCtx = { original: 'retry', writes: [] as string[], attempts: 0 };
  const pausedRetry = await runner.run(retryAgent, 'go', { context: retryCtx });
  pausedRetry.state.approve(pausedRetry.interruptions[0]!);
  let retryThrew = false;
  try {
    await runner.run(retryAgent, pausedRetry.state);
  } catch {
    retryThrew = true;
  }
  const retried = await runner.run(retryAgent, pausedRetry.state);

  // State D's resume waits in a tool's isEnabled check, then throws. Meanwhile state E, which shares
  // D's context, resumes, interrupts on its next call, and has that call approved.
  const gate = { holdD: false, eApproved: false };
  const gated = tool({
    name: 'gated',
    description: 'Record a value after approval.',
    parameters: {
      type: 'object' as const,
      properties: { value: { type: 'string' } },
      required: ['value'] as const,
      additionalProperties: false as const,
    },
    needsApproval: true,
    isEnabled: async ({ agent }) => {
      if (gate.holdD && agent.name === 'ConcurrentAgentD') {
        await condition(() => gate.eApproved);
        throw new Error('isEnabled failed');
      }
      return true;
    },
    execute: async (args, runContext) => {
      (runContext!.context as ResumeContext).writes.push((args as { value: string }).value);
      return 'ok';
    },
  });
  const agentD = new Agent<ResumeContext>({ name: 'ConcurrentAgentD', model: 'gpt-4o-mini', tools: [gated] });
  const agentE = new Agent<ResumeContext>({ name: 'ConcurrentAgentE', model: 'gpt-4o-mini', tools: [gated] });
  const concurrentCtx: ResumeContext = { original: 'concurrent', writes: [] };
  const pausedD = await runner.run(agentD, 'go', { context: concurrentCtx });
  const pausedE = await runner.run(agentE, 'go', { context: concurrentCtx });
  const stateD = pausedD.state;
  const stateE = await RunState.fromStringWithContext(agentE, pausedE.state.toString(), stateD._context);
  stateE.approve(stateE.getInterruptions()[0]!);
  gate.holdD = true;
  const failedD = runner.run(agentD, stateD).then(
    () => false,
    () => true
  );
  const pausedAgainE = await runner.run(agentE, stateE);
  pausedAgainE.state.approve(pausedAgainE.interruptions[0]!);
  gate.eApproved = true;
  const concurrentFailedResumeThrew = await failedD;
  gate.holdD = false;
  const finishedE = await runner.run(agentE, pausedAgainE.state);

  return {
    callerAfter: ctx.after,
    resumedUsesCallerObject: result.runContext.context === ctx,
    writes: ctx.writes,
    approvalRounds,
    failedResumeThrew,
    ...owners,
    sharedContextWrites: sharedCtx.writes,
    sharedContextInterruptions: resumed.reduce((count, { interruptions }) => count + interruptions.length, 0),
    retryThrew,
    retryAttempts: retryCtx.attempts,
    retryWrites: retryCtx.writes,
    retryInterruptions: retried.interruptions.length,
    concurrentFailedResumeThrew,
    concurrentWrites: concurrentCtx.writes,
    concurrentInterruptions: finishedE.interruptions.length,
  };
}
