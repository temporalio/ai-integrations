import { Agent, type AgentOptions, RunState, handoff, tool } from '@openai/agents-core';
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
  rehydratedWrites: string[];
  rehydratedApprovalOwners: number;
  rehydratedApprovedCallIds: number;
  handoffTarget: string | undefined;
  handoffFilters: string[];
  disabledHandoffThrew: boolean;
  sharedFieldsWrites: string[];
  sharedFieldsTemperatures: Array<number | undefined>;
  failedConversionThrew: boolean;
  failedConversionWrites: string[];
  failedConversionInterruptions: number;
  versions: string[];
  lastAgentVersion: string | undefined;
  customChildWrites: string[];
  customChildInterruptions: number;
  customChildSavedApprovalOwners: number;
  replacedChildWrites: string[];
  replacedChildInterruptions: number;
  failedRestoreThrew: boolean;
  failedRestoreWrites: string[];
  failedRestoreInterruptions: number;
  registeredVersions: Array<string | undefined>;
  lastAgentRegisteredVersion: string | undefined;
  rejectedResumesThrew: boolean[];
  rejectedExecutions: string[];
  rejectedInterruptions: number;
  failedRestores: number;
  failedRestoreApprovalOwners: number;
  failedRestoreApprovedCallIds: number;
  recoveredWrites: string[];
  recoveredApprovalOwners: number;
  recoveredApprovedCallIds: number;
}

/** An agent whose `clone` carries a private field, which can't be copied onto another object. */
class VersionedAgent extends Agent<ResumeContext> {
  #version: string;

  constructor(config: AgentOptions<ResumeContext>, version: string) {
    super(config);
    this.#version = version;
  }

  get version(): string {
    return this.#version;
  }

  setVersion(version: string): void {
    this.#version = version;
  }

  override clone(config: Partial<AgentOptions<ResumeContext>>): VersionedAgent {
    return new VersionedAgent({ ...this, ...config } as AgentOptions<ResumeContext>, this.#version);
  }
}

/** Versions that `RegisteredVersionAgent.clone` registers for each clone. */
const registeredVersions = new WeakMap<Agent<any, any>, string>();

/** An agent whose `clone` returns a plain `Agent` and registers its version outside the clone. */
class RegisteredVersionAgent extends Agent<ResumeContext> {
  override clone(config: Partial<AgentOptions<ResumeContext>>): Agent<ResumeContext> {
    const clone = super.clone(config);
    registeredVersions.set(clone, registeredVersions.get(this)!);
    return clone;
  }
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
 * resume fails while another state on its context advances and is approved again. Then each resume
 * rehydrates the saved state onto the original agent and context. Then a handoff changes between an
 * interruption and its resume. Then an agent's tools and model settings are replaced with equal copies
 * before a resume, and the approved tool changes them. Then a resume fails to convert a handoff target,
 * and the saved state must still carry its approval. Then an agent class whose clone keeps a private
 * field changes that field before the resume. Then a resume replaces a handoff target's object, because
 * its class overrides `clone` or the target itself was replaced. Last, a `clone` that returns a plain
 * `Agent` keeps state outside it, which changes before the resume. Then an approved call fails after
 * the resume replaced its agent's object, and is rejected through the original state, whose retry must
 * not run it. Last, the restore of a saved state fails repeatedly after merging its approvals, which must
 * not add a copy of them each time.
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

  // Each round saves the state as a string and rehydrates it onto the original agent and the caller's
  // context. The agent has a handoff, which stays unchanged here.
  const handoffFilters: string[] = [];
  const firstTarget = new Agent<ResumeContext>({ name: 'FirstTarget', model: 'gpt-4o-mini' });
  const secondTarget = new Agent<ResumeContext>({ name: 'SecondTarget', model: 'gpt-4o-mini' });
  const routing = handoff(firstTarget, {
    inputFilter: (data) => {
      handoffFilters.push('first');
      return data;
    },
  });
  const routingAgent = new Agent<ResumeContext>({
    name: 'RoutingAgent',
    model: 'gpt-4o-mini',
    tools: [setAfter],
    handoffs: [routing],
  });
  const rehydratedCtx: ResumeContext = { original: 'rehydrated', writes: [] };
  let rehydrated = await runner.run(routingAgent, 'go', { context: rehydratedCtx });
  const rehydratedRunContext = rehydrated.runContext;
  for (let round = 0; rehydrated.interruptions.length > 0 && round < 10; round++) {
    const state = await RunState.fromStringWithContext(routingAgent, rehydrated.state.toString(), rehydratedRunContext);
    state.approve(state.getInterruptions()[0]!);
    rehydrated = await runner.run(routingAgent, state);
  }
  const rehydratedOwners = countApprovalOwners(rehydratedRunContext);

  // The handoff's target and input filter change between the interruption and the resume, and then
  // the handoff is disabled before another resume. Each resume must use the handoff as it is now.
  const pausedRoute = await runner.run(routingAgent, 'go', { context: { original: 'route', writes: [] } });
  pausedRoute.state.approve(pausedRoute.interruptions[0]!);
  routing.agent = secondTarget;
  routing.inputFilter = (data) => {
    handoffFilters.push('second');
    return data;
  };
  const routed = await runner.run(routingAgent, pausedRoute.state);
  const pausedDisabled = await runner.run(routingAgent, 'go', { context: { original: 'disabled', writes: [] } });
  pausedDisabled.state.approve(pausedDisabled.interruptions[0]!);
  routing.isEnabled = async () => false;
  let disabledHandoffThrew = false;
  try {
    await runner.run(routingAgent, pausedDisabled.state);
  } catch {
    disabledHandoffThrew = true;
  }

  // Conversion shares an agent's tools array and model settings with the converted agent. They're
  // replaced with equal copies before the resume, and the approved tool then adds a tool and changes a
  // setting, which the next model call must see.
  const extra = tool({
    name: 'extra',
    description: 'Record a value.',
    parameters: {
      type: 'object' as const,
      properties: { value: { type: 'string' } },
      required: ['value'] as const,
      additionalProperties: false as const,
    },
    execute: async (args, runContext) => {
      (runContext!.context as ResumeContext).writes.push((args as { value: string }).value);
      return 'ok';
    },
  });
  const sharedFieldsTemperatures: Array<number | undefined> = [];
  const sharedFieldsAgent: Agent<ResumeContext> = new Agent<ResumeContext>({
    name: 'SharedFieldsAgent',
    model: 'gpt-4o-mini',
    instructions: (_runContext, current) => {
      sharedFieldsTemperatures.push(current.modelSettings.temperature);
      return 'Extend yourself.';
    },
    modelSettings: { temperature: 0.1 },
    tools: [
      tool({
        name: 'extend',
        description: 'Add a tool and change a setting after approval.',
        parameters: { type: 'object', properties: {}, required: [], additionalProperties: false },
        needsApproval: true,
        execute: async () => {
          sharedFieldsAgent.tools.push(extra);
          sharedFieldsAgent.modelSettings.temperature = 0.9;
          return 'ok';
        },
      }),
    ],
  });
  const sharedFieldsCtx: ResumeContext = { original: 'shared-fields', writes: [] };
  const pausedSharedFields = await runner.run(sharedFieldsAgent, 'go', { context: sharedFieldsCtx });
  pausedSharedFields.state.approve(pausedSharedFields.interruptions[0]!);
  sharedFieldsAgent.tools = [...sharedFieldsAgent.tools];
  sharedFieldsAgent.modelSettings = { ...sharedFieldsAgent.modelSettings };
  await runner.run(sharedFieldsAgent, pausedSharedFields.state);

  // The run pauses on the handoff target's call. A resume while the target has a raw function tool
  // fails to convert it. Once the tool is removed, the state saved as a string must still carry the
  // approval: a new context rehydrated from it must run the call without asking again.
  const failChild = new Agent<ResumeContext>({ name: 'FailChild', model: 'gpt-4o-mini', tools: [setAfter] });
  const failRoot = new Agent<ResumeContext>({ name: 'FailRoot', model: 'gpt-4o-mini', handoffs: [failChild] });
  const failedConversionCtx: ResumeContext = { original: 'failed-conversion', writes: [] };
  const pausedFail = await runner.run(failRoot, 'go', { context: failedConversionCtx });
  pausedFail.state.approve(pausedFail.interruptions[0]!);
  failChild.tools = [setAfter, (() => 'raw') as any];
  let failedConversionThrew = false;
  try {
    await runner.run(failRoot, pausedFail.state);
  } catch {
    failedConversionThrew = true;
  }
  failChild.tools = [setAfter];
  const failRehydrated = await RunState.fromString(failRoot, pausedFail.state.toString());
  const failedConversionResumed = await runner.run(failRoot, failRehydrated);

  // The version changes between the interruption and the resume, and the agent the resumed run uses must
  // have the new one, as a fresh clone does.
  const versions: string[] = [];
  const versionedAgent = new VersionedAgent(
    {
      name: 'VersionedAgent',
      model: 'gpt-4o-mini',
      instructions: (_runContext, current) => {
        versions.push((current as VersionedAgent).version);
        return 'Set the value.';
      },
      tools: [setAfter],
    },
    'before'
  );
  const pausedVersioned = await runner.run(versionedAgent, 'go', { context: { original: 'versioned', writes: [] } });
  pausedVersioned.state.approve(pausedVersioned.interruptions[0]!);
  versionedAgent.setVersion('after');
  const versioned = await runner.run(versionedAgent, pausedVersioned.state);

  // The run pauses on a handoff target's call. The resume reuses the root, but gives the target a new
  // object, because its class overrides clone. The approved call must run, and the state saved again
  // afterwards must still carry its approval.
  const customChild = new VersionedAgent({ name: 'CustomChild', model: 'gpt-4o-mini', tools: [setAfter] }, 'v');
  const customRoot = new Agent<ResumeContext>({ name: 'CustomRoot', model: 'gpt-4o-mini', handoffs: [customChild] });
  const customChildCtx: ResumeContext = { original: 'custom-child', writes: [] };
  const pausedCustomChild = await runner.run(customRoot, 'go', { context: customChildCtx });
  pausedCustomChild.state.approve(pausedCustomChild.interruptions[0]!);
  const customChildResumed = await runner.run(customRoot, pausedCustomChild.state);
  const customChildSaved = await RunState.fromString(customRoot, pausedCustomChild.state.toString());

  // The handoff target is replaced by another agent with the same name before the resume.
  const replacedChild = new Agent<ResumeContext>({ name: 'ReplacedChild', model: 'gpt-4o-mini', tools: [setAfter] });
  const replacedRoot = new Agent<ResumeContext>({
    name: 'ReplacedRoot',
    model: 'gpt-4o-mini',
    handoffs: [replacedChild],
  });
  const replacedChildCtx: ResumeContext = { original: 'replaced-child', writes: [] };
  const pausedReplacedChild = await runner.run(replacedRoot, 'go', { context: replacedChildCtx });
  pausedReplacedChild.state.approve(pausedReplacedChild.interruptions[0]!);
  replacedRoot.handoffs = [replacedChild.clone({})];
  const replacedChildResumed = await runner.run(replacedRoot, pausedReplacedChild.state);

  // The handoff to the paused target is removed, so the restore throws after the conversion. Once the
  // handoff is back, the state saved as a string must still carry the approval.
  const restoreChild = new Agent<ResumeContext>({ name: 'RestoreChild', model: 'gpt-4o-mini', tools: [setAfter] });
  const restoreRoot = new Agent<ResumeContext>({ name: 'RestoreRoot', model: 'gpt-4o-mini', handoffs: [restoreChild] });
  const pausedRestore = await runner.run(restoreRoot, 'go', { context: { original: 'restore', writes: [] } });
  pausedRestore.state.approve(pausedRestore.interruptions[0]!);
  restoreRoot.handoffs = [];
  let failedRestoreThrew = false;
  try {
    await runner.run(restoreRoot, pausedRestore.state);
  } catch {
    failedRestoreThrew = true;
  }
  restoreRoot.handoffs = [restoreChild];
  const restoreRehydrated = await RunState.fromString(restoreRoot, pausedRestore.state.toString());
  const failedRestoreResumed = await runner.run(restoreRoot, restoreRehydrated);

  // The registered version changes between the interruption and the resume, and the agent the resumed run
  // uses must have the new one, as a fresh clone does.
  const versionsSeen: Array<string | undefined> = [];
  const registeredAgent = new RegisteredVersionAgent({
    name: 'RegisteredVersionAgent',
    model: 'gpt-4o-mini',
    instructions: (_runContext, current) => {
      versionsSeen.push(registeredVersions.get(current));
      return 'Set the value.';
    },
    tools: [setAfter],
  });
  registeredVersions.set(registeredAgent, 'before');
  const pausedRegistered = await runner.run(registeredAgent, 'go', { context: { original: 'registered', writes: [] } });
  pausedRegistered.state.approve(pausedRegistered.interruptions[0]!);
  registeredVersions.set(registeredAgent, 'after');
  const registered = await runner.run(registeredAgent, pausedRegistered.state);

  // The run pauses on a handoff target's call, and the approved call throws once its resume has given the
  // target a new object, because its class overrides clone or the target was replaced. The call is then
  // always rejected through the original state, and the retry of that state must not run it again, even
  // if the call was approved for good. A state saved as a string and rehydrated onto the original agent
  // must not run it again either.
  const rejectedExecutions: string[] = [];
  const rejectable = tool({
    name: 'rejectable',
    description: 'Fail once after approval.',
    parameters: {
      type: 'object' as const,
      properties: { value: { type: 'string' } },
      required: ['value'] as const,
      additionalProperties: false as const,
    },
    needsApproval: true,
    errorFunction: null,
    execute: async (args) => {
      const { value } = args as { value: string };
      const firstRun = !rejectedExecutions.includes(value);
      rejectedExecutions.push(value);
      if (firstRun) throw new Error('rejectable failed');
      return 'ok';
    },
  });
  const rejectAfterFailedResume = async (
    child: Agent<ResumeContext>,
    replacement?: Agent<ResumeContext>,
    alwaysApprove = false
  ) => {
    const root = new Agent<ResumeContext>({ name: `${child.name}Root`, model: 'gpt-4o-mini', handoffs: [child] });
    const paused = await runner.run(root, 'go', { context: { original: child.name, writes: [] } });
    const interruption = paused.interruptions[0]!;
    paused.state.approve(interruption, { alwaysApprove });
    if (replacement !== undefined) root.handoffs = [replacement];
    let threw = false;
    try {
      await runner.run(root, paused.state);
    } catch {
      threw = true;
    }
    paused.state.reject(interruption, { alwaysReject: true });
    const retried = await runner.run(root, paused.state);
    return { threw, interruptions: retried.interruptions.length };
  };
  const rejectChild = new Agent<ResumeContext>({ name: 'RejectChild', model: 'gpt-4o-mini', tools: [rejectable] });
  const alwaysChild = new Agent<ResumeContext>({ name: 'AlwaysChild', model: 'gpt-4o-mini', tools: [rejectable] });
  const rejected = [
    await rejectAfterFailedResume(
      new VersionedAgent({ name: 'RejectCustomChild', model: 'gpt-4o-mini', tools: [rejectable] }, 'v')
    ),
    await rejectAfterFailedResume(rejectChild, rejectChild.clone({})),
    await rejectAfterFailedResume(alwaysChild, alwaysChild.clone({}), true),
  ];
  const persistedAgent = new Agent<ResumeContext>({
    name: 'PersistedAgent',
    model: 'gpt-4o-mini',
    tools: [rejectable],
  });
  const pausedPersisted = await runner.run(persistedAgent, 'go', { context: { original: 'persisted', writes: [] } });
  const persistedState = await RunState.fromString(persistedAgent, pausedPersisted.state.toString());
  const persistedInterruption = persistedState.getInterruptions()[0]!;
  persistedState.approve(persistedInterruption, { alwaysApprove: true });
  let persistedThrew = false;
  try {
    await runner.run(persistedAgent, persistedState);
  } catch {
    persistedThrew = true;
  }
  persistedState.reject(persistedInterruption, { alwaysReject: true });
  const persistedRetried = await runner.run(persistedAgent, persistedState);
  rejected.push({ threw: persistedThrew, interruptions: persistedRetried.interruptions.length });

  // A saved state is rehydrated onto the original agent. Its restore merges the approvals, then fails in a
  // tool's isEnabled check, three times. Neither the failures nor the resume that then succeeds may add a
  // copy of the approvals.
  const guard = { fail: false };
  const guarded = tool({
    name: 'guarded',
    description: 'Record a value after approval.',
    parameters: {
      type: 'object' as const,
      properties: { value: { type: 'string' } },
      required: ['value'] as const,
      additionalProperties: false as const,
    },
    needsApproval: true,
    isEnabled: async () => {
      if (guard.fail) throw new Error('isEnabled failed');
      return true;
    },
    execute: async (args, runContext) => {
      (runContext!.context as ResumeContext).writes.push((args as { value: string }).value);
      return 'ok';
    },
  });
  const guardedAgent = new Agent<ResumeContext>({ name: 'GuardedAgent', model: 'gpt-4o-mini', tools: [guarded] });
  const pausedGuarded = await runner.run(guardedAgent, 'go', { context: { original: 'guarded', writes: [] } });
  pausedGuarded.state.approve(pausedGuarded.interruptions[0]!);
  const guardedState = await RunState.fromString(guardedAgent, pausedGuarded.state.toString());
  guard.fail = true;
  let failedRestores = 0;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await runner.run(guardedAgent, guardedState);
    } catch {
      failedRestores++;
    }
  }
  const failedRestoreOwners = countApprovalOwners(guardedState._context);
  guard.fail = false;
  const recovered = await runner.run(guardedAgent, guardedState);
  const recoveredOwners = countApprovalOwners(guardedState._context);

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
    rehydratedWrites: rehydratedCtx.writes,
    rehydratedApprovalOwners: rehydratedOwners.approvalOwners,
    rehydratedApprovedCallIds: rehydratedOwners.approvedCallIds,
    handoffTarget: routed.lastAgent?.name,
    handoffFilters,
    disabledHandoffThrew,
    sharedFieldsWrites: sharedFieldsCtx.writes,
    sharedFieldsTemperatures,
    failedConversionThrew,
    failedConversionWrites: failedConversionResumed.runContext.context.writes,
    failedConversionInterruptions: failedConversionResumed.interruptions.length,
    versions,
    lastAgentVersion: (versioned.lastAgent as VersionedAgent | undefined)?.version,
    customChildWrites: customChildCtx.writes,
    customChildInterruptions: customChildResumed.interruptions.length,
    customChildSavedApprovalOwners: countApprovalOwners(customChildSaved._context).approvalOwners,
    replacedChildWrites: replacedChildCtx.writes,
    replacedChildInterruptions: replacedChildResumed.interruptions.length,
    failedRestoreThrew,
    failedRestoreWrites: failedRestoreResumed.runContext.context.writes,
    failedRestoreInterruptions: failedRestoreResumed.interruptions.length,
    registeredVersions: versionsSeen,
    lastAgentRegisteredVersion: registered.lastAgent && registeredVersions.get(registered.lastAgent),
    rejectedResumesThrew: rejected.map(({ threw }) => threw),
    rejectedExecutions,
    rejectedInterruptions: rejected.reduce((count, { interruptions }) => count + interruptions, 0),
    failedRestores,
    failedRestoreApprovalOwners: failedRestoreOwners.approvalOwners,
    failedRestoreApprovedCallIds: failedRestoreOwners.approvedCallIds,
    recoveredWrites: recovered.runContext.context.writes,
    recoveredApprovalOwners: recoveredOwners.approvalOwners,
    recoveredApprovedCallIds: recoveredOwners.approvedCallIds,
  };
}
