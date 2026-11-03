// First, so its module-scope tool is made before the plugin's Workflow module loads.
// eslint-disable-next-line import/order
import { moduleScopeTool } from './module-scope-agent-tool';
import {
  Agent,
  Handoff,
  MemorySession,
  RunState,
  getGlobalTraceProvider,
  setDefaultModelProvider,
  tool,
  toolNamespace,
  withTrace,
  withTraceContext,
  type MCPToolFilterContext,
  type MCPToolMetaContext,
  type Model,
  type ModelProvider,
  type ModelSettings,
  type Tool,
} from '@openai/agents-core';
import { SandboxAgent } from '@openai/agents-core/sandbox';
import {
  ActivityFailure,
  ApplicationFailure,
  CancellationScope,
  condition,
  continueAsNew,
  defineSignal,
  isCancellation,
  proxyActivities,
  setHandler,
  sleep,
} from '@temporalio/workflow';
import { WorkflowStream } from '@temporalio/workflow-streams/workflow';
import {
  TemporalOpenAIRunner,
  activityAsTool,
  agentAsTool,
  statelessMcpServer,
  statefulMcpServer,
  temporalSandboxClient,
} from '../../workflow';

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
  // The run pauses on an approval. The resumed run keeps this context, as upstream, and the filter
  // throws when it lists the tools again.
  const context = { session: new FilterSession(), resumed: false };
  const toolFilter = async ({ runContext }: MCPToolFilterContext) => {
    const { session, resumed } = runContext.context as typeof context;
    if (resumed) throw new Error('the tool filter failed on resume');
    return session.isAdmin();
  };
  const agent = new Agent({
    name: 'FirstAgent',
    instructions: 'You have access to MCP tools.',
    model: 'gpt-4o-mini',
    mcpServers: [statelessMcpServer('filterStateless', { toolFilter })],
    tools: [
      tool({
        name: 'confirm',
        description: 'Confirm.',
        parameters: { type: 'object' as const, properties: {}, required: [], additionalProperties: false as const },
        strict: true,
        needsApproval: true,
        execute: async () => 'confirmed',
      }),
    ],
  });
  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, 'Hello', { context });
  for (const interruption of result.interruptions) result.state.approve(interruption);
  context.resumed = true;
  await runner.run(agent, result.state);
  return 'resumed';
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

function failureType(err: unknown): string {
  const failure = err instanceof ActivityFailure ? err.cause : err;
  return failure instanceof ApplicationFailure ? `${failure.type}:${failure.nonRetryable}` : String(err);
}

export async function statefulMcpFactoryResolverWorkflow(): Promise<string[]> {
  const server = statefulMcpServer('factoryResolver');
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
  | 'undefined-tool'
  | 'undefined-handoff';

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
    // For example from a circular import.
    'undefined-tool': () => undefined as unknown as Tool<any>,
    // With an `undefined` handoff, set below.
    'undefined-handoff': () => nested.asTool({ toolName: 'ask_nested' }),
  };
  const lead = new Agent({
    name: 'Lead',
    instructions: 'Use the tool.',
    model: 'lead-model',
    tools: [tools[kind]()],
  });
  // The Agent constructor rejects an `undefined` handoff, so one can only appear later.
  if (kind === 'undefined-handoff') lead.handoffs = [undefined as unknown as Agent];
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

export type RunningAgentRejection =
  | 'handoff-callback-model-object'
  | 'handoff-callback-model-provider'
  | 'agent-start-model-object'
  | 'nested-agent-start-model-object'
  | 'tool-start-model-provider'
  | 'tool-start-run-model-object'
  | 'input-builder-model-provider'
  | 'tool-start-clear-model-provider'
  | 'tool-start-memory-session'
  | 'tool-start-tracing-key'
  | 'tool-start-run-config-tracing-key'
  | 'tool-option-model-provider-undefined'
  | 'tool-start-run-model-name';

/**
 * An agent the run reaches only as it goes, through a custom `Handoff`'s callback or an `agent_start`
 * listener, with a `Model` object or an `asTool()` tool that would call a model in the Workflow.
 * The `tool-start-*` and `input-builder-*` kinds change an `asTool()` tool's `runConfig` after the
 * lead's model call, when the tool is invoked; `tool-start-run-model-name` is the supported control.
 * The `*-model-provider-undefined` and `-clear-` kinds remove the runner's provider with an explicit
 * `modelProvider: undefined`, which upstream spreads over the inherited one; a registered default
 * provider would then run the inline model. `tool-start-memory-session` sets a `MemorySession` in the
 * tool's `runOptions`, and the `*-tracing-key` kinds a tracing key in its `runOptions` or `runConfig`.
 * Returns the run's failure, or its output when a nested run's failure became the tool's output.
 */
export async function runningAgentRejectedWorkflow(kind: RunningAgentRejection): Promise<string> {
  const inlineModel: Model = {
    async getResponse() {
      throw new Error('the inline model ran in the Workflow');
    },
    // eslint-disable-next-line require-yield
    async *getStreamedResponse() {
      throw new Error('the inline model ran in the Workflow');
    },
  };
  const declared = new Agent({ name: 'Declared', instructions: 'Answer as declared.', model: 'declared-model' });
  const nested = new Agent({ name: 'Nested', instructions: 'Answer as nested.', model: 'nested-model' });
  let lead: Agent<any, any>;
  if (kind === 'handoff-callback-model-object' || kind === 'handoff-callback-model-provider') {
    const picked =
      kind === 'handoff-callback-model-object'
        ? new Agent({ name: 'Picked', model: inlineModel })
        : new Agent({
            name: 'Picked',
            model: 'picked-model',
            tools: [
              nested.asTool({ toolName: 'ask_nested', runConfig: { modelProvider: { getModel: () => inlineModel } } }),
            ],
          });
    const desk = new Handoff(declared, async () => picked);
    desk.toolName = 'transfer_to_desk';
    lead = new Agent({ name: 'Lead', instructions: 'Route.', model: 'lead-model', handoffs: [desk] });
  } else if (kind === 'agent-start-model-object') {
    lead = new Agent({ name: 'Lead', instructions: 'Route.', model: 'lead-model' });
    lead.on('agent_start', () => (lead.model = inlineModel));
  } else if (
    kind === 'tool-start-model-provider' ||
    kind === 'tool-start-run-model-object' ||
    kind === 'input-builder-model-provider' ||
    kind === 'tool-start-clear-model-provider' ||
    kind === 'tool-start-memory-session' ||
    kind === 'tool-start-tracing-key' ||
    kind === 'tool-start-run-config-tracing-key' ||
    kind === 'tool-option-model-provider-undefined' ||
    kind === 'tool-start-run-model-name'
  ) {
    setDefaultModelProvider({ getModel: () => inlineModel });
    // The run model reaches `Nested` only through the tool's runConfig, which the changes replace.
    const modelLess = new Agent({ name: 'Nested', instructions: 'Answer as nested.' });
    const runConfig: { model?: string | Model; modelProvider?: ModelProvider; tracing?: { apiKey: string } } =
      kind === 'tool-option-model-provider-undefined'
        ? { model: 'nested-run-model', modelProvider: undefined }
        : { model: 'nested-run-model' };
    const runOptions: { session?: MemorySession; tracing?: { apiKey: string } } = {};
    const change = () => {
      if (kind === 'tool-start-run-model-object') runConfig.model = inlineModel;
      else if (kind === 'tool-start-memory-session') runOptions.session = new MemorySession();
      else if (kind === 'tool-start-tracing-key') runOptions.tracing = { apiKey: 'sk-tool' };
      else if (kind === 'tool-start-run-config-tracing-key') runConfig.tracing = { apiKey: 'sk-tool' };
      else if (kind === 'tool-start-run-model-name') runConfig.model = 'changed-run-model';
      else if (kind === 'tool-start-clear-model-provider') runConfig.modelProvider = undefined;
      else if (kind !== 'tool-option-model-provider-undefined')
        runConfig.modelProvider = { getModel: () => inlineModel };
    };
    lead = new Agent({
      name: 'Lead',
      instructions: 'Use the tool.',
      model: 'lead-model',
      tools: [
        modelLess.asTool({
          toolName: 'ask_nested',
          runConfig,
          runOptions,
          ...(kind === 'input-builder-model-provider'
            ? {
                inputBuilder: ({ params }: { params: unknown }) => {
                  change();
                  return (params as { input: string }).input;
                },
              }
            : {}),
        }),
      ],
      toolUseBehavior: 'stop_on_first_tool',
    });
    if (kind !== 'input-builder-model-provider') lead.on('agent_tool_start', change);
  } else {
    nested.on('agent_start', () => (nested.model = inlineModel));
    lead = new Agent({
      name: 'Lead',
      instructions: 'Use the tool.',
      model: 'lead-model',
      tools: [nested.asTool({ toolName: 'ask_nested' })],
      toolUseBehavior: 'stop_on_first_tool',
    });
  }
  try {
    return (await new TemporalOpenAIRunner().run(lead, 'go')).finalOutput ?? '';
  } catch (err: unknown) {
    if (err instanceof ApplicationFailure) return `${err.type}: ${err.message}`;
    throw err;
  }
}

/**
 * The lead calls `Nested` through `asTool()`. With `stopOnFirstTool`, the run ends on the tool's
 * output, so a cancellation that `asTool()` returned as text would otherwise complete the run. With
 * `maxTurns: 1`, the run fails with another error after the tool instead.
 */
export async function nestedToolCancelWorkflow(input: {
  stopOnFirstTool: boolean;
  stream: boolean;
  maxTurns?: number;
}): Promise<string> {
  new WorkflowStream();
  const nested = new Agent({ name: 'Nested', instructions: 'Answer as nested.', model: 'nested-model' });
  const lead = new Agent({
    name: 'Lead',
    instructions: 'Use the tool.',
    model: 'lead-model',
    tools: [nested.asTool({ toolName: 'ask_nested' })],
    toolUseBehavior: input.stopOnFirstTool ? 'stop_on_first_tool' : 'run_llm_again',
  });
  const runner = new TemporalOpenAIRunner({ defaultModelParams: { streamingTopic: 'events' } });
  if (!input.stream) return (await runner.run(lead, 'go', { maxTurns: input.maxTurns })).finalOutput ?? '';
  const result = await runner.run(lead, 'go', { stream: true, maxTurns: input.maxTurns });
  for await (const _event of result);
  await result.completed;
  return result.finalOutput ?? '';
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

export type CopiedAgentToolKind = 'original' | 'alias' | 'spread' | 'namespace' | 'override' | 'alias-override';

export interface CopiedAgentToolApprovalResult {
  output: string;
  executions: number;
  resumes: number;
  /** The interruptions of each pause, in order. */
  paused: string[][];
  pending: string[];
}

/** An `Agent` subclass whose `asTool()` tools need approval before the nested run. */
class ApprovedAgent<TContext = unknown> extends Agent<TContext, any> {
  override asTool(options: any): any {
    return super.asTool({ ...options, needsApproval: true });
  }
}

/**
 * `Nested` pauses on an approval behind `ask_nested`: the `asTool()` tool, the deprecated alias's
 * tool, a spread copy or a `toolNamespace()` copy. Approves it on `result.state` and resumes, at
 * most three times. With `override` and `alias-override`, `Nested` is an `ApprovedAgent`, so the
 * parent pauses on `ask_nested` itself first.
 */
export async function copiedAgentToolApprovalWorkflow(
  kind: CopiedAgentToolKind
): Promise<CopiedAgentToolApprovalResult> {
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
    execute: async () => `confirmed:${++executions}`,
  });
  const nestedConfig = { name: 'Nested', instructions: 'Answer as nested.', tools: [confirmAction] };
  const nested =
    kind === 'override' || kind === 'alias-override' ? new ApprovedAgent(nestedConfig) : new Agent(nestedConfig);
  const askNested =
    kind === 'alias' || kind === 'alias-override'
      ? // eslint-disable-next-line @typescript-eslint/no-deprecated -- tests the deprecated alias
        agentAsTool(nested, { toolName: 'ask_nested' })
      : nested.asTool({ toolName: 'ask_nested' });
  const parentTool =
    kind === 'spread'
      ? { ...askNested }
      : kind === 'namespace'
        ? toolNamespace({ name: 'desk', description: 'The desk.', tools: [askNested] })[0]!
        : askNested;
  const parent = new Agent({ name: 'Parent', instructions: 'Reach nested.', tools: [parentTool] });
  const runner = new TemporalOpenAIRunner();
  let result = await runner.run(parent, 'go', { runConfig: { model: 'run-model' } });
  let resumes = 0;
  const paused: string[][] = [];
  while (result.interruptions.length > 0 && resumes < 3) {
    paused.push(result.interruptions.map((interruption) => interruption.name ?? ''));
    for (const interruption of result.interruptions) result.state.approve(interruption);
    result = await runner.run(parent, result.state, { runConfig: { model: 'run-model' } });
    resumes++;
  }
  return {
    output: result.interruptions.length > 0 ? '' : (result.finalOutput ?? ''),
    executions,
    resumes,
    paused,
    pending: result.interruptions.map((interruption) => interruption.name ?? ''),
  };
}

export interface AgentToolStreamResult {
  output: string;
  onStream: string[];
  on: string[];
}

/**
 * A nested `asTool()` run streams to its `onStream` option and `on()` handlers. With `onAfterStart`,
 * the `on()` handler is registered after the run starts, while the lead's model call is pending.
 * With `useLocalActivity`, model calls are local Activities, which can't stream.
 */
export async function agentToolStreamWorkflow(
  prompt: string,
  onAfterStart = false,
  useLocalActivity = false
): Promise<AgentToolStreamResult> {
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
  const register = () =>
    askStreamer.on('raw_model_stream_event', ({ event }) => {
      if (event.type === 'raw_model_stream_event' && event.data.type === 'output_text_delta') {
        on.push(event.data.delta);
      }
    });
  if (!onAfterStart) register();
  const lead = new Agent({ name: 'Lead', instructions: 'Use the tool.', model: 'lead-model', tools: [askStreamer] });
  const runner = new TemporalOpenAIRunner({ defaultModelParams: { streamingTopic: 'events', useLocalActivity } });
  const running = runner.run(lead, prompt);
  if (onAfterStart) register();
  const result = await running;
  return { output: result.finalOutput ?? '', onStream, on };
}

/**
 * Creates `ask_gated` with approval required, then reuses its options object for `ask_open`, and
 * renames an agent after creating its tool with the default name. Also passes `ask_early`, made at
 * module scope before the plugin's Workflow module loaded. Returns the pending approvals.
 */
export async function agentToolOptionsReuseWorkflow(prompt: string): Promise<string[]> {
  const nested = new Agent({ name: 'Nested', instructions: 'Answer as nested.', model: 'nested-model' });
  const options: { toolName: string; needsApproval: boolean } = { toolName: 'ask_gated', needsApproval: true };
  const gated = nested.asTool(options);
  options.toolName = 'ask_open';
  options.needsApproval = false;
  const open = nested.asTool(options);
  const renamed = new Agent({ name: 'Original Name', instructions: 'Answer as renamed.', model: 'nested-model' });
  const byDefaultName = renamed.asTool({});
  renamed.name = 'Updated Name';
  const lead = new Agent({
    name: 'Lead',
    instructions: 'Use the tools.',
    model: 'lead-model',
    tools: [gated, open, byDefaultName, moduleScopeTool],
  });
  const result = await new TemporalOpenAIRunner().run(lead, prompt);
  return result.interruptions.map((i) => i.name ?? '');
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

export async function statefulMcpCancelledAfterRefusalWorkflow(
  mode: 'finally' | 'nonCancellable' | 'stop'
): Promise<string> {
  // The session times out and its retry is refused while the Workflow waits.
  const server = statefulMcpServer('refusedCancel', { serverSessionConfig: { startToCloseTimeout: '5 seconds' } });
  await server.connect();
  if (mode === 'stop') {
    // A scope between connect() and cleanup() that a Signal cancels.
    const stop = new CancellationScope();
    setHandler(resumeSignal, () => stop.cancel());
    try {
      await stop.run(async () => {
        try {
          await server.listTools();
          await condition(() => false);
        } finally {
          await CancellationScope.nonCancellable(() => server.cleanup());
        }
      });
    } catch (err: unknown) {
      if (isCancellation(err)) return 'stopped';
      throw err;
    }
    return 'unreachable';
  }
  try {
    await server.listTools();
    await condition(() => false);
    return 'unreachable';
  } finally {
    // After the Workflow's cancellation, this must not replace it with the
    // session's DedicatedWorkerFailure, also when it runs in a scope that
    // can't be cancelled.
    if (mode === 'nonCancellable') {
      await CancellationScope.nonCancellable(() => server.cleanup());
    } else {
      await server.cleanup();
    }
  }
}

export interface AgentHooksResult {
  events: string[];
  nextTurnOutput: string;
}

/**
 * `agent.on(...)` listeners, as in the upstream lifecycle hooks guide. The tool-end
 * listener writes the tool result into the run context, and a second turn runs only
 * when it did, so the Workflow's commands depend on the hook. With `guard`, an `agent_start`
 * listener throws in the second turn.
 */
export async function agentHooksWorkflow(guard = false): Promise<AgentHooksResult> {
  const fetchValue = activityAsTool<(input: { value: string }) => Promise<string>>({
    name: 'fetchValue',
    description: 'Fetch a value',
    parameters: {
      type: 'object',
      properties: { value: { type: 'string' } },
      required: ['value'],
      additionalProperties: false,
    },
  });
  const agent = new Agent<{ hookValue?: string }>({
    name: 'HookAgent',
    instructions: (runContext) => `hook=${runContext.context.hookValue ?? ''}`,
    model: 'gpt-4o-mini',
    tools: [fetchValue],
  });
  const events: string[] = [];
  agent.on('agent_start', (_ctx, started) => events.push(`agent_start:${started === agent}`));
  agent.on('agent_tool_end', (runContext, tool, result) => {
    events.push(`agent_tool_end:${tool.name}`);
    runContext.context.hookValue = result;
  });
  agent.on('agent_end', (_ctx, output) => events.push(`agent_end:${output}`));
  if (guard) {
    agent.on('agent_start', (runContext) => {
      if (runContext.context.hookValue !== undefined) throw new Error('a guard stopped the run');
    });
  }

  const runner = new TemporalOpenAIRunner();
  const context: { hookValue?: string } = {};
  await runner.run(agent, 'go', { context });
  if (context.hookValue === undefined) return { events, nextTurnOutput: '' };
  const next = await runner.run(agent, 'again', { context });
  return { events, nextTurnOutput: next.finalOutput ?? '' };
}

/**
 * A custom `Handoff` whose callback picks the desk from the run context, as the
 * upstream handoffs guide describes. Each desk has its own Activity tool.
 */
export async function customHandoffWorkflow(tier: string): Promise<string> {
  const deskTool = (name: string) =>
    activityAsTool<(input: Record<string, never>) => Promise<string>>({
      name,
      description: `Look up the ${name} desk`,
      parameters: { type: 'object', properties: {}, required: [], additionalProperties: false },
    });
  const standard = new Agent<{ tier: string }>({
    name: 'standard',
    model: 'gpt-4o-mini',
    tools: [deskTool('standardDesk')],
  });
  const vip = new Agent<{ tier: string }>({ name: 'vip', model: 'gpt-4o-mini', tools: [deskTool('vipDesk')] });
  const desk = new Handoff(standard, async (runContext) => (runContext.context.tier === 'vip' ? vip : standard));
  desk.toolName = 'transfer_to_desk';
  desk.toolDescription = 'Route to the right desk';
  const router = new Agent<{ tier: string }>({ name: 'router', model: 'gpt-4o-mini', handoffs: [desk] });
  const result = await new TemporalOpenAIRunner().run(router, 'help', { context: { tier } });
  return `${result.lastAgent?.name}: ${result.finalOutput}`;
}

/**
 * A summary provider labels each model call with the agent it gets: lead, nested expert, and
 * billing, before and after its approval resumes. A custom `Handoff`'s callback picks billing, which
 * the graph doesn't declare.
 */
export async function summaryAgentWorkflow(): Promise<string> {
  const expert = new Agent({ name: 'expert', model: 'm', instructions: 'Answer as the expert.' });
  const refund = tool({
    name: 'refund',
    description: 'Refund the order.',
    parameters: { type: 'object' as const, properties: {}, required: [], additionalProperties: false as const },
    strict: true,
    needsApproval: true,
    execute: async () => 'refunded',
  });
  const billing = new Agent({ name: 'billing', model: 'm', instructions: 'Answer as billing.', tools: [refund] });
  const support = new Agent({ name: 'support', model: 'm', instructions: 'Answer as support.' });
  const desk = new Handoff(support, async () => billing);
  desk.toolName = 'transfer_to_billing';
  const lead = new Agent({
    name: 'lead',
    model: 'm',
    instructions: 'Use the tools.',
    tools: [expert.asTool({ toolName: 'ask_expert' })],
    handoffs: [desk],
  });
  const runner = new TemporalOpenAIRunner({
    defaultModelParams: { summary: { provide: (agent) => `agent:${agent?.name ?? 'none'}` } },
  });
  const first = await runner.run(lead, 'go');
  for (const interruption of first.interruptions) first.state.approve(interruption);
  return (await runner.run(lead, first.state)).finalOutput ?? '';
}

export interface ImplicitSettingsCase {
  runSettings: ModelSettings;
  /** Run `Unnamed` as the `asTool()` tool of a lead with this model. */
  leadModel?: string;
  /** `Unnamed`'s own `modelSettings`. */
  agentSettings?: ModelSettings;
  /** Clear the `reasoning` and `text` defaults with run settings set to `undefined`. */
  clearDefaults?: boolean;
  /** Clear the `reasoning.effort` and `text.verbosity` defaults with nested run settings set to `undefined`. */
  clearNestedDefaults?: boolean;
  /** `Unnamed` calls its `echo` tool before it answers. */
  callsTool?: boolean;
}

/** `Unnamed` names no model, in a run with its own model settings, run directly or as an `asTool()` tool. */
export async function implicitModelSettingsWorkflow(input: ImplicitSettingsCase): Promise<string> {
  const echo = tool({
    name: 'echo',
    description: 'Echo.',
    parameters: { type: 'object' as const, properties: {}, required: [], additionalProperties: false as const },
    strict: true,
    execute: async () => 'echoed',
  });
  const unnamed = new Agent({
    name: 'Unnamed',
    instructions: input.callsTool ? 'Use echo, then answer.' : 'Answer.',
    ...(input.agentSettings && { modelSettings: input.agentSettings }),
    ...(input.callsTool && { tools: [echo] }),
  });
  const agent =
    input.leadModel === undefined
      ? unnamed
      : new Agent({
          name: 'Lead',
          instructions: 'Use the tool.',
          model: input.leadModel,
          tools: [unnamed.asTool({ toolName: 'ask_unnamed' })],
          toolUseBehavior: 'stop_on_first_tool',
        });
  let modelSettings = input.runSettings;
  if (input.clearDefaults) modelSettings = { ...modelSettings, reasoning: undefined, text: undefined };
  if (input.clearNestedDefaults)
    modelSettings = { ...modelSettings, reasoning: { effort: undefined }, text: { verbosity: undefined } };
  return (await new TemporalOpenAIRunner().run(agent, 'go', { runConfig: { modelSettings } })).finalOutput ?? '';
}

/** `runConfig.model` with an agent that sets its own model. */
export async function runModelOverrideWorkflow(): Promise<string> {
  const agent = new Agent({ name: 'own', model: 'own-model', instructions: 'Answer.' });
  return (await new TemporalOpenAIRunner().run(agent, 'go', { runConfig: { model: 'run-model' } })).finalOutput ?? '';
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
  | 'activity'
  | 'asToolRunOptions'
  | 'asToolRunConfig';

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
      case 'asToolRunOptions':
        agent.asTool({ toolName: 'ask', runOptions: { tracing: { apiKey: 'sk-tool' } } });
        break;
      case 'asToolRunConfig':
        agent.asTool({ toolName: 'ask', runConfig: { tracing: { apiKey: 'sk-tool' } } });
        break;
    }
  } catch (e) {
    return `refused: ${(e as Error).message}`;
  }
  return 'not refused';
}
