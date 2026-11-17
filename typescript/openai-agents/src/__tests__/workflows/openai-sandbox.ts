import { Agent, Handoff, RunState, tool, type FunctionTool, type RunContext, type Tool } from '@openai/agents-core';
import {
  Capability,
  Manifest,
  SandboxAgent,
  isEnvValueReference,
  type SandboxClient,
  type SandboxRunConfig,
  type SandboxSessionLike,
  type SandboxSessionState,
} from '@openai/agents-core/sandbox';
import {
  ActivityFailure,
  ApplicationFailure,
  condition,
  continueAsNew,
  defineQuery,
  defineSignal,
  setHandler,
} from '@temporalio/workflow';
import { TemporalOpenAIRunner, workerEnvValue, temporalSandboxClient } from '../../workflow';
import { serializeSessionEnvelope, type TemporalSandboxSessionState } from '../../common/sandbox-activity-types';

class TestSandboxCapability extends Capability {
  readonly type = 'test_sandbox';

  private session(): SandboxSessionLike {
    if (!this._session) throw new Error('test_sandbox capability is not bound to a session');
    return this._session;
  }

  override tools(): Tool<any>[] {
    const makeTool = (
      name: string,
      properties: Record<string, unknown>,
      required: string[],
      execute: (args: any) => Promise<string>
    ): FunctionTool =>
      ({
        type: 'function',
        name,
        description: name,
        parameters: { type: 'object', properties, required, additionalProperties: false } as any,
        strict: true,
        invoke: async (_ctx: RunContext<any>, input: string): Promise<string> => execute(JSON.parse(input)),
        needsApproval: async () => false,
        isEnabled: async () => true,
      }) as FunctionTool;

    return [
      makeTool('run_command', { cmd: { type: 'string' } }, ['cmd'], async ({ cmd }) => {
        return this.session().execCommand!({ cmd });
      }),
      makeTool('read_file', { path: { type: 'string' } }, ['path'], async ({ path }) => {
        const data = await this.session().readFile!({ path });
        return typeof data === 'string' ? data : new TextDecoder().decode(data);
      }),
      makeTool(
        'write_file',
        { path: { type: 'string' }, diff: { type: 'string' } },
        ['path', 'diff'],
        async ({ path, diff }) => {
          const editor = this.session().createEditor!();
          const result = await editor.createFile({ type: 'create_file', path, diff });
          return result?.output ?? 'ok';
        }
      ),
    ];
  }
}

export async function sandboxAgentWorkflow(): Promise<string> {
  const agent = new SandboxAgent({
    name: 'sandbox-e2e',
    model: 'gpt-4o-mini',
    capabilities: [new TestSandboxCapability()],
    defaultManifest: new Manifest({
      entries: { 'data.bin': { type: 'file', content: new Uint8Array([0, 1, 2, 253, 254, 255]) } },
    }),
  });
  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, 'run a command', {
    runConfig: { sandbox: { client: temporalSandboxClient('fake') } },
  });
  return `${result.finalOutput}`;
}

/**
 * A sandbox tool that interrupts for approval, so the run preserves its owned session. It records
 * its call in the run context's `writes`.
 */
class ApprovalSandboxCapability extends Capability {
  readonly type = 'approval_sandbox';

  override tools(): Tool<any>[] {
    return [
      {
        type: 'function',
        name: 'run_command',
        description: 'run_command',
        parameters: {
          type: 'object',
          properties: { cmd: { type: 'string' } },
          required: ['cmd'],
          additionalProperties: false,
        } as any,
        strict: true,
        invoke: async (ctx: RunContext<{ writes: string[] }>, input: string): Promise<string> => {
          if (!this._session) throw new Error('approval_sandbox capability is not bound to a session');
          ctx.context.writes.push('sandbox-tool');
          return this._session.execCommand!(JSON.parse(input));
        },
        needsApproval: async () => true,
        isEnabled: async () => true,
      } as FunctionTool,
    ];
  }
}

/**
 * Pauses for approval and resumes, with `serialized` from the paused state's string. Returns the output and
 * the tool calls the caller's context object got.
 */
export async function sandboxApprovalResumeWorkflow(serialized = false): Promise<string> {
  const agent = new SandboxAgent({
    name: 'sandbox-approval',
    model: 'gpt-4o-mini',
    capabilities: [new ApprovalSandboxCapability()],
    defaultManifest: new Manifest({
      environment: { API_KEY: workerEnvValue('OPENAI_AGENTS_TEST_MANIFEST_SECRET') },
    }),
  });
  const runner = new TemporalOpenAIRunner();
  const runConfig = { sandbox: { client: temporalSandboxClient('fake') } };

  const context = { writes: [] as string[] };
  const result = await runner.run(agent, 'run a command', { runConfig, context });
  if (result.interruptions.length === 0) return 'no-interruption';
  for (const interruption of result.interruptions) result.state.approve(interruption);
  const state = serialized ? await RunState.fromString(agent, result.state.toString()) : result.state;

  const resumed = await runner.run(agent, state, { runConfig });
  return `${resumed.finalOutput} writes=${context.writes.join(',')}`;
}

/**
 * Pauses for approval and resumes from the paused state's string. With `changed`, the resume's agent names
 * another Worker variable; with `extra-reference`, the string gains a reference that the agent's manifest
 * doesn't have.
 */
export async function sandboxResumeEnvironmentWorkflow(mode: 'changed' | 'extra-reference'): Promise<string> {
  const agentNaming = (variable: string) =>
    new SandboxAgent({
      name: 'sandbox-approval',
      model: 'gpt-4o-mini',
      capabilities: [new ApprovalSandboxCapability()],
      defaultManifest: new Manifest({ environment: { API_KEY: workerEnvValue(variable) } }),
    });
  const agent = agentNaming('OPENAI_AGENTS_TEST_MANIFEST_SECRET');
  const runner = new TemporalOpenAIRunner();
  const runConfig = { sandbox: { client: temporalSandboxClient('fake') } };
  const result = await runner.run(agent, 'run a command', { runConfig, context: { writes: [] as string[] } });
  if (result.interruptions.length === 0) return 'no-interruption';
  for (const interruption of result.interruptions) result.state.approve(interruption);
  const serialized: unknown = JSON.parse(result.state.toString());
  if (mode === 'extra-reference') addEnvironmentReference(serialized);
  const resumeAgent = mode === 'changed' ? agentNaming('OPENAI_AGENTS_TEST_OTHER_SECRET') : agent;
  const state = await RunState.fromString(resumeAgent, JSON.stringify(serialized));
  return `${(await runner.run(resumeAgent, state, { runConfig })).finalOutput}`;
}

/** Adds `EXTRA`, a reference to another Worker variable, next to each serialized `API_KEY` reference. */
function addEnvironmentReference(value: unknown): void {
  if (value === null || typeof value !== 'object') return;
  const environment = (value as { environment?: Record<string, { name?: unknown }> }).environment;
  if (environment?.API_KEY?.name === 'OPENAI_AGENTS_TEST_MANIFEST_SECRET') {
    environment.EXTRA = { ...environment.API_KEY, name: 'OPENAI_AGENTS_TEST_OTHER_SECRET' };
  }
  for (const child of Object.values(value)) addEnvironmentReference(child);
}

/** `run_now` runs a command at once, and `run_approved` after approval. */
class EchoSandboxCapability extends Capability {
  readonly type = 'echo_sandbox';

  constructor(private readonly onSession?: (state: SandboxSessionState) => void) {
    super();
  }

  override tools(): Tool<any>[] {
    const runCommand = (name: string, needsApproval: boolean): FunctionTool =>
      ({
        type: 'function',
        name,
        description: name,
        parameters: {
          type: 'object',
          properties: { cmd: { type: 'string' } },
          required: ['cmd'],
          additionalProperties: false,
        } as any,
        strict: true,
        invoke: async (_ctx: RunContext<any>, input: string): Promise<string> => {
          if (!this._session) throw new Error('echo_sandbox capability is not bound to a session');
          this.onSession?.(this._session.state);
          return this._session.execCommand!(JSON.parse(input));
        },
        needsApproval: async () => needsApproval,
        isEnabled: async () => true,
      }) as FunctionTool;
    return [runCommand('run_now', false), runCommand('run_approved', true)];
  }
}

/**
 * A sandbox whose manifest names the Worker variable `variable` runs a command, pauses for approval of
 * another, and resumes. Returns the outputs of both commands.
 */
export async function sandboxWorkerEnvWorkflow(variable: string): Promise<string> {
  const agent = new SandboxAgent({
    name: 'sandbox-worker-env',
    model: 'gpt-4o-mini',
    capabilities: [new EchoSandboxCapability()],
    defaultManifest: new Manifest({ environment: { API_KEY: workerEnvValue(variable) } }),
  });
  const runner = new TemporalOpenAIRunner();
  const runConfig = { sandbox: { client: temporalSandboxClient('local') } };
  const result = await runner.run(agent, 'run commands', { runConfig });
  if (result.interruptions.length === 0) return 'no-interruption';
  for (const interruption of result.interruptions) result.state.approve(interruption);
  const resumed = await runner.run(agent, result.state, { runConfig });
  return [...result.newItems, ...resumed.newItems]
    .flatMap((item) => (item.type === 'tool_call_output_item' ? [`${item.output}`] : []))
    .join('\n');
}

export const approvalPausedQuery = defineQuery<boolean>('approvalPaused');
export const approvalResumeSignal = defineSignal('approvalResume');

/**
 * Runs a sandbox agent whose manifest names the Worker variable `variables[0]` until it pauses for approval,
 * then approves and resumes from `result.state` once per later variable, each time with a new client and a
 * manifest naming that variable. Waits for `approvalResumeSignal` first if `waitForSignal`. Returns each
 * command's output.
 */
export async function sandboxApprovalSequenceWorkflow(variables: string[], waitForSignal = false): Promise<string> {
  let paused = false;
  let resume = !waitForSignal;
  setHandler(approvalPausedQuery, () => paused);
  setHandler(approvalResumeSignal, () => {
    resume = true;
  });
  const agent = new SandboxAgent({
    name: 'sandbox-approval-sequence',
    model: 'gpt-4o-mini',
    capabilities: [new EchoSandboxCapability()],
  });
  const runConfig = (variable: string) => ({
    sandbox: {
      client: temporalSandboxClient('lifecycle'),
      manifest: new Manifest({ environment: { KEY: workerEnvValue(variable) } }),
    },
  });
  const runner = new TemporalOpenAIRunner();
  let result = await runner.run(agent, 'run commands', { runConfig: runConfig(variables[0]!) });
  paused = true;
  await condition(() => resume);
  for (const variable of variables.slice(1)) {
    for (const interruption of result.interruptions) result.state.approve(interruption);
    result = await runner.run(agent, result.state, { runConfig: runConfig(variable) });
  }
  return result.newItems
    .flatMap((item) => (item.type === 'tool_call_output_item' ? [`${item.output}`] : []))
    .join('\n');
}

/**
 * A sandbox agent runs `run_now` and pauses for approval of `run_approved`. `pause` returns the paused state's string.
 * Otherwise approves and resumes from `result.state` (`live`), from `RunState.fromString` (`runState`), or
 * from that with the session's state as `runConfig.sandbox.sessionState` (`sessionState`). Given `pausedState`,
 * resumes that instead of running first. Returns each command's output.
 */
export async function sandboxRootResumeWorkflow(
  resume: 'pause' | 'live' | 'runState' | 'sessionState',
  pausedState?: string
): Promise<string> {
  let sessionState: SandboxSessionState | undefined;
  const agent = new SandboxAgent({
    name: 'sandbox-root',
    model: 'gpt-4o-mini',
    capabilities: [new EchoSandboxCapability((state) => (sessionState = state))],
  });
  const client = temporalSandboxClient('lifecycle');
  const runner = new TemporalOpenAIRunner();
  const paused = pausedState ? undefined : await runner.run(agent, 'run', { runConfig: { sandbox: { client } } });
  if (resume === 'pause') return paused!.state.toString();
  const state =
    resume === 'live' ? paused!.state : await RunState.fromString(agent, pausedState ?? paused!.state.toString());
  for (const interruption of state.getInterruptions()) state.approve(interruption);
  const resumed = await runner.run(agent, state, {
    runConfig: { sandbox: { client, ...(resume === 'sessionState' ? { sessionState } : {}) } },
  });
  return resumed.newItems
    .flatMap((item) => (item.type === 'tool_call_output_item' ? [`${item.output}`] : []))
    .join('\n');
}

/**
 * Without `saved`, a sandbox agent with its manifest at `root` runs `run_now`, pauses for approval of `run_approved`,
 * and returns its session's serialized state. Given `saved`, a new run resumes that state as
 * `runConfig.sandbox.sessionState` and returns each command's output.
 */
export async function sandboxSavedSessionResumeWorkflow(root: string, saved?: string): Promise<string> {
  const captured: { session?: TemporalSandboxSessionState } = {};
  const agent = new SandboxAgent({
    name: 'sandbox-root',
    model: 'gpt-4o-mini',
    defaultManifest: new Manifest({ root }),
    capabilities: [new EchoSandboxCapability((state) => (captured.session = state as TemporalSandboxSessionState))],
  });
  const client = temporalSandboxClient('lifecycle');
  const runner = new TemporalOpenAIRunner();
  if (saved === undefined) {
    await runner.run(agent, 'run', { runConfig: { sandbox: { client } } });
    const session = captured.session!;
    return JSON.stringify(serializeSessionEnvelope(session.sessionId, session, session.providerState));
  }
  const sessionState = await client.deserializeSessionState(JSON.parse(saved));
  const resumed = await runner.run(agent, 'run', { runConfig: { sandbox: { client, sessionState } } });
  return resumed.newItems
    .flatMap((item) => (item.type === 'tool_call_output_item' ? [`${item.output}`] : []))
    .join('\n');
}

/**
 * A plain agent asks a nested `SandboxAgent` through `asTool()`, which runs `run_now` and pauses for
 * approval of `run_approved`. Returns the paused state's string or, with `continueWithState`,
 * continues as new with it. Given `pausedState`, approves and resumes that instead, and returns each
 * tool's output.
 */
export async function sandboxNestedRootResumeWorkflow(
  pausedState?: string,
  continueWithState = false
): Promise<string> {
  const nested = new SandboxAgent({
    name: 'nested-sandbox',
    model: 'gpt-4o-mini',
    capabilities: [new EchoSandboxCapability()],
  });
  const lead = new Agent({ name: 'lead', model: 'gpt-4o-mini', tools: [nested.asTool({ toolName: 'ask_sandbox' })] });
  const runConfig = { sandbox: { client: temporalSandboxClient('lifecycle') } };
  const runner = new TemporalOpenAIRunner();
  if (pausedState === undefined) {
    const paused = (await runner.run(lead, 'run', { runConfig })).state.toString();
    if (continueWithState) await continueAsNew<typeof sandboxNestedRootResumeWorkflow>(paused);
    return paused;
  }
  const state = await RunState.fromString(lead, pausedState);
  for (const interruption of state.getInterruptions()) state.approve(interruption);
  const resumed = await runner.run(lead, state, { runConfig });
  return resumed.newItems
    .flatMap((item) => (item.type === 'tool_call_output_item' ? [`${item.output}`] : []))
    .join('\n');
}

export const sessionCreatedQuery = defineQuery<boolean>('sessionCreated');
export const proceedSignal = defineSignal('proceed');

/**
 * Creates a session on the `lifecycle` provider, waits for `proceedSignal`, runs each command, and
 * deletes the session. Returns the commands' outputs, or the failure type of the first command that
 * failed.
 */
export async function sandboxOperationsWorkflow(commands: string[]): Promise<string> {
  let created = false;
  let proceed = false;
  setHandler(sessionCreatedQuery, () => created);
  setHandler(proceedSignal, () => {
    proceed = true;
  });
  const session = await temporalSandboxClient('lifecycle').create();
  created = true;
  await condition(() => proceed);
  const outputs: string[] = [];
  try {
    for (const cmd of commands) outputs.push(await session.execCommand!({ cmd }));
  } catch (err) {
    if (err instanceof ActivityFailure && err.cause instanceof ApplicationFailure) {
      outputs.push(`failed: ${err.cause.type}`);
    } else {
      throw err;
    }
  } finally {
    await session.delete!();
  }
  return outputs.join('\n');
}

/**
 * Runs a sandbox agent on the `lifecycle` provider until it pauses for approval, then continues as new
 * with the paused state's string and resumes from it there. Returns the resumed run's command outputs.
 */
export async function sandboxContinueAsNewWorkflow(serialized?: string): Promise<string> {
  const agent = new SandboxAgent({
    name: 'sandbox-continue-as-new',
    model: 'gpt-4o-mini',
    capabilities: [new EchoSandboxCapability()],
  });
  const runner = new TemporalOpenAIRunner();
  const runConfig = { sandbox: { client: temporalSandboxClient('lifecycle') } };
  if (serialized === undefined) {
    const result = await runner.run(agent, 'run commands', { runConfig });
    if (result.interruptions.length === 0) return 'no-interruption';
    for (const interruption of result.interruptions) result.state.approve(interruption);
    return await continueAsNew<typeof sandboxContinueAsNewWorkflow>(result.state.toString());
  }
  const resumed = await runner.run(agent, await RunState.fromString(agent, serialized), { runConfig });
  return resumed.newItems
    .flatMap((item) => (item.type === 'tool_call_output_item' ? [`${item.output}`] : []))
    .join('\n');
}

/** A `SandboxAgent` that names no model, with the default capabilities, which include compaction. */
export async function sandboxNoModelWorkflow(): Promise<string> {
  const agent = new SandboxAgent({ name: 'sandbox-no-model', instructions: 'Answer.' });
  const result = await new TemporalOpenAIRunner().run(agent, 'hello', {
    runConfig: { sandbox: { client: temporalSandboxClient('fake') } },
  });
  return `${result.finalOutput}`;
}

/**
 * A `SandboxAgent` named `parentName` reaches `Nested` through `ask_nested`, the `asTool()` tool or a
 * spread copy, and `Nested` pauses on an approval. Approves it and resumes, with the same agents or, if
 * `rebuilt`, with new ones built the same way. Returns the output and the approved action's runs, or the
 * resumed run's failure.
 */
export async function sandboxNestedApprovalWorkflow(
  copied: boolean,
  parentName = 'sandbox-parent',
  rebuilt = false
): Promise<string> {
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
  const buildAgent = () => {
    const nested = new Agent({
      name: 'Nested',
      instructions: 'Answer as nested.',
      model: 'nested-model',
      tools: [confirmAction],
    });
    const askNested = nested.asTool({ toolName: 'ask_nested' });
    return new SandboxAgent({
      name: parentName,
      instructions: 'Reach nested.',
      model: 'gpt-4o-mini',
      capabilities: [new TestSandboxCapability()],
      tools: [copied ? { ...askNested } : askNested],
    });
  };
  const agent = buildAgent();
  const runner = new TemporalOpenAIRunner();
  const runConfig = { sandbox: { client: temporalSandboxClient('fake') } };

  const result = await runner.run(agent, 'go', { runConfig });
  if (result.state._sandbox === undefined) return 'no-sandbox-state';
  for (const interruption of result.interruptions) result.state.approve(interruption);
  try {
    const resumed = await runner.run(rebuilt ? buildAgent() : agent, result.state, { runConfig });
    return `${resumed.finalOutput} executions=${executions}`;
  } catch (err) {
    if (err instanceof ApplicationFailure) return `${err.type}: ${err.message} executions=${executions}`;
    throw err;
  }
}

export async function sandboxManifestResumeWorkflow(): Promise<string> {
  const client = temporalSandboxClient('fake');
  const session = await client.create(
    new Manifest({
      entries: { 'base.txt': { type: 'file', content: 'base' } },
      environment: { API_KEY: workerEnvValue('OPENAI_AGENTS_TEST_MANIFEST_SECRET') },
    })
  );
  await session.applyManifest!(new Manifest({ entries: { 'added.txt': { type: 'file', content: 'added' } } }));
  const live = 'added.txt' in session.state.manifest.entries;
  const resumed = await client.resume(session.state);
  const persisted = 'added.txt' in resumed.state.manifest.entries;
  const reference = isEnvValueReference(resumed.state.manifest.environment.API_KEY);
  return `live=${live} persisted=${persisted} reference=${reference}`;
}

export async function sandboxExecWorkflow(): Promise<string> {
  const client = temporalSandboxClient('fake');
  const session = await client.create();
  await session.start!();
  const result = await session.exec!({ cmd: 'echo hello' });
  await session.readFile!({ path: '/workspace/out.txt' });
  const editor = session.createEditor!();
  await editor.createFile({ type: 'create_file', path: '/workspace/out.txt', diff: 'secret-content' });
  return `exit=${result.exitCode}`;
}

export async function sandboxArchiveLimitsWorkflow(): Promise<string> {
  const client = temporalSandboxClient('fake');
  const session = await client.create();
  session.setArchiveLimits!({ maxInputBytes: 42 });
  await session.persistWorkspace!();
  await session.hydrateWorkspace!(new Uint8Array([1, 2, 3]));
  return 'ok';
}

export async function sandboxValidationWorkflow(): Promise<string> {
  const runner = new TemporalOpenAIRunner();

  try {
    await runner.run(new SandboxAgent({ name: 'sandbox', model: 'gpt-4o-mini' }), 'hello');
    return 'FAIL: no-config should have thrown';
  } catch (e) {
    if (!/runConfig\.sandbox is not configured/.test((e as Error).message)) {
      return `FAIL: unexpected no-config error: ${(e as Error).message}`;
    }
  }

  try {
    const sandbox = new SandboxAgent({ name: 'sandbox_target', model: 'gpt-4o-mini' });
    const router = new Agent({ name: 'router', model: 'gpt-4o-mini', handoffs: [sandbox] });
    await runner.run(router, 'hello');
    return 'FAIL: handoff-no-config should have thrown';
  } catch (e) {
    if (!/runConfig\.sandbox is not configured/.test((e as Error).message)) {
      return `FAIL: unexpected handoff-no-config error: ${(e as Error).message}`;
    }
  }

  try {
    await runner.run(new SandboxAgent({ name: 'sandbox', model: 'gpt-4o-mini' }), 'hello', {
      runConfig: { sandbox: {} },
    });
    return 'FAIL: null-client should have thrown';
  } catch (e) {
    if (!/runConfig\.sandbox\.client must be set/.test((e as Error).message)) {
      return `FAIL: unexpected null-client error: ${(e as Error).message}`;
    }
  }

  try {
    await runner.run(new SandboxAgent({ name: 'sandbox', model: 'gpt-4o-mini' }), 'hello', {
      runConfig: { sandbox: { client: { backendId: 'raw' } as any } },
    });
    return 'FAIL: raw-client should have thrown';
  } catch (e) {
    if (!/Do not pass a raw sandbox client directly/.test((e as Error).message)) {
      return `FAIL: unexpected raw-client error: ${(e as Error).message}`;
    }
  }

  try {
    await runner.run(new SandboxAgent({ name: 'sandbox', model: 'gpt-4o-mini' }), 'hello', {
      runConfig: { sandbox: { client: temporalSandboxClient('fake'), session: { state: {} } as any } },
    });
    return 'FAIL: raw-session should have thrown';
  } catch (e) {
    if (!/Do not pass a raw sandbox session directly/.test((e as Error).message)) {
      return `FAIL: unexpected raw-session error: ${(e as Error).message}`;
    }
  }

  // As upstream, an asTool() tool returns the error of a nested SandboxAgent with no sandbox config as
  // its output, and the run goes on.
  const lead = new Agent({
    name: 'lead',
    model: 'gpt-4o-mini',
    tools: [new SandboxAgent({ name: 'sandbox', model: 'gpt-4o-mini' }).asTool({ toolName: 'ask_sandbox' })],
    toolUseBehavior: 'stop_on_first_tool',
  });
  const output = `${(await runner.run(lead, 'hello')).finalOutput}`;
  if (!/^An error occurred while running the tool/.test(output)) {
    return `FAIL: unexpected nested-no-config output: ${output}`;
  }

  return 'OK';
}

/**
 * A listener replaces a sandbox config's client with a raw one after the runner checked it: on the
 * run's own config when `SandboxAgent` starts (`agent-start`), or on an `asTool()` tool's
 * `runOptions.sandbox` when the lead calls the tool (`tool-start`). The output says when the raw client
 * ran or the nested `SandboxAgent` started, which upstream's generic tool error would otherwise hide.
 */
export async function sandboxClientChangedWorkflow(kind: 'agent-start' | 'tool-start'): Promise<string> {
  let rawRan = false;
  // A sandbox client that runs in the Workflow, which the plugin must never call.
  const rawSandboxClient = {
    backendId: 'raw',
    async create(): Promise<never> {
      rawRan = true;
      throw new Error('the raw sandbox client ran in the Workflow');
    },
  } as unknown as SandboxClient;
  const sandbox: SandboxRunConfig = { client: temporalSandboxClient('fake') };
  const agent = new SandboxAgent({
    name: 'sandbox-changed',
    model: 'gpt-4o-mini',
    capabilities: [new TestSandboxCapability()],
  });
  let output: string;
  if (kind === 'agent-start') {
    agent.on('agent_start', () => (sandbox.client = rawSandboxClient));
    output = `${(await new TemporalOpenAIRunner().run(agent, 'run a command', { runConfig: { sandbox } })).finalOutput}`;
  } else {
    const lead = new Agent({
      name: 'lead',
      model: 'gpt-4o-mini',
      tools: [agent.asTool({ toolName: 'ask_sandbox', runOptions: { sandbox } })],
      toolUseBehavior: 'stop_on_first_tool',
    });
    lead.on('agent_tool_start', () => (sandbox.client = rawSandboxClient));
    let started = false;
    agent.on('agent_start', () => (started = true));
    output = `${(await new TemporalOpenAIRunner().run(lead, 'help')).finalOutput}`;
    if (started) output = `the nested SandboxAgent started: ${output}`;
  }
  return rawRan ? `the raw sandbox client ran in the Workflow: ${output}` : output;
}

/**
 * A custom `Handoff` whose callback returns `vip`, which asks a nested `SandboxAgent` through
 * `agent.asTool()`. With `inGraph`, `vip` is also a handoff of the router, so the runner sees it up front.
 */
export async function customHandoffSandboxToolWorkflow(inGraph: boolean): Promise<string> {
  const specialist = new SandboxAgent({
    name: 'sandbox-specialist',
    model: 'gpt-4o-mini',
    capabilities: [new TestSandboxCapability()],
  });
  const vip = new Agent({
    name: 'vip',
    model: 'gpt-4o-mini',
    tools: [specialist.asTool({ toolName: 'ask_sandbox' })],
  });
  const standard = new Agent({ name: 'standard', model: 'gpt-4o-mini' });
  const desk = new Handoff(standard, async () => vip);
  desk.toolName = 'transfer_to_desk';
  const router = new Agent({ name: 'router', model: 'gpt-4o-mini', handoffs: inGraph ? [desk, vip] : [desk] });
  const result = await new TemporalOpenAIRunner().run(router, 'help', {
    runConfig: { sandbox: { client: temporalSandboxClient('fake') } },
  });
  return `${result.lastAgent?.name}: ${result.finalOutput}`;
}

/** A custom `Handoff` whose callback returns a `SandboxAgent`, with no `runConfig.sandbox`. */
export async function customHandoffSandboxAgentWithoutConfigWorkflow(): Promise<string> {
  const specialist = new SandboxAgent({ name: 'sandbox-specialist', model: 'gpt-4o-mini' });
  const standard = new Agent({ name: 'standard', model: 'gpt-4o-mini' });
  const desk = new Handoff(standard, async () => specialist);
  desk.toolName = 'transfer_to_desk';
  const router = new Agent({ name: 'router', model: 'gpt-4o-mini', handoffs: [desk] });
  await new TemporalOpenAIRunner().run(router, 'help');
  return 'FAIL: should have thrown';
}
