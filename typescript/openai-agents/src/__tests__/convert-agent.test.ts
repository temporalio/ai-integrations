import test from 'ava';
import { Agent, Handoff, MemorySession, handoff, type FunctionTool } from '@openai/agents-core';
import { SandboxAgent } from '@openai/agents-core/sandbox';
import { ApplicationFailure } from '@temporalio/common';
import { ActivityBackedModel } from '../workflow/activity-backed-model';
import { getAgentToolRecord, installAgentToolInterception } from '../workflow/agent-tools';
import { convertAgent } from '../workflow/convert-agent';
import { temporalSandboxClient } from '../workflow/sandbox-client';

const DEFAULT_MODEL_PARAMS = { startToCloseTimeout: '60s' as const };

test('convertAgent: raw function in tools throws AgentsWorkflowError', (t) => {
  const rawFn = function someTool() {
    /* not a tool() / activityAsTool() */
  };
  const agent = new Agent({ name: 'X', model: 'm', tools: [rawFn as any] });
  const err = t.throws(() => convertAgent(agent, DEFAULT_MODEL_PARAMS)) as ApplicationFailure;
  t.true(err instanceof ApplicationFailure);
  t.is(err.type, 'AgentsWorkflowError');
  t.true(err.nonRetryable);
  t.regex(err.message, /raw function|tool\(\) or activityAsTool\(\)/);
});

test('convertAgent: non-string Model object throws AgentsWorkflowError', (t) => {
  const fakeModel = { getResponse: async () => ({}) as any, async *getStreamedResponse() {} };
  const agent = new Agent({ name: 'X', model: fakeModel as any });
  const err = t.throws(() => convertAgent(agent, DEFAULT_MODEL_PARAMS)) as ApplicationFailure;
  t.is(err.type, 'AgentsWorkflowError');
  t.regex(err.message, /string/i);
});

test('convertAgent: no model and no override throws AgentsWorkflowError', (t) => {
  const agent = new Agent({ name: 'NoModel' });
  (agent as any).model = undefined;
  const err = t.throws(() => convertAgent(agent, DEFAULT_MODEL_PARAMS)) as ApplicationFailure;
  t.is(err.type, 'AgentsWorkflowError');
  t.regex(err.message, /no model declared/);
});

test('convertAgent: cyclic handoffs terminate (seen-map prevents stack overflow)', (t) => {
  const agentA = new Agent({ name: 'A', model: 'm' });
  const agentB = new Agent({ name: 'B', model: 'm' });
  agentA.handoffs = [agentB];
  agentB.handoffs = [agentA];

  const converted = convertAgent(agentA, DEFAULT_MODEL_PARAMS);
  t.is(converted.name, 'A');
  const convertedB = converted.handoffs[0] as Agent<any, any>;
  t.is(convertedB.name, 'B');
  t.is(convertedB.handoffs[0], converted);
});

test('convertAgent: modelNameOverride replaces agent.model in the ActivityBackedModel', (t) => {
  const agent = new Agent({ name: 'X', model: 'original-model' });
  const converted = convertAgent(agent, DEFAULT_MODEL_PARAMS, undefined, 'override-model');
  const model = (converted as any).model as ActivityBackedModel;
  t.true(model instanceof ActivityBackedModel);
  t.is((model as any).modelName, 'override-model');
});

test('convertAgent: raw function tool on handoff target Agent is caught', (t) => {
  const rawFn = function () {};
  const specialist = new Agent({ name: 'Specialist', model: 'm', tools: [rawFn as any] });
  const triage = new Agent({ name: 'Triage', model: 'm', handoffs: [specialist] });
  const err = t.throws(() => convertAgent(triage, DEFAULT_MODEL_PARAMS)) as ApplicationFailure;
  t.is(err.type, 'AgentsWorkflowError');
  t.regex(err.message, /Specialist/);
});

test('convertAgent: raw function tool on handoff() wrapper agent is caught', (t) => {
  const rawFn = function () {};
  const specialist = new Agent({ name: 'Specialist', model: 'm', tools: [rawFn as any] });
  const triage = new Agent({ name: 'Triage', model: 'm', handoffs: [handoff(specialist)] });
  const err = t.throws(() => convertAgent(triage, DEFAULT_MODEL_PARAMS)) as ApplicationFailure;
  t.is(err.type, 'AgentsWorkflowError');
  t.regex(err.message, /Specialist/);
});

test('convertAgent: does not mutate the original Handoff object', (t) => {
  const specialist = new Agent({ name: 'Specialist', model: 'm' });
  const h = handoff(specialist);
  const originalAgent = h.agent;
  const originalOnInvoke = h.onInvokeHandoff;
  const triage = new Agent({ name: 'Triage', model: 'm', handoffs: [h] });

  convertAgent(triage, DEFAULT_MODEL_PARAMS);

  t.is(h.agent, originalAgent, 'h.agent must not be mutated by convertAgent');
  t.is(h.onInvokeHandoff, originalOnInvoke, 'h.onInvokeHandoff must not be mutated');
});

test('convertAgent: cloned Handoff is a distinct instance with the Handoff prototype preserved', (t) => {
  const specialist = new Agent({ name: 'S', model: 'm' });
  const h = handoff(specialist, {
    toolNameOverride: 'transfer_via_handoff',
    toolDescriptionOverride: 'Hand off to specialist',
  });
  const triage = new Agent({ name: 'Triage', model: 'm', handoffs: [h] });

  const converted = convertAgent(triage, DEFAULT_MODEL_PARAMS);
  const convertedH = converted.handoffs[0] as Handoff<any, any>;

  t.true(convertedH instanceof Handoff);
  t.is(Object.getPrototypeOf(convertedH), Object.getPrototypeOf(h));
  t.not(convertedH, h, 'must be a different Handoff instance from the original');
  t.is(convertedH.toolName, h.toolName, 'toolNameOverride must survive the clone');
  t.is(convertedH.toolDescription, h.toolDescription, 'toolDescriptionOverride must survive the clone');
});

test('convertAgent: wrappedOnInvoke calls original onInvokeHandoff and returns converted agent', async (t) => {
  let fired = false;
  let receivedCtx: any;
  let receivedArgs: any;
  const specialist = new Agent({ name: 'S', model: 'm' });
  const h = handoff(specialist);

  const originalReturn: any = specialist;
  h.onInvokeHandoff = async (ctx, args) => {
    fired = true;
    receivedCtx = ctx;
    receivedArgs = args;
    return originalReturn;
  };

  const triage = new Agent({ name: 'Triage', model: 'm', handoffs: [h] });
  const converted = convertAgent(triage, DEFAULT_MODEL_PARAMS);
  const convertedH = converted.handoffs[0] as Handoff<any, any>;

  t.not(convertedH.onInvokeHandoff, h.onInvokeHandoff);

  const mockCtx = { context: undefined } as any;
  const returnedAgent = await convertedH.onInvokeHandoff(mockCtx, 'payload');

  t.true(fired, 'original onInvokeHandoff must fire from inside the wrapper');
  t.is(receivedCtx, mockCtx);
  t.is(receivedArgs, 'payload');
  t.is(returnedAgent.name, 'S');
  t.true((returnedAgent as any).model instanceof ActivityBackedModel);
  t.not(returnedAgent, specialist);
});

test('installAgentToolInterception: installing again keeps one wrapper and its records', (t) => {
  const asTool = Agent.prototype.asTool;
  const agent = new Agent({ name: 'Nested', model: 'm' });
  const tool = agent.asTool({ toolName: 'ask_nested' });
  installAgentToolInterception();
  t.is(Agent.prototype.asTool, asTool);
  t.is(getAgentToolRecord(tool)?.agent, agent);
});

test('installAgentToolInterception: records tools of Agent subclasses with their options', (t) => {
  const agent = new SandboxAgent({ name: 'Sandboxed', model: 'm' });
  const options = { toolName: 'ask_sandboxed', toolDescription: 'Ask.' };
  const record = getAgentToolRecord(agent.asTool(options));
  t.is(record?.agent, agent);
  t.is(record?.options, options);
});

test('convertAgent: an agent that is a handoff target and an asTool() source converts once', (t) => {
  const nested = new Agent({ name: 'Nested', model: 'own-model' });
  const tool = nested.asTool({ toolName: 'ask_nested', toolDescription: 'Ask nested.' });
  const parent = new Agent({ name: 'Parent', model: 'm', handoffs: [nested], tools: [tool] });

  const converted = convertAgent(parent, DEFAULT_MODEL_PARAMS, undefined, 'run-model');
  const convertedNested = converted.handoffs[0] as Agent<any, any>;
  const rebuilt = converted.tools[0] as FunctionTool;
  t.is((convertedNested as any).model.modelName, 'run-model');
  t.not(rebuilt, tool, 'the tool is rebuilt on the converted agent');
  t.is(rebuilt.name, 'ask_nested');
  t.is(rebuilt.description, 'Ask nested.');
  t.deepEqual(rebuilt.parameters, tool.parameters);
  t.is(parent.tools[0], tool, 'the declared agent keeps its tool');
});

test('convertAgent: asTool() agents keep their string model and their own tools are checked', (t) => {
  const rawFn = function () {};
  const inner = new Agent({ name: 'Inner', model: 'inner-model', tools: [rawFn as any] });
  const nested = new Agent({ name: 'Nested', model: 'own-model', tools: [inner.asTool({ toolName: 'ask_inner' })] });
  const parent = new Agent({ name: 'Parent', model: 'm', tools: [nested.asTool({ toolName: 'ask_nested' })] });
  const err = t.throws(() => convertAgent(parent, DEFAULT_MODEL_PARAMS)) as ApplicationFailure;
  t.is(err.type, 'AgentsWorkflowError');
  t.regex(err.message, /Agent 'Inner': Provided tool is a raw function/);

  inner.tools = [];
  const converted = convertAgent(parent, DEFAULT_MODEL_PARAMS);
  t.true(converted.tools[0] !== parent.tools[0]);
  t.is(nested.model, 'own-model', 'the declared agent is not changed');
});

test('convertAgent: asTool() options that would run in the Workflow are rejected', (t) => {
  const nested = new Agent({ name: 'Nested', model: 'm' });
  const rejected = (options: Record<string, unknown>) => {
    const parent = new Agent({
      name: 'Parent',
      model: 'm',
      tools: [nested.asTool({ toolName: 'ask_nested', ...options })],
    });
    return t.throws(() => convertAgent(parent, DEFAULT_MODEL_PARAMS)) as ApplicationFailure;
  };
  t.is(rejected({ runConfig: { model: { getResponse: async () => ({}) } } }).type, 'AgentsWorkflowError');
  t.is(rejected({ runConfig: { sandbox: {} } }).type, 'SandboxConfigurationError');
  t.is(rejected({ runOptions: { session: new MemorySession() } }).type, 'UnsafeSessionError');

  const sandbox = { client: temporalSandboxClient('sandbox') };
  const parent = new Agent({
    name: 'Parent',
    model: 'm',
    tools: [
      nested.asTool({ toolName: 'ask_nested', runConfig: { model: 'tool-model', sandbox }, runOptions: { sandbox } }),
    ],
  });
  t.notThrows(() => convertAgent(parent, DEFAULT_MODEL_PARAMS));
});
