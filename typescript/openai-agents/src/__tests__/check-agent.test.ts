import test from 'ava';
import { Agent, MemorySession, RunContext, Runner, handoff } from '@openai/agents-core';
import { SandboxAgent } from '@openai/agents-core/sandbox';
import { ApplicationFailure } from '@temporalio/common';
import { getAgentToolRecord, installAgentToolInterception } from '../workflow/agent-tools';
import { checkAgent, installRunnerCheck, installRunningAgentCheck } from '../workflow/check-agent';
import { temporalSandboxClient } from '../workflow/sandbox-client';
import { textResponse } from './stubs/openai-agents';

test('checkAgent: raw function in tools throws AgentsWorkflowError', (t) => {
  const rawFn = function someTool() {
    /* not a tool() / activityAsTool() */
  };
  const agent = new Agent({ name: 'X', model: 'm', tools: [rawFn as any] });
  const err = t.throws(() => checkAgent(agent)) as ApplicationFailure;
  t.true(err instanceof ApplicationFailure);
  t.is(err.type, 'AgentsWorkflowError');
  t.true(err.nonRetryable);
  t.regex(err.message, /raw function|tool\(\) or activityAsTool\(\)/);
});

test('checkAgent: non-string Model object throws AgentsWorkflowError', (t) => {
  const fakeModel = { getResponse: async () => ({}) as any, async *getStreamedResponse() {} };
  const agent = new Agent({ name: 'X', model: fakeModel as any });
  const err = t.throws(() => checkAgent(agent)) as ApplicationFailure;
  t.is(err.type, 'AgentsWorkflowError');
  t.regex(err.message, /string/i);
});

test('checkAgent: no model and no override throws AgentsWorkflowError', (t) => {
  const agent = new Agent({ name: 'NoModel' });
  (agent as any).model = undefined;
  const err = t.throws(() => checkAgent(agent)) as ApplicationFailure;
  t.is(err.type, 'AgentsWorkflowError');
  t.regex(err.message, /no model declared/);
  t.notThrows(() => checkAgent(agent, 'run-model'));
});

test('checkAgent: cyclic handoffs terminate (seen-set prevents stack overflow)', (t) => {
  const agentA = new Agent({ name: 'A', model: 'm' });
  const agentB = new Agent({ name: 'B', model: 'm' });
  agentA.handoffs = [agentB];
  agentB.handoffs = [agentA];
  t.notThrows(() => checkAgent(agentA));
});

test('checkAgent: raw function tool on handoff target Agent is caught', (t) => {
  const rawFn = function () {};
  const specialist = new Agent({ name: 'Specialist', model: 'm', tools: [rawFn as any] });
  const triage = new Agent({ name: 'Triage', model: 'm', handoffs: [specialist] });
  const err = t.throws(() => checkAgent(triage)) as ApplicationFailure;
  t.is(err.type, 'AgentsWorkflowError');
  t.regex(err.message, /Specialist/);
});

test('checkAgent: raw function tool on handoff() wrapper agent is caught', (t) => {
  const rawFn = function () {};
  const specialist = new Agent({ name: 'Specialist', model: 'm', tools: [rawFn as any] });
  const triage = new Agent({ name: 'Triage', model: 'm', handoffs: [handoff(specialist)] });
  const err = t.throws(() => checkAgent(triage)) as ApplicationFailure;
  t.is(err.type, 'AgentsWorkflowError');
  t.regex(err.message, /Specialist/);
});

test('installAgentToolInterception: installing again keeps one wrapper and its records', (t) => {
  const asTool = Agent.prototype.asTool;
  const agent = new Agent({ name: 'Nested', model: 'm' });
  const tool = agent.asTool({ toolName: 'ask_nested' });
  installAgentToolInterception();
  t.is(Agent.prototype.asTool, asTool);
  t.is(getAgentToolRecord(tool)?.agent, agent);
});

test('installAgentToolInterception: records tools of Agent subclasses with the options they were made with', (t) => {
  const agent = new SandboxAgent({ name: 'Sandboxed', model: 'm' });
  const runConfig = { model: 'tool-model' };
  const options = { toolName: 'ask_sandboxed', runConfig };
  const record = getAgentToolRecord(agent.asTool(options));
  (options as any).runConfig = { modelProvider: {} };
  t.is(record?.agent, agent);
  t.is(record?.runConfig, runConfig);
});

test('installRunningAgentCheck: installing again keeps one wrapper, which checks nothing outside a Workflow', async (t) => {
  const getSystemPrompt = Agent.prototype.getSystemPrompt;
  installRunningAgentCheck();
  t.is(Agent.prototype.getSystemPrompt, getSystemPrompt);
  const fakeModel = { getResponse: async () => ({}) as any, async *getStreamedResponse() {} };
  const agent = new Agent({ name: 'X', instructions: 'Answer.', model: fakeModel as any });
  t.is(await agent.getSystemPrompt(new RunContext()), 'Answer.');
});

test("checkAgent: installs the check before each model call on another copy's Agent class, once", async (t) => {
  // Stands in for the Agent class of another copy of @openai/agents-core, and a subclass of it.
  class OtherCopyAgent {
    name = 'Other';
    model = 'm';
    async getSystemPrompt() {
      return 'Answer.';
    }
  }
  class OtherCopySubclass extends OtherCopyAgent {
    override async getSystemPrompt() {
      return `${await super.getSystemPrompt()}!`;
    }
  }
  const getSystemPrompt = OtherCopyAgent.prototype.getSystemPrompt;
  const subclassGetSystemPrompt = OtherCopySubclass.prototype.getSystemPrompt;
  checkAgent(new OtherCopySubclass() as unknown as Agent);
  const wrapper = OtherCopyAgent.prototype.getSystemPrompt;
  t.not(wrapper, getSystemPrompt);
  t.is(OtherCopySubclass.prototype.getSystemPrompt, subclassGetSystemPrompt);
  checkAgent(new OtherCopyAgent() as unknown as Agent);
  t.is(OtherCopyAgent.prototype.getSystemPrompt, wrapper);
  t.is(await new OtherCopySubclass().getSystemPrompt(), 'Answer.!');

  const err = t.throws(() => checkAgent({ name: 'Plain', model: 'm' } as unknown as Agent)) as ApplicationFailure;
  t.is(err.type, 'AgentsWorkflowError');
  t.regex(err.message, /Agent 'Plain' is not an @openai\/agents-core Agent/);
});

test('installRunnerCheck: installing again keeps one wrapper, which checks nothing outside a Workflow', async (t) => {
  const run = Runner.prototype.run;
  installRunnerCheck();
  t.is(Runner.prototype.run, run);
  const fakeModel = {
    getResponse: async () => textResponse('done'),
    async *getStreamedResponse() {},
  };
  // Outside a Workflow, a Runner with another provider and a Model object runs as upstream.
  const runner = new Runner({ modelProvider: { getModel: () => fakeModel as any }, model: fakeModel as any });
  t.is((await runner.run(new Agent({ name: 'X' }), 'hi')).finalOutput, 'done');
});

test('checkAgent: asTool() agents and their own tools are checked, and nothing is changed', (t) => {
  const rawFn = function () {};
  const inner = new Agent({ name: 'Inner', model: 'inner-model', tools: [rawFn as any] });
  const nested = new Agent({ name: 'Nested', model: 'own-model', tools: [inner.asTool({ toolName: 'ask_inner' })] });
  const tool = nested.asTool({ toolName: 'ask_nested' });
  const parent = new Agent({ name: 'Parent', model: 'm', tools: [tool] });
  const err = t.throws(() => checkAgent(parent)) as ApplicationFailure;
  t.is(err.type, 'AgentsWorkflowError');
  t.regex(err.message, /Agent 'Inner': Provided tool is a raw function/);

  inner.tools = [];
  t.notThrows(() => checkAgent(parent));
  t.is(parent.tools[0], tool);
  t.is(nested.model, 'own-model');
});

test('checkAgent: asTool() options that would run in the Workflow are rejected', (t) => {
  const nested = new Agent({ name: 'Nested', model: 'm' });
  const rejected = (options: Record<string, unknown>) => {
    const parent = new Agent({
      name: 'Parent',
      model: 'm',
      tools: [nested.asTool({ toolName: 'ask_nested', ...options })],
    });
    return t.throws(() => checkAgent(parent)) as ApplicationFailure;
  };
  t.is(rejected({ runConfig: { model: { getResponse: async () => ({}) } } }).type, 'AgentsWorkflowError');
  t.is(rejected({ runConfig: { modelProvider: { getModel: () => ({}) } } }).type, 'AgentsWorkflowError');
  // Upstream spreads runConfig over the inherited config, so this clears the runner's provider.
  t.is(rejected({ runConfig: { modelProvider: undefined } }).type, 'AgentsWorkflowError');
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
  t.notThrows(() => checkAgent(parent));
});
