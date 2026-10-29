const core = require('@openai/agents-core');
const copy = require('@openai/agents-core-copy');
core.setTracingDisabled(true);
copy.setTracingDisabled(true);
const { proxySinks, condition, defineQuery, defineSignal, setHandler } = require('@temporalio/workflow');
const { TemporalOpenAIRunner } = require('./lib/workflow');
const { textResponse } = require('./lib/__tests__/stubs/openai-agents-fakes');
const z = require('boundary-zod');

exports.boundaryProof = async function boundaryProof(kind) {
  const { report } = proxySinks();
  const inlineModel = {
    async getResponse() { report.inline(kind); return textResponse('INLINE'); },
    async *getStreamedResponse() { report.inline(kind); throw new Error('INLINE_STREAM'); },
  };
  const foreign = kind.startsWith('copy-');
  const sdk = foreign ? copy : core;
  let guardCalls = 0;
  let lead;
  if (kind === 'copy-string') {
    lead = new copy.Agent({ name: 'Lead', model: 'lead-model' });
  } else if (kind.endsWith('listener')) {
    lead = new sdk.Agent({ name: 'Lead', model: 'lead-model' });
    lead.on('agent_start', () => { lead.model = inlineModel; });
  } else if (kind.endsWith('handoff')) {
    const declared = new core.Agent({ name: 'Declared', model: 'declared-model' });
    const picked = new sdk.Agent({ name: 'Picked', model: inlineModel });
    const handoff = new core.Handoff(declared, async () => picked);
    handoff.toolName = 'transfer_to_desk';
    lead = new core.Agent({ name: 'Lead', model: 'lead-model', instructions: 'Route.', handoffs: [handoff] });
  } else {
    const nested = new sdk.Agent({ name: 'Nested', model: 'nested-model', instructions: 'Answer as nested.' });
    const runConfig = {};
    const options = { toolName: 'ask_nested', runConfig };
    if (kind === 'schema-provider') {
      options.parameters = z.object({ input: z.string() }).refine(() => {
        runConfig.modelProvider = { getModel: () => inlineModel };
        return true;
      });
    }
    if (kind === 'undefined-provider') {
      core.setDefaultModelProvider({ getModel: () => inlineModel });
      options.runConfig.modelProvider = undefined;
    }
    if (kind === 'input-model') options.inputBuilder = ({ params }) => {
      runConfig.model = inlineModel;
      nested.model = undefined;
      return params.input;
    };
    if (kind === 'tool-provider') options.inputBuilder = ({ params }) => {
      runConfig.modelProvider = { getModel: () => inlineModel };
      return params.input;
    };
    if (kind === 'copy-nested-listener') nested.on('agent_start', () => { nested.model = inlineModel; });
    const tool = nested.asTool(options);
    if (kind === 'input-guard') tool.inputGuardrails = [{ name: 'block', run: async () => {
      guardCalls++; return { behavior: { type: 'rejectContent', message: 'BLOCKED' } };
    } }];
    if (kind === 'output-guard') tool.outputGuardrails = [{ name: 'block', run: async () => {
      guardCalls++; return { behavior: { type: 'rejectContent', message: 'BLOCKED' } };
    } }];
    if (kind === 'approval') tool.needsApproval = async () => true;
    lead = new core.Agent({ name: 'Lead', model: 'lead-model', instructions: 'Use the tool.',
      tools: [tool], toolUseBehavior: 'stop_on_first_tool' });
  }
  const patched = ['@temporalio/openai-agents/agentToolInterception',
    '@temporalio/openai-agents/runningAgentCheck'].some(key => core.Agent.prototype[Symbol.for(key)]);
  try {
    const result = await new TemporalOpenAIRunner().run(lead, 'go');
    if (kind === 'native-held') {
      let released = false;
      setHandler(defineQuery('held'), () => true);
      setHandler(defineSignal('release'), () => { released = true; });
      await condition(() => released);
    }
    return { output: String(result.finalOutput), separateCopies: core.Agent !== copy.Agent, patched: !!patched,
      guardCalls, interruptions: result.interruptions.length };
  } catch (error) {
    return { output: `${error.type || error.name}: ${error.message}`, separateCopies: core.Agent !== copy.Agent,
      patched: !!patched };
  }
};
