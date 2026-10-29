// Framework-only example: a provider cannot intercept a selected Model object.
const assert = require('node:assert/strict');
const { Agent, Runner, Usage, setTracingDisabled } = require('@openai/agents-core');
setTracingDisabled(true);
let providerCalls = 0;
let modelCalls = 0;
const model = {
  async getResponse() {
    modelCalls++;
    return {
      output: [{ type: 'message', id: 'message', role: 'assistant', status: 'completed',
        content: [{ type: 'output_text', text: 'answer' }] }],
      usage: new Usage({ requests: 1, inputTokens: 1, outputTokens: 1, totalTokens: 2 }),
    };
  },
  async *getStreamedResponse() { throw new Error('unused'); },
};
const runner = new Runner({ modelProvider: { getModel() { providerCalls++; return model; } } });
(async () => {
  const result = await runner.run(new Agent({ name: 'A', model }), 'hello');
  assert.equal(result.finalOutput, 'answer');
  assert.equal(providerCalls, 0);
  assert.equal(modelCalls, 1);
  console.log(JSON.stringify({ finalOutput: result.finalOutput, providerCalls, modelCalls }));
})().catch(error => { console.error(error); process.exitCode = 1; });
