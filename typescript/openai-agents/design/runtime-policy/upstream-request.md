# Draft request for execution hooks in agents-core

Target: `openai/openai-agents-js`. Not posted.

## Request

Provide optional runtime hooks at effective run-configuration consumption and model resolution. Temporal's integration needs model and sandbox operations to run as Activities, while preserving the original agents and native `agent.asTool()` objects.

`ModelProvider` covers only string model selection. `Runner` returns a selected `Model` object directly. A graph check before `run()` cannot cover an agent returned later by a custom handoff, or a model changed by an agent lifecycle listener. Native tools also parse schemas and call input builders before merging nested run configuration. A hook before that work sees stale values.

The one-file framework example below runs against locked agents-core 0.14.3: a selected model object executes once while the supplied provider is never called. That is valid upstream behavior; the request is an extension point for runtimes that need a stronger execution contract.

## Executed framework example

Run with Node 24.22.0 and `@openai/agents-core` 0.14.3:

```javascript
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
```

Actual output:

```json
{"finalOutput":"answer","providerCalls":0,"modelCalls":1}
```

## Proposed contract

- `prepareRun(runner, agent, options)` sees the final run configuration before session or sandbox operations. It can reject unsupported setups or return private options.
- `resolveModel(runner, agent, selectedModel)` receives the actual executing agent and upstream's selected model immediately before provider/model resolution. It returns the model to use, or rejects it.
- The hooks cover ordinary and streamed execution, sandbox runtime model resolution, callback-selected agents and native nested runs. An optional runtime registration lives outside overridable tool/run config.
- Separate supported SDK package copies consult the same versioned runtime registration. Runtime installation, replacement and disposal need a documented environment scope. Public exports and supported package versions let a bundler reject older copies before execution.
- Without a registered runtime, current selection, provider and tool behavior stay the same. The hooks do not require exposing or rebinding the private native-tool source registry.

A local Worker proof replaced Temporal's `asTool`, `getSystemPrompt` and `Runner.run` interception with these two calls and a Temporal policy. It rejected callback- and listener-selected model objects across SDK copies without inline execution during replay. Native tool options, assigned guardrails, approvals and streaming controls were preserved. The implementation is a prototype, not a proposed final API shape.

## Maintainer questions

Would a runtime policy be an acceptable supported extension point? If another existing hook can guarantee these consumption points, it would be preferable. The two methods and versioned registration above are intended to make the required behavior reviewable, rather than prescribe a final global-registration API.
