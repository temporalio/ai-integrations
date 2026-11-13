# OpenAI Agents SDK Integration for Temporal

Release stage: Generally Available.

Run [OpenAI Agents SDK](https://github.com/openai/openai-agents-js) agents as Temporal Workflows. Agent orchestration runs in the Workflow; model calls run as Activities, so they retry durably and are not repeated during Workflow replay.

## Install

```bash
# Or `pnpm add`/`yarn add`
npm install @temporalio/openai-agents @openai/agents-core @openai/agents-openai openai
```

`@openai/agents-core`, `@openai/agents-openai`, and `openai` are peer dependencies.

## Quick start

Create a Workflow that uses `TemporalOpenAIRunner` instead of the upstream `Runner`:

```ts
import { Agent } from '@openai/agents-core';
import { TemporalOpenAIRunner } from '@temporalio/openai-agents/workflow';

export async function haikuAgentWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'Assistant',
    instructions: 'You only respond in haikus.',
    model: 'gpt-4o-mini',
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}
```

Register `OpenAIAgentsPlugin` on the Worker so model Activities, Workflow interceptors, sinks, and polyfills are installed:

```ts
import { OpenAIProvider } from '@openai/agents-openai';
import { OpenAIAgentsPlugin } from '@temporalio/openai-agents';
import { NativeConnection, Worker } from '@temporalio/worker';

async function main() {
  const connection = await NativeConnection.connect();
  const plugin = new OpenAIAgentsPlugin({
    modelProvider: new OpenAIProvider(),
    modelParams: { startToCloseTimeout: '30s' },
  });

  const worker = await Worker.create({
    connection,
    taskQueue: 'my-task-queue',
    workflowsPath: require.resolve('./workflows'),
    plugins: [plugin],
  });

  await worker.run();
}

main();
```

Register the same plugin type on the Client so model parameters and tracing options propagate to new Workflows:

```ts
import { OpenAIProvider } from '@openai/agents-openai';
import { Client, Connection } from '@temporalio/client';
import { OpenAIAgentsPlugin } from '@temporalio/openai-agents';

async function main() {
  const connection = await Connection.connect();
  const plugin = new OpenAIAgentsPlugin({
    modelProvider: new OpenAIProvider(),
  });

  const client = new Client({
    connection,
    plugins: [plugin],
  });

  const result = await client.workflow.execute('haikuAgentWorkflow', {
    args: ['Tell me about recursion in programming.'],
    taskQueue: 'my-task-queue',
    workflowId: 'haiku-workflow',
  });

  console.log(result);
}

main();
```

## Workflow runner

`TemporalOpenAIRunner` mirrors the OpenAI Agents SDK `Runner` for Workflow-safe execution. Use `run(agent, input, options?)` with familiar options such as `maxTurns`, `context`, `session`, and guardrails.

Differences from the upstream runner:

- `runConfig.model` must be a model name string. The Worker's `modelProvider` resolves it inside the model Activity. As upstream, it applies to agents that set no model of their own. Upstream's docs say it overrides every agent's model, but its code and tests let an agent's own model win, and the plugin follows the code. The Python plugin follows the Python SDK, where the run's model wins.
- `signal` is not supported. Use Temporal cancellation APIs, such as `CancellationScope`, to cancel Workflow work.
- If the cancellation scope `run()` started in is cancelled, `run()` throws the `CancelledFailure`, and a streamed run fails with it, even if the run ended on a tool's output or failed another way. As upstream, an `agent.asTool()` tool returns a nested run's failure, including a cancelled model Activity, as its output, and with `toolUseBehavior: 'stop_on_first_tool'` the run ends on that output.

### Upgrading open runs

The runner now runs your agents as written. Runs started on an earlier version pick up these changes as they continue, most with no replay error:

- `runConfig.model` applies only to agents that set no model of their own. Agents that set one switch to it from their next model call.
- A GPT-5 agent that sets no `modelSettings` gets upstream's default settings for its model, such as `reasoning.effort` and `text.verbosity`, and explicit GPT-5 `providerData.reasoning` reaches the model. Set `modelSettings: {}` on the agent to send none. An agent that names no model, in a run without `runConfig.model`, gets the defaults for the Worker's default model (`OPENAI_DEFAULT_MODEL`), as upstream does in one process. If the Worker's model provider picks a model that isn't GPT-5 for such an agent, for example on an OpenAI-compatible endpoint, set `OPENAI_DEFAULT_MODEL` on the Worker to that model, or give the agent `modelSettings: {}`.
- An agent with a `prompt` template and no model or tools gets the template's model and tools. Before, the Worker's default model and an empty tool list replaced them.
- A `SandboxAgent` that names no model, in a run without `runConfig.model`, also gets the defaults for the Worker's default model, but the run's `reasoning` and `text` settings win over them, where upstream lets the defaults win. Its compaction threshold follows `gpt-5.4-mini`, the Workflow's default model, unless the agent or `runConfig.model` names a model.
- `Agent` subclass overrides, such as `getSystemPrompt`, run, and a `SandboxAgent`'s compaction threshold follows the model it or `runConfig.model` names.
- Agent hooks (`agent.on(...)`) fire, and a custom `Handoff` hands off to the agent its callback returns.
- Nested `agent.asTool()` runs share the parent's run context, so their context writes and usage add up in the parent's result. Their model calls use the runner's Activity options, including `useLocalActivity` and `cancellationType`.
- As upstream, one `SandboxAgent` object serves one run at a time. A second concurrent run, or a second parallel call of its `asTool()` tool, fails with `SandboxAgent '...' cannot be reused concurrently across runs`. Create one `SandboxAgent` per concurrent run, or set `modelSettings: { parallelToolCalls: false }` on an agent that calls a sandbox agent's tool.
- A run cancelled during a nested model call ends cancelled, where it failed before. So does a run with `stop_on_first_tool` whose tool caught its own Activity's cancellation, which completed before.
- An upstream `Runner` can't run agents in a Workflow, even with your own Workflow-safe model provider; use `TemporalOpenAIRunner`.
- A `RunState` passed to `run()` is resumed as it is, as upstream: its tools see your context objects, including a context given to `RunState.fromStringWithContext`, not a copy rebuilt from JSON; the state's own agents run, not the ones you pass to `run()`; and the resume doesn't list the agent's tools before the approved tools run. A state with sandbox sessions is still restored from its string first, with the same context objects. A failed resume run again from the same state continues where it failed, as upstream, and doesn't run the approved tools again; before, it ran them again.

Let runs that use `agentAsTool` or `agent.asTool()`, `runConfig.model`, `useLocalActivity` or `cancellationType`, agent hooks, custom `Handoff` callbacks, a shared `SandboxAgent` or a `RunState` resume finish on the old Worker, or pin them to it with Worker Versioning. Upgrade the Workers that run model Activities before the Workers that run Workflows: an older Activity Worker drops the marker that lets a model leave out an implicit reasoning effort.

A passing replay test of old histories doesn't show these runs are safe. Every model call is the same Activity type, and replay doesn't compare Activity inputs, so a run that now makes a different model call can replay without an error and continue with the wrong answer. Model calls are also scheduled a little later in a turn, so work started at the same time, such as another tool's Activity or a parallel branch, can swap places with them. Replay then fails with `[TMPRL1100]`, or, when both are model calls, each gets the other's recorded answer. Histories of runs cancelled during a nested model call, recorded before the upgrade, fail replay, because those runs now end cancelled. So do histories of runs that resumed a `RunState` and listed tools again, through an MCP server with `cacheToolsList: false` or a tool filter that awaits an Activity or timer; that used the context rebuilt from JSON; or that ran a failed resume again from the same state.

Before resuming a `RunState` saved on an earlier version, rename any `asTool()` agent that has the same name as the starting agent or one of its handoff targets, and keep the names of those agents. The framework also finds `asTool()` agents by name when it restores a `RunState`, and `agentAsTool` tools weren't included, so a saved run can resume on the wrong agent.

`agentAsTool(agent, { toolName, toolDescription, customOutputExtractor })` is deprecated and will be removed in a later release. It returns `agent.asTool({ toolName, toolDescription, customOutputExtractor })`; call that instead. The alias also accepts an agent from another copy of `@openai/agents-core`, unless its class overrides `asTool()`. The runner rejects the `asTool()` tools of such agents, so moving them to `agent.asTool()` needs a single copy of the package. As upstream, a nested approval now pauses the parent run, where `agentAsTool` failed the run with `NestedAgentInterruption`: after `run()`, check `result.interruptions` when a nested agent has tools that need approval.

### Run state and approvals

`TemporalOpenAIRunner.run` accepts a `RunState` as the second argument, matching the upstream runner. This supports human approval flows that pause, wait for a Signal or Update, then continue as new.

```ts
import { Agent, RunState, tool } from '@openai/agents-core';
import { TemporalOpenAIRunner } from '@temporalio/openai-agents/workflow';
import { condition, continueAsNew, defineSignal, setHandler } from '@temporalio/workflow';

const approveSignal = defineSignal('approve');

interface ApprovalInput {
  resumeFromRunState?: string;
}

export async function approvalWorkflow(input: ApprovalInput = {}): Promise<string> {
  const action = tool({
    name: 'dangerousAction',
    description: 'Perform an action that needs approval',
    parameters: {
      type: 'object' as const,
      properties: { reason: { type: 'string' } },
      required: ['reason'] as const,
      additionalProperties: false as const,
    },
    needsApproval: true,
    execute: async (args) => `did: ${(args as { reason: string }).reason}`,
  });

  const agent = new Agent({
    name: 'Approver',
    instructions: 'Use dangerousAction when asked.',
    model: 'gpt-4o-mini',
    tools: [action],
  });
  const runner = new TemporalOpenAIRunner();

  if (input.resumeFromRunState !== undefined) {
    const state = await RunState.fromString(agent, input.resumeFromRunState);
    for (const interruption of state.getInterruptions()) {
      state.approve(interruption);
    }
    const resumed = await runner.run(agent, state);
    return resumed.finalOutput ?? '';
  }

  let approved = false;
  setHandler(approveSignal, () => {
    approved = true;
  });

  const result = await runner.run(agent, 'please act');
  if (result.interruptions.length === 0) return result.finalOutput ?? '';

  await condition(() => approved);
  await continueAsNew<typeof approvalWorkflow>({ resumeFromRunState: result.state.toString() });
  throw new Error('unreachable');
}
```

The agent passed to `RunState.fromString` must define the same tool names, handoff graph, and MCP servers as the run that produced the serialized state.

### Streaming

> **Experimental** — streaming support is subject to change prior to General Availability.

Pass `{ stream: true }` to `run()` to stream incremental events as the model responds. The call returns the upstream `StreamedRunResult`, an async-iterable of agent-SDK stream events. Inside a Workflow, each model call executes as a streaming Activity (`invokeModelStreamActivity`) that consumes `Model.getStreamedResponse` and returns the collected list of native model events. The Workflow yields those events deterministically from the Activity's return value — it never polls the live stream, so replay stays deterministic.

External consumers (UIs, tracing pipelines) observe events as they arrive by hosting a [`WorkflowStream`](../workflow-streams/README.md) in the Workflow and subscribing with `WorkflowStreamClient`. The streaming Activity publishes each event to the topic configured on `modelParams.streamingTopic`. The topic is required for streaming runs; calling `run(agent, input, { stream: true })` without a configured topic throws before any Activity is scheduled.

Streaming requires a `streamingTopic`. Set it on the `modelParams` of the plugin registered on the **Client** — the Client interceptor injects the config header into every Workflow-starting call:

```ts
const plugin = new OpenAIAgentsPlugin({
  modelProvider: new OpenAIProvider(),
  modelParams: { streamingTopic: 'events', streamingBatchInterval: '100ms' },
});

const client = new Client({ connection, plugins: [plugin] });
```

Workflows started without that Client — by a Schedule or the Temporal UI/CLI — receive no config header. Supply Workflow-authored defaults through the runner's `defaultModelParams`; unlike the header, these can carry a function-form `summary`. Host a `WorkflowStream` in the Workflow to receive the publishes:

```ts
import { Agent } from '@openai/agents-core';
import { TemporalOpenAIRunner } from '@temporalio/openai-agents/workflow';
import { WorkflowStream } from '@temporalio/workflow-streams/workflow';

export async function streamingWorkflow(prompt: string): Promise<string> {
  new WorkflowStream();
  const agent = new Agent({ name: 'Assistant', instructions: '...' });
  const runner = new TemporalOpenAIRunner({ defaultModelParams: { streamingTopic: 'events' } });
  const result = await runner.run(agent, prompt, { stream: true });
  for await (const event of result) {
    if (event.type === 'raw_model_stream_event' && event.data.type === 'output_text_delta') {
      // event.data.delta — incremental text
    }
  }
  return result.finalOutput ?? '';
}
```

The two layers combine per field: the Client's `modelParams` overrides the runner's `defaultModelParams`.

External subscribers receive the unwrapped native model events directly:

```ts
import { WorkflowStreamClient } from '@temporalio/workflow-streams/client';

for await (const item of WorkflowStreamClient.create(client, workflowId).topic('events').subscribe()) {
  // item.data — native model stream event
}
```

The hosted `WorkflowStream` retains every published event on the Workflow heap and in history until `truncate(upToOffset)` is called. A streaming agent publishes many events per run, so a long-running or high-volume Workflow that never truncates will eventually hit Temporal's Workflow history size limits. Call `truncate(upToOffset)` periodically to keep the heap and history bounded; see [`@temporalio/workflow-streams`](../workflow-streams/README.md) for the offset model and batching.

Streaming is incompatible with `useLocalActivity`, which supports neither Activity heartbeats nor the Workflow Stream signal channel. See [`@temporalio/workflow-streams`](../workflow-streams/README.md) for the publisher/subscriber API and delivery semantics.

### Sessions

Use `WorkflowSafeMemorySession` for conversation history stored on the Workflow heap and rebuilt by replay. This replaces the upstream `MemorySession`, which is not replay safe because it depends on host process state.

```ts
import { Agent } from '@openai/agents-core';
import { TemporalOpenAIRunner, WorkflowSafeMemorySession } from '@temporalio/openai-agents/workflow';

export async function chatWorkflow(prompts: string[]): Promise<string[]> {
  const agent = new Agent({
    name: 'ChatAgent',
    instructions: 'Use the conversation history to answer.',
    model: 'gpt-4o-mini',
  });
  const runner = new TemporalOpenAIRunner();
  const session = new WorkflowSafeMemorySession();

  const replies: string[] = [];
  for (const prompt of prompts) {
    const result = await runner.run(agent, prompt, { session });
    replies.push(result.finalOutput ?? '');
  }
  return replies;
}
```

`WorkflowSafeMemorySession` history lives on the Workflow heap and is rebuilt by replay within a single run. It does **not** automatically survive `continueAsNew` — a continued run starts with an empty session. To carry history across a continue-as-new boundary, capture the items and re-seed the new run's session via the constructor's `initialItems`:

```ts
// 1. Before continuing, capture the current history:
const items = await session.getItems();
await continueAsNew(/* ...your Workflow args..., */ items);

// 2. The continued run declares a Workflow parameter to receive those items,
//    and re-seeds the session from them:
const session = new WorkflowSafeMemorySession({ initialItems: items });
```

## Worker and Client plugin setup

The Worker plugin does five things:

- Registers the model Activity used by Workflow-side model calls.
- Registers Activities for configured MCP server providers.
- Registers Activities for configured sandbox client providers.
- Adds Workflow, Activity, and Nexus interceptors for trace propagation.
- Installs Workflow bundle polyfills required by the OpenAI Agents SDK.

The model Activities are named `invokeModelActivity` and `invokeModelStreamActivity`. Provider Activity names start with the provider's name, such as `<name>-stateless-call-tool`. A different Activity with one of these names now fails with `Duplicate registration of Activity '<name>'`; earlier versions kept the last one without a warning. If it is registered before `OpenAIAgentsPlugin` (in the Worker's `activities` or by an earlier plugin), `Worker.create` fails. If a plugin applied after `OpenAIAgentsPlugin` replaces one, `worker.run()` fails before the Worker polls. `Worker.create` applies the Worker's `plugins` in order, then the `NativeConnection`'s. A plugin applied after `OpenAIAgentsPlugin` that wraps these Activities counts as replacing them; wrap Activity execution with an Activity interceptor (`interceptors.activity`) instead.

Another `OpenAIAgentsPlugin`'s Activity counts as the same only when it was built from the same objects: the same `modelProvider` and `hostedToolCredentials` for the model Activities, and the same provider for a provider's Activities. So one instance on both the `NativeConnection` and the Worker, or two instances built from one options object, still start, while two instances with equal but separate providers fail.

The Client plugin injects the OpenAI Agents config header into Workflow-starting calls. Attach one `OpenAIAgentsPlugin` instance per Client or Connection configuration. If multiple instances are chained, the last config header wins.

`modelParams` controls scheduling for the model Activity:

```ts
const plugin = new OpenAIAgentsPlugin({
  modelProvider: new OpenAIProvider(),
  modelParams: {
    startToCloseTimeout: '30s',
    retry: { maximumAttempts: 5, initialInterval: '1s' },
    useLocalActivity: false,
  },
});
```

See `ModelActivityOptions` for the public field list. `summary` can be a string in plugin config; `ModelSummaryProvider` functions are intentionally not serialized into Workflow headers.

## Tools

Inline function tools, hosted tools, Activity-backed tools, Nexus operation tools, nested agent tools, and MCP tools can all be used from a Temporal-backed agent. Any tool that does I/O must run outside the Workflow sandbox, usually through an Activity or Nexus Operation.

### Activity-backed tools

Use `activityAsTool` for HTTP calls, database access, file system work, or other I/O. The tool name must match a registered Activity.

```ts
import { Agent } from '@openai/agents-core';
import { activityAsTool } from '@temporalio/openai-agents/workflow';
import type * as activities from './activities';

const weatherTool = activityAsTool<typeof activities.getWeather>(
  {
    name: 'getWeather',
    description: 'Get the weather for a city',
    parameters: {
      type: 'object',
      properties: { location: { type: 'string' } },
      required: ['location'],
      additionalProperties: false,
    },
  },
  {
    startToCloseTimeout: '10s',
    retryPolicy: { maximumAttempts: 3 },
  }
);

const agent = new Agent({
  name: 'WeatherAgent',
  instructions: 'Use the getWeather tool when asked about weather.',
  model: 'gpt-4o-mini',
  tools: [weatherTool],
});
```

The generic type is compile-time only. At runtime the Activity is invoked by name through `proxyActivities`.

### Inline and hosted tools

For deterministic computation, use `tool()` from `@openai/agents-core` directly. Inline tools run in the Workflow sandbox and must not perform I/O, consume nondeterministic randomness, or read wall-clock time beyond Temporal's deterministic replacements.

Hosted tools from `@openai/agents-openai`, such as `webSearchTool()`, run server-side through the model provider during the model Activity.

```ts
import { Agent, tool } from '@openai/agents-core';
import { webSearchTool } from '@openai/agents-openai';

const addNumbers = tool({
  name: 'addNumbers',
  description: 'Add two numbers',
  parameters: {
    type: 'object' as const,
    properties: { a: { type: 'number' }, b: { type: 'number' } },
    required: ['a', 'b'] as const,
    additionalProperties: false as const,
  },
  execute: async (args) => String((args as { a: number; b: number }).a + (args as { a: number; b: number }).b),
});

const agent = new Agent({
  name: 'SearchAgent',
  instructions: 'You have web search and arithmetic.',
  model: 'gpt-4o-mini',
  tools: [addNumbers, webSearchTool()],
});
```

### Nexus operation tools

Use `nexusOperationAsTool` to expose a [Nexus](https://docs.temporal.io/nexus) Operation as an agent tool. The Workflow starts the Operation through a Nexus client and feeds the stringified result back to the agent.

```ts
import { Agent } from '@openai/agents-core';
import { nexusOperationAsTool } from '@temporalio/openai-agents/workflow';
import * as nexus from 'nexus-rpc';

const weatherService = nexus.service('weather', {
  getWeather: nexus.operation<{ location: string }, { tempC: number }>(),
});

const weatherTool = nexusOperationAsTool(
  weatherService.operations.getWeather,
  {
    name: 'getWeather',
    description: 'Get the weather for a city',
    parameters: {
      type: 'object',
      properties: { location: { type: 'string' } },
      required: ['location'],
      additionalProperties: false,
    },
  },
  { service: weatherService, endpoint: 'weather-endpoint' }
);

const agent = new Agent({
  name: 'WeatherAgent',
  instructions: 'Use the weather tool.',
  model: 'gpt-4o-mini',
  tools: [weatherTool],
});
```

### Nested agent tools

Use the framework's `agent.asTool()` to expose another `Agent` as a tool. The nested run happens in the Workflow, and its model calls are model Activities, like the parent's:

```ts
import { Agent } from '@openai/agents-core';

const specialist = new Agent({
  name: 'Specialist',
  instructions: 'Answer precisely.',
  model: 'gpt-4o-mini',
});

const triage = new Agent({
  name: 'Triage',
  instructions: 'Delegate specialist questions.',
  model: 'gpt-4o-mini',
  tools: [
    specialist.asTool({
      toolName: 'ask_specialist',
      toolDescription: 'Ask the specialist agent',
    }),
  ],
});
```

As in the framework, the nested run takes the run's `model` (for agents that set none) and `modelSettings`, without `toolChoice` and `parallelToolCalls`, and shares the parent's run context, so its context writes and usage add up in the parent. It doesn't take `traceIncludeSensitiveData`, guardrails or `handoffInputFilter`; in particular, its traces include tool inputs and outputs unless you set `runConfig: { traceIncludeSensitiveData: false }` in `asTool()`'s options. Its model Activities use the same options as the runner's. A nested approval pauses the parent run: approve it on the parent's `result.state` and resume, as in [Run state and approvals](#run-state-and-approvals).

`asTool({ onStream })`, or a handler added with the tool's `on(...)`, makes the nested run stream. That needs the runner's `streamingTopic` and a hosted `WorkflowStream`, as in [Streaming](#streaming); otherwise the tool's output is the `StreamingTopicNotConfigured` error. Its events go to the parent's topic, without the nested agent's name.

As upstream, the tool returns a nested run's failure to the parent model as its output, which starts with `An error occurred while running the tool. Please try again.` This includes model Activity failures that are not retried, such as an authentication error, model Activity timeouts, the plugin's checks in the nested run below, and a nested `SandboxAgent` in a run without `runConfig.sandbox`. The parent model may call the tool again until the run reaches `maxTurns`, and that error doesn't carry the cause. A `ModelSummaryProvider` gets the agent of each model call, nested ones included.

The framework doesn't expose the agent behind an `asTool()` tool, so the plugin records it: in the Workflow, before your Workflow module loads, it wraps `Agent.prototype.asTool`. The runner uses that record to check the nested agent before the run. Create these tools in Workflow code with `agent.asTool()`. Copies, such as the ones `toolNamespace()` makes, keep the record, and a nested approval behind one resumes when you pass `result.state` back to `run()`. But the framework's `RunState.fromString` finds a nested agent only through the tool `asTool()` returned, so after it, as after `continueAsNew`, a nested approval behind a copy stays pending. A run with sandbox sessions is restored that way on each resume, so there the runner fails the resume with `AgentsWorkflowError` instead. The runner rejects an `asTool()` tool it can't trace back to its agent: one made by an agent from another copy of `@openai/agents-core`, or one created outside a Worker with the plugin before `@temporalio/openai-agents/workflow` was imported.

The runner rejects nested setups that would run model calls or sandbox operations in the Workflow:

- a nested agent, or an agent it hands off to, with a `Model` object instead of a model name;
- `asTool({ runConfig: { modelProvider } })`, even `modelProvider: undefined`, which removes the runner's provider; or a `Model` object as `runConfig.model`;
- a `runConfig.sandbox` or `runOptions.sandbox` whose client isn't from `temporalSandboxClient(name)`, or whose live `session` isn't from such a client;
- a `MemorySession` as `runOptions.session`.

The runner also checks each agent just before it calls its model. This covers agents the check before the run can't see: one that a custom `Handoff`'s callback returns, or one that an `agent.on(...)` listener changes. Its model must be a name, and its `asTool()` tools must pass the checks above. In the outer run, the run fails. In a nested run, the error goes back to the parent model as the tool's output, and the model isn't called. The check wraps `Agent.prototype.getSystemPrompt`, which the framework calls once per model call, so an `Agent` subclass that overrides it must call `super.getSystemPrompt()`. An agent from another copy of `@openai/agents-core`, such as one a dependency brings, is checked the same way: the runner wraps that copy's `Agent.prototype.getSystemPrompt` before the agent's first model call. Its `asTool()` tools are rejected, because the plugin records only its own copy's tools.

The framework reads an `asTool()` tool's `runConfig` and `runOptions` only when the nested run starts, after the tool's `parameters` schema, its `inputBuilder` and any `agent_tool_start` listener ran. So the plugin also checks each run as it starts, by wrapping `Runner.prototype.run` in the Workflow: the run's model provider must be the runner's, its `runConfig.model` a name, its session not a `MemorySession`, and its sandbox config must pass the checks above. A nested run that fails this check doesn't start, and the error is the tool's output. The run then uses a frozen copy of its sandbox config, so a later change to your object, for example in an `agent_start` listener, doesn't reach it. Because of this check, an upstream `Runner` can't run agents in a Workflow; use `TemporalOpenAIRunner`.

## MCP servers

The integration supports stateless and stateful [Model Context Protocol](https://modelcontextprotocol.io/) servers.

Both providers require factories to return a server with callable `connect`, `close`, `listTools` and `callTool` methods. An invalid result fails the Activity with a non-retryable `ApplicationFailure` of type `MCPServerFactoryTypeError` before the server is connected or registered. Plain objects implementing these methods are accepted. A partial server that previously worked for one operation must now provide all four methods. Other invalid results previously failed with ordinary runtime errors that could be retried. Fix these factories before upgrading: deploying a corrected Worker will not automatically resume an Activity that has already failed.

### Stateless MCP servers

Use stateless servers when each tool call is independent. Register a provider on the Worker:

```ts
import { MCPServerStreamableHttp } from '@openai/agents-core';
import { OpenAIProvider } from '@openai/agents-openai';
import { OpenAIAgentsPlugin, StatelessMCPServerProvider } from '@temporalio/openai-agents';

const unitConversionMcp = new StatelessMCPServerProvider(
  'unitConversion',
  (factoryArgument?) =>
    new MCPServerStreamableHttp({ name: 'unitConversion', url: 'https://mcp.example.com/unit-conversion' })
);

const plugin = new OpenAIAgentsPlugin({
  modelProvider: new OpenAIProvider(),
  mcpServerProviders: [unitConversionMcp],
});
```

Reference the same provider name from Workflow code:

```ts
import { Agent } from '@openai/agents-core';
import { statelessMcpServer, TemporalOpenAIRunner } from '@temporalio/openai-agents/workflow';

export async function mcpWorkflow(query: string): Promise<string> {
  const agent = new Agent({
    name: 'UnitConverter',
    instructions: 'Use unit conversion tools to answer questions.',
    model: 'gpt-4o-mini',
    mcpServers: [statelessMcpServer('unitConversion')],
  });

  const result = await new TemporalOpenAIRunner().run(agent, query);
  return result.finalOutput ?? '';
}
```

### Stateful MCP servers

Use stateful servers when a persistent connection or session is required. The plugin starts a dedicated in-process Worker pinned to a per-run Task Queue and routes MCP operations to it.

```ts
import { MCPServerStreamableHttp } from '@openai/agents-core';
import { OpenAIProvider } from '@openai/agents-openai';
import { OpenAIAgentsPlugin, StatefulMCPServerProvider } from '@temporalio/openai-agents';
import { NativeConnection } from '@temporalio/worker';

const connection = await NativeConnection.connect();
const dbMcp = new StatefulMCPServerProvider(
  'database',
  (factoryArgument?) => new MCPServerStreamableHttp({ name: 'database', url: 'https://mcp.example.com/database' }),
  connection
);

const plugin = new OpenAIAgentsPlugin({
  modelProvider: new OpenAIProvider(),
  mcpServerProviders: [dbMcp],
});
```

In the Workflow, call `connect()` before use and `cleanup()` in a `finally` block:

```ts
import { Agent } from '@openai/agents-core';
import { statefulMcpServer, TemporalOpenAIRunner } from '@temporalio/openai-agents/workflow';

export async function statefulMcpWorkflow(prompt: string): Promise<string> {
  const server = statefulMcpServer('database');
  await server.connect();
  try {
    const agent = new Agent({
      name: 'DbAgent',
      instructions: 'You have database access.',
      model: 'gpt-4o-mini',
      mcpServers: [server],
    });
    const result = await new TemporalOpenAIRunner().run(agent, prompt);
    return result.finalOutput ?? '';
  } finally {
    await server.cleanup();
  }
}
```

Dedicated Worker startup and heartbeat failures surface as `ApplicationFailure` type (exported as `DEDICATED_WORKER_FAILURE_TYPE`).

Temporal cannot recreate a stateful server's session on another Worker. If its session Activity is retried after the server connected, for example after a lost host with `serverSessionConfig.heartbeatTimeout` or after the session Activity's start-to-close timeout, the retry fails with a non-retryable `ApplicationFailure` of that type instead of connecting a new server with an empty session, and `cleanup()` then throws an `ApplicationFailure` of that type whose cause is the session Activity's `ActivityFailure`, unless the scope that called `cleanup()` or `connect()`, or a scope around `cleanup()` but not around `connect()`, is cancelled, as when the Workflow is cancelled. After a Workflow catches a cancellation, its root scope stays cancelled, so for the rest of the run `cleanup()` throws no refusal for a server connected outside `CancellationScope.nonCancellable()`, even from a non-cancellable scope; and with the agent SDK's default tool error handling, an MCP tool call that fails with `DedicatedWorkerFailure` becomes tool output for the model, so such a refusal can go unseen. To recover, catch it and connect a new server, which starts a new, empty session (but see the note on servers with the same name below). A retry after a failed factory call, a failed first `connect()`, or a Worker shutdown before the server connected connects again. A retry after a lost host or a timed-out attempt fails even when the server never connected: the Worker sends heartbeats later, and the plugin cannot wait for one, so every heartbeat says a session may be open until the attempt ends without one. Before it connects, the session Activity reads its Workflow's description (`DescribeWorkflowExecution`) until the Temporal Server shows its first heartbeat, so the Worker's client needs permission to describe the Workflow; without it, the session Activity fails with a non-retryable `DedicatedWorkerFailure` that says so. Other describe errors are retried, and the session Activity's last failure carries them. A graceful shutdown of the session's Worker, as in a deploy, ends the session Activity rather than retrying it, so `cleanup()` doesn't throw, and the server's later operations fail with `DedicatedWorkerFailure`.

Some consequences:

- The session Activity's `serverSessionConfig.startToCloseTimeout`, one hour by default, is the longest a server can stay connected, since the attempt then times out and its retry is refused. Raise it for long runs, such as ones that wait for a person, and set a `heartbeatTimeout` to notice a lost Worker sooner.
- An Activity reset or unpause with `resetHeartbeat` clears the session's heartbeat details, so the next attempt connects a new, empty server. A Workflow reset to a point where the server is connected also gives the new run a new, empty server, with no error.
- Sessions started on an earlier version left no such heartbeat details, so the deploy of this version can still give an open session a new, empty server one last time.
- Code that caught `ActivityFailure` from `cleanup()`, which earlier versions threw when the session failed, must also catch the `ApplicationFailure` of type `DedicatedWorkerFailure` that it now throws after a refusal. A session that fails for another reason still makes `cleanup()` throw the `ActivityFailure`.
- The session Activity's heartbeats go through your payload codecs, so its `heartbeatTimeout` must comfortably exceed a heartbeat's encoding time. Encoding time also delays each server's first operation, so keep it well under the 30 seconds the session Activity waits for its first heartbeat, and under the operations' `scheduleToStartTimeout`. The Worker sends a failed Activity's result only after its last heartbeat, so an `encode` that never returns keeps the session Activity's attempt from ending and a Worker shutdown from finishing; the plugin can't bound this.
- Servers with the same name in one run use the same Task Queue. After `cleanup()`, the session Activity stops only once a heartbeat response tells it about the cancellation: about one heartbeat interval with a `heartbeatTimeout`, and without one possibly not before the run ends. Until then it serves the operations of a new server with the same name, with the earlier server, and the new server's own session Activity fails because one is already running. The other way round, an earlier server's call that is still queued when `cleanup()` runs can later run with a new server of the same name, unless its `scheduleToStartTimeout` is 0. Give each server in a run a distinct name.

Only the session Activity's Worker serves the server's operations. With a schedule-to-start timeout (the default `config.scheduleToStartTimeout` is 30 seconds), an operation fails with `DedicatedWorkerFailure` when it expires. With `scheduleToStartTimeout: 0`, an operation fails with `DedicatedWorkerFailure` as soon as the session Activity has ended, for example after a graceful shutdown of its Worker, and an operation still waiting then is cancelled. A caller's own cancellation still cancels the operation, and wins over the session's end.

A run follows its session this way only once it records the patch marker `openai-agents-stateful-mcp-session-watch`, which it does at its first operation with `scheduleToStartTimeout: 0`. Histories recorded on an earlier version have no marker, so they schedule such operations as before and replay on this version; a run that continues on this version records the marker at its next such operation. A run with the marker fails replay on an earlier version, and so can any run whose session failed with `DedicatedWorkerFailure` (a refused retry or a refused describe) and that then went on, so do not roll back while such runs are open. Drain the earlier version's Workers before starting this one: while Workers of both versions run, a run whose session gets a refusal can fail replay when it moves from one to the other.

### MCP tool filters

A static `toolFilter` on the server your factory returns, such as `createMCPToolStaticFilter({ blocked: ['delete_file'] })`, applies in the list-tools Activity, so the model is not offered a blocked tool. The filter applies only to the tool list: a run that listed tools before the Worker was upgraded keeps that list while it is cached.

A callable filter gets the agent and the run context, which exist only in the Workflow, so the Worker refuses one on the factory's server: every list-tools and call-tool Activity for that server fails with a non-retryable `ApplicationFailure` of type `MCPToolFilterUnsupported` (exported as `TOOL_FILTER_UNSUPPORTED_TYPE`). A new run fails at its first list-tools Activity. A `statelessMcpServer()` run that listed tools on an older Worker instead gets a tool error, which the model reads, for each call to that server, and the run keeps going. So remove the callable filter from the factory's server before you upgrade Workers. Pass it as the `toolFilter` option of `statelessMcpServer()` or `statefulMcpServer()` instead:

```ts
const server = statelessMcpServer('files', {
  toolFilter: async ({ agent }, tool) => tool.name !== 'delete_file' || agent.name === 'FileAdmin',
});
```

The agent applies this option in the Workflow each time it lists tools; the handle caches the server's full tool list, not the filtered one. A callable filter runs for each tool on every listing, gets the name the server was registered with as `serverName`, and must be deterministic. An Activity or timer it awaits adds commands that older plugin versions never schedule, so let Workflows that use the option finish before you roll the plugin back. A filter assigned to a stateful handle later (`server.toolFilter = ...`) gets the handle's internal name, `<name>-stateful`. A callable filter also runs when a run resumes from a `RunState`, for example after a tool approval. A state from `RunState.fromString`, as after `continueAsNew`, has the context rebuilt from JSON: plain objects, without class methods. A filter that throws while `RunState.fromString` lists tools makes that call throw its error; during a resume in `run()`, the run fails with an `AgentsWorkflowError`.

Because the handle caches the full list, the agent SDK's `invalidateServerToolsCache(name)` does not refresh a handle that has a filter. Call the handle's `invalidateToolsCache()` as well, which also works on older plugin versions. On older versions, `invalidateServerToolsCache(name)` alone refreshed a stateful handle with a `toolFilter` property assigned. Such a run schedules one list-tools Activity fewer on this version and fails replay, so let it finish before you upgrade.

Older versions also let the agent SDK cache the filtered list of a handle with a `toolFilter` property assigned, per server and agent name, so a later run with another context got the first run's tools. This version filters every listing for its own run. A run that started on an older version, whose filter depends on the run context, and that called a tool the filter now removes fails replay on this version. Let such runs finish before you upgrade.

### MCP request metadata

A `toolMetaResolver` builds each tool call's MCP `_meta` from the run context, which exists only in the Workflow. Pass it as the `toolMetaResolver` option of `statelessMcpServer()` or `statefulMcpServer()`; it runs in Workflow code and must be deterministic. The call-tool Activity passes its result to the server's `callTool`. The Worker can't run a `toolMetaResolver` on the factory's server, so it refuses one: every list-tools and call-tool Activity for that server fails with a non-retryable `ApplicationFailure` of type `MCPToolMetaResolverUnsupported` (exported as `TOOL_META_RESOLVER_UNSUPPORTED_TYPE`).

```ts
const server = statelessMcpServer('tickets', {
  toolMetaResolver: ({ runContext }) => ({ tenant: (runContext.context as { tenant: string }).tenant }),
});
```

A resolver also runs for tool calls after a run resumes from a `RunState`, for example after a tool approval. A state from `RunState.fromString`, as after `continueAsNew`, has the context rebuilt from JSON: plain objects, without class methods. If it throws there, the model gets a tool error and the server isn't called.

A new run with a `toolMetaResolver` on the factory's server fails at its first list-tools Activity. A `statelessMcpServer()` run that listed tools on an older Worker instead gets a tool error, which the model reads, for each call to that server, and the run keeps going. So remove the `toolMetaResolver` from the factory's server before you upgrade Workers.

The `_meta` values are recorded in the call-tool Activity input in Workflow History, so don't put secrets in them.

A call-tool Activity that runs on a Worker with an older plugin version drops the `_meta`, and nothing fails. Upgrade every Worker that runs the call-tool Activities, including Workers on a `taskQueue` you pass to `statelessMcpServer()`, before Workflows rely on `_meta`, or use Worker Versioning.

Plugin versions before this option ignore it and don't run its resolver. An older version can replay a History only if it produces the same Workflow commands in the same order, and a resolver passed as this option can change them even when it is synchronous and never fails: the agent awaits it, which can reorder the Activities of parallel tool calls, and its changes to the run context can change later commands. An older version fails such a History with a nondeterminism error (`TMPRL1100`). Before you roll Workers back to an older plugin version, let the Workflows that use this option finish. Workflows with no resolver, or with a resolver set on the handle itself (`server.toolMetaResolver = ...`), run the same resolver code on older versions, so this option doesn't change their commands. On a `statefulMcpServer()` handle, a resolver set this way gets the handle's internal name, `<name>-stateful`, as `serverName`; the option gets `<name>`.

Adding a resolver to a Workflow that has open runs, including moving one off the factory's server, can fail their replay for the same reasons, so guard the change with `patched()` or let those runs finish first.

## Sandbox agents

> **Experimental** — sandbox support is subject to change prior to General Availability.

An OpenAI Agents SDK `SandboxAgent` runs commands, edits files, and keeps a workspace inside a sandbox. This plugin dispatches each of those operations as an Activity.

Process handles returned by `exec` remain usable across Activities only when the sandbox backend can reconnect them after `resume`; Unix-local and Docker backends cannot.

Register your sandbox backend client under a name on the Worker plugin:

```ts
import { UnixLocalSandboxClient } from '@openai/agents-core/sandbox/local';
import { OpenAIProvider } from '@openai/agents-openai';
import { OpenAIAgentsPlugin, SandboxClientProvider } from '@temporalio/openai-agents';

const plugin = new OpenAIAgentsPlugin({
  modelProvider: new OpenAIProvider(),
  sandboxClientProviders: [new SandboxClientProvider('local', new UnixLocalSandboxClient())],
});
```

Reference the same provider name from Workflow code:

```ts
import { SandboxAgent, shell } from '@openai/agents-core/sandbox';
import { TemporalOpenAIRunner, temporalSandboxClient } from '@temporalio/openai-agents/workflow';

export async function sandboxWorkflow(prompt: string): Promise<string> {
  const agent = new SandboxAgent({
    name: 'Coder',
    model: 'gpt-4o-mini',
    capabilities: [shell()],
  });

  const result = await new TemporalOpenAIRunner().run(agent, prompt, {
    runConfig: { sandbox: { client: temporalSandboxClient('local') } },
  });
  return result.finalOutput ?? '';
}
```

## Credentials

A credential written into an Activity argument stays in Workflow history. Leave it out of Workflow code and supply it on the Worker instead.

Declare the tool in Workflow code:

```ts
import { Agent, hostedMcpTool } from '@openai/agents-core';

const docsTool = hostedMcpTool({ serverLabel: 'docs', serverUrl: 'https://mcp.example.com' });
const agent = new Agent({ name: 'DocsAgent', model: 'gpt-4o-mini', tools: [docsTool] });
```

Supply the credential on the Worker:

```ts
import { OpenAIProvider } from '@openai/agents-openai';
import { OpenAIAgentsPlugin } from '@temporalio/openai-agents';

const plugin = new OpenAIAgentsPlugin({
  modelProvider: new OpenAIProvider(),
  hostedToolCredentials: (identity) => {
    if (identity.tool !== 'hostedMcp' || identity.serverLabel !== 'docs') return undefined;
    const authorization = process.env.DOCS_MCP_TOKEN;
    if (!authorization) throw new Error('DOCS_MCP_TOKEN is not set on this Worker');
    return { authorization };
  },
});
```

For sandbox Manifest environment values, `workerEnvValue(name)` names an allowlisted Worker environment variable that only the Worker resolves.

```ts
import { workerEnvValue } from '@temporalio/openai-agents/workflow';
import { Manifest } from '@openai/agents-core/sandbox';

const manifest = new Manifest({ environment: { DB_PASSWORD: workerEnvValue('WORKER_DB_PASSWORD') } });
```

Allow the variable on the Worker plugin with `resolvableWorkerEnvVars: ['WORKER_DB_PASSWORD']`.

## Tracing

OpenAI Agents SDK tracing works across Client, Workflow, Activity, Nexus, and MCP boundaries. Workflow replay does not duplicate spans.

### OpenAI hosted traces

Enable the upstream hosted exporter before constructing the plugin:

```ts
import { OpenAITracingExporter } from '@openai/agents-openai';
import { addTraceProcessor, BatchTraceProcessor } from '@openai/agents-core';

addTraceProcessor(new BatchTraceProcessor(new OpenAITracingExporter()));
```

**Register this processor in the Worker process, not inside Workflow code.**

**Warning**: Using `setDefaultOpenAITracingExporter()` after constructing the plugin will break plugin internals. Either call it before constructing the plugin, or use `addTraceProcessor` as described above.

### Tracing keys per run

The exporter sends each span with its trace's key, or with its default key (`OPENAI_API_KEY` or `setTracingExportApiKey()`) when the trace has none. A trace's own tracing key, set with `tracing: { apiKey }` on a run or `tracingApiKey` on a trace, is not supported. The plugin passes traces to Workflows, Activities and Nexus handlers in headers, which are stored in Event History, so it cannot pass the key, and their spans would be exported with another key. In Workflow code, `TemporalOpenAIRunner.run` with `tracing: { apiKey }` or inside such a trace, `agent.asTool()` with `tracing: { apiKey }` in its `runConfig` or `runOptions`, and `withTrace(name, fn, { tracingApiKey })`, throw. A key set on those options later fails the nested run, and the error is the tool's output. Starting, signaling, querying or updating a Workflow, or scheduling an Activity, child Workflow or Nexus operation, under such a trace throws too. Set the tracing key on every Client and Worker process instead. An Activity or Nexus handler called under a trace runs in the trace the plugin restored there: an agent run joins it and ignores `tracing: { apiKey }`, and the Worker's `tracingApiKey` option gives its spans a key. When the caller had no trace, the run starts its own trace and keeps `tracing: { apiKey }`, and the resolver is not called. A trace that the Activity or handler starts itself, with `withTrace(name, fn, { tracingApiKey })`, keeps its key in both cases; it is separate from the Workflow's trace, and Temporal calls under it throw. A run that used such a key in Workflow code before you deploy this version gets the error when it replays: let those runs finish first, or use Worker Versioning.

To use a different key per run, resolve it on the Worker with `tracingApiKey`. The plugin calls it for Workflow spans, and for spans made in Activities (including stateful MCP operations) and Nexus handlers under the trace the plugin restored there. It gets the trace ID, and when known the trace name and the Workflow ID and type, and returns a key or `undefined` for the default:

```ts
const plugin = new OpenAIAgentsPlugin({
  modelProvider: new OpenAIProvider(),
  tracingApiKey: ({ workflowId }) => (workflowId?.startsWith('tenant-a-') ? tenantATracingKey : undefined),
});
```

The resolver runs only on Workers. A trace that a Client starts, and the spans the Client makes in it, keep the Client's default key. To keep a run under one key, start the Workflow outside a Client trace, so that the run starts its own trace, or have the resolver return the Client's key for that trace.

### OpenTelemetry

Install `@opentelemetry/sdk-trace-base` before importing the OTel subpath:

```bash
# Or `pnpm add`/`yarn add`
npm install @opentelemetry/sdk-trace-base
```

```ts
import { trace } from '@opentelemetry/api';
import { OpenAIProvider } from '@openai/agents-openai';
import { OpenAIAgentsPlugin } from '@temporalio/openai-agents';
import { createTracerProvider } from '@temporalio/openai-agents/otel';

trace.setGlobalTracerProvider(createTracerProvider());

const plugin = new OpenAIAgentsPlugin({
  modelProvider: new OpenAIProvider(),
  interceptorOptions: { useOtelInstrumentation: true },
});
```

Use `createTracerProvider` from this package for replay-safe OTel ID generation. If you need a different provider class, configure it with `TemporalIdGenerator` and mark it before registering it:

```ts
import { trace } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { markReplaySafeTracerProvider, TemporalIdGenerator } from '@temporalio/openai-agents/otel';

const provider = markReplaySafeTracerProvider(
  new NodeTracerProvider({
    idGenerator: new TemporalIdGenerator(),
  })
);

trace.setGlobalTracerProvider(provider);
```

Only mark providers that use `TemporalIdGenerator`; the marker opts into the plugin's replay-safety checks.

### Temporal orchestration spans

Set `addTemporalSpans: true` to emit `temporal:*` agent-SDK spans for orchestration operations such as Workflow starts, Signals, Queries, Updates, Activities, child Workflows, Nexus Operations, and continue-as-new:

```ts
const plugin = new OpenAIAgentsPlugin({
  modelProvider: new OpenAIProvider(),
  interceptorOptions: { addTemporalSpans: true },
});
```

These spans are agent-SDK spans, so they reach the hosted OpenAI dashboard, custom `TracingProcessor`s, and OTel when enabled.

## Pre-built workflow bundles

When using `bundleWorkflowCode`, pass the plugin to the bundler as well as to the Worker. The bundler hook installs Workflow-side Web API polyfills and registers the Workflow trace interceptor.

```ts
import { OpenAIProvider } from '@openai/agents-openai';
import { OpenAIAgentsPlugin } from '@temporalio/openai-agents';
import { bundleWorkflowCode } from '@temporalio/worker';

const plugin = new OpenAIAgentsPlugin({ modelProvider: new OpenAIProvider() });
const bundle = await bundleWorkflowCode({
  workflowsPath: require.resolve('./workflows'),
  plugins: [plugin],
});
```

## Import paths

Most applications use two import paths: `@temporalio/openai-agents` in Worker and Client code, and `@temporalio/openai-agents/workflow` in Workflow code. The other subpaths are for tracing setup or manual Worker wiring.

| Import path                                      | Import from      | Use for                                                     |
| :----------------------------------------------- | :--------------- | :---------------------------------------------------------- |
| `@temporalio/openai-agents`                      | Worker or Client | Plugin setup, MCP providers, model option types             |
| `@temporalio/openai-agents/workflow`             | Workflow         | Runner, Workflow-safe tools, sessions, MCP handles          |
| `@temporalio/openai-agents/otel`                 | Worker or Client | Replay-safe OpenTelemetry setup                             |
| `@temporalio/openai-agents/workflow-interceptor` | Worker bundling  | Manual `workflowInterceptorModules` wiring without a plugin |

`@temporalio/openai-agents` exports `OpenAIAgentsPlugin`, MCP provider classes, `SandboxClientProvider`, model option types, hosted tool credential types, `DEDICATED_WORKER_FAILURE_TYPE`, `TOOL_FILTER_UNSUPPORTED_TYPE` and `TOOL_META_RESOLVER_UNSUPPORTED_TYPE`. `OpenAIAgentsTraceClientInterceptor` is also public for Clients that need manual interceptor wiring instead of plugin registration.

`@temporalio/openai-agents/workflow` exports Workflow-safe APIs: `TemporalOpenAIRunner`, `WorkflowSafeMemorySession`, `activityAsTool`, `nexusOperationAsTool`, `agentAsTool` (deprecated), `statelessMcpServer`, `statefulMcpServer`, `temporalSandboxClient`, `workerEnvValue`, related option/definition types, `DEDICATED_WORKER_FAILURE_TYPE`, `TOOL_FILTER_UNSUPPORTED_TYPE` and `TOOL_META_RESOLVER_UNSUPPORTED_TYPE`.

`@temporalio/openai-agents/otel` exports `createTracerProvider`, `TemporalIdGenerator`, `markReplaySafeTracerProvider`, `isReplaySafeTracerProvider`, and `TemporalOpenAIAgentsTracerProviderOptions`. Install optional peer dependency `@opentelemetry/sdk-trace-base` only when you use this OTel integration.

`@temporalio/openai-agents/workflow-interceptor` exports the Workflow trace interceptor module. The plugin registers it automatically; import this path only when assembling Worker interceptor modules yourself.

There is no public TypeScript testing subpath.

## Feature support

Any OpenAI Agents SDK `ModelProvider` can be passed as `modelProvider`. The provider runs in the model Activity, never inside the Workflow sandbox.

| Feature                 | Status                   | Notes                                                                                          |
| :---------------------- | :----------------------- | :--------------------------------------------------------------------------------------------- |
| Multi-turn agents       | Supported                | Agent loop runs durably in the Workflow                                                        |
| Handoffs                | Supported                | `Agent` and `handoff()` forms                                                                  |
| Inline function tools   | Supported                | Must be deterministic                                                                          |
| Activity-backed tools   | Supported                | Via `activityAsTool()`                                                                         |
| Nexus operation tools   | Supported                | Via `nexusOperationAsTool()`                                                                   |
| Nested agent tools      | Supported                | Via `agent.asTool()`                                                                           |
| Hosted tools            | Supported                | Executed server-side by the model provider                                                     |
| Stateless MCP servers   | Supported                | Via `StatelessMCPServerProvider` and `statelessMcpServer()`                                    |
| Stateful MCP servers    | Supported                | Via `StatefulMCPServerProvider` and `statefulMcpServer()`                                      |
| Sandbox agents          | Supported (experimental) | Via `SandboxClientProvider` and `temporalSandboxClient()`                                      |
| Sessions                | Supported                | Via `WorkflowSafeMemorySession`; upstream `MemorySession` is rejected                          |
| Run state and approvals | Supported                | Serialize with `result.state.toString()` and rehydrate with `RunState.fromString`              |
| Guardrails              | Supported                | Guardrail callbacks must be deterministic                                                      |
| Agent lifecycle hooks   | Supported                | `agent.on(...)` listeners run in the Workflow and must be deterministic; see below the table   |
| Tracing                 | Supported                | OpenAI hosted traces, custom `TracingProcessor`s, OTel, and optional `temporal:*` spans        |
| Agent context           | Supported                | Activity tools receive a copy                                                                  |
| `continueAsNew`         | Supported                | Plugin config propagates to the continuation                                                   |
| Child Workflows         | Supported                | Plugin config propagates to children                                                           |
| Local Activities        | Supported                | Set `useLocalActivity: true` in `modelParams`                                                  |
| Run model               | Supported                | `runConfig.model` names the model for agents that set none                                     |
| Streaming               | Supported (experimental) | `run(agent, input, { stream: true })`; requires `streamingTopic` and a hosted `WorkflowStream` |
| Voice agents            | Not supported            |                                                                                                |

As in Node, an `agent.on(...)` listener that throws fails the run. In a nested `agent.asTool()` run, its error becomes the tool's output for the parent model, and the parent run goes on. In a streamed run, its error rejects `result.completed` as it is, as the run's other errors do there, so unless the Workflow catches it, it fails the Workflow Task, which retries, and the run stays open. The framework doesn't await a listener's Promise, so an `async` listener that rejects fails the Workflow Task, which retries.
