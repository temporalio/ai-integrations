# Strands Agents

Release stage: Pre-release.

This Temporal [Plugin](https://docs.temporal.io/develop/plugins-guide) allows you to run [Strands Agents](https://strandsagents.com/) inside Temporal Workflows, routing model invocations, tool calls, and MCP tool calls through Temporal Activities for durable execution, Temporal-managed retries, and timeouts.

## Installation

```sh
npm install @temporalio/strands-agents @strands-agents/sdk
```

## Quickstart

`workflow.ts` defines the workflow:

```ts
import { proxyActivities, defineWorkflow } from '@temporalio/workflow';
import { TemporalAgent } from '@temporalio/strands-agents';

export async function myWorkflow(prompt: string): Promise<string> {
  const agent = new TemporalAgent({
    activityOptions: { startToCloseTimeout: '60 seconds' },
  });
  const result = await agent.invoke(prompt);
  return result.lastMessage?.toString() ?? '';
}
```

`worker.ts` runs the worker:

```ts
import { Worker, NativeConnection } from '@temporalio/worker';
import { StrandsPlugin } from '@temporalio/strands-agents';

const connection = await NativeConnection.connect({ address: 'localhost:7233' });
const worker = await Worker.create({
  connection,
  taskQueue: 'strands',
  workflowsPath: require.resolve('./workflow'),
  plugins: [new StrandsPlugin()],
});
await worker.run();
```

`client.ts` starts the workflow:

```ts
import { Client } from '@temporalio/client';
import { myWorkflow } from './workflow';

const client = new Client();
const result = await client.workflow.execute(myWorkflow, {
  args: ['Hello'],
  taskQueue: 'strands',
  workflowId: 'strands-quickstart',
});
console.log(result);
```

## Models

`new StrandsPlugin({ models })` takes a mapping of `name → factory`. Each factory is called lazily on first use (on the worker, outside the workflow sandbox) and the constructed model is cached for the worker's lifetime. `new TemporalAgent({ model: 'name', ... })` selects which factory to invoke and carries the activity options for that agent's model calls. If `models` is omitted, the plugin registers a single `BedrockModel` factory under the name `"bedrock"`, matching Strands' own implicit default.

```ts
import { BedrockModel } from '@strands-agents/sdk/models/bedrock';
import { AnthropicModel } from '@strands-agents/sdk/models/anthropic';
import { TemporalAgent, StrandsPlugin } from '@temporalio/strands-agents';

// workflow
export async function multiModelWorkflow(prompt: string): Promise<string> {
  const a = new TemporalAgent({
    model: 'claude',
    activityOptions: { startToCloseTimeout: '60 seconds' },
  });
  const b = new TemporalAgent({
    model: 'bedrock',
    activityOptions: { startToCloseTimeout: '60 seconds' },
  });
  // ...
}

// worker
new StrandsPlugin({
  models: {
    claude: () => new AnthropicModel({ apiKey: '...' }),
    bedrock: () => new BedrockModel({}),
  },
});
```

Each `TemporalAgent` carries its own activity options (timeouts, retry policy, task queue, streaming topic) and dispatches to the shared model activity, which resolves the model name against the registered factories at runtime. A name not present in `models` throws inside the activity.

Before its first model call, each `TemporalAgent` reads the model's `modelId` and `contextWindowLimit` from its `getConfig()` through the `strandsGetModelConfig` activity, with the agent's activity options. Strands' traces and [proactive compression](https://strandsagents.com/docs/user-guide/sdk/agents/conversation-management/#proactive-context-compression) then use the real model's values, as they do outside Temporal. If you scaled `compressionThreshold` by the model's window divided by 200,000 to make up for the missing limit, remove that scaling.

A worker model that keeps the conversation server-side, such as `OpenAIModel` with `stateful: true`, gets the agent's `modelState`, such as the last response id, on each model call through `TemporalAgent`, and the agent clears its messages after each turn, as natively. As in Strands, such an agent cannot use a `conversationManager` or `contextManager`; because `TemporalAgent` learns that the model is stateful when it reads the config, it throws at the agent's first invocation instead of at construction. Catch that error, or the workflow task fails and is retried until you remove the manager and redeploy the worker. With no manager given, such an agent keeps Strands' default sliding window manager, which Strands picks before the agent reads the config, where natively it would use `NullConversationManager`. Through the plugin that manager has nothing to trim: the agent clears its messages after each turn, before the manager runs, and a context window overflow reaches the agent as a model error. A custom model that reports `stateful` but does not use `modelState` ran statelessly through earlier versions, with the agent keeping its messages; it now gets only each turn's new message, and an agent that gives it a manager now throws. Report `stateful` as false for such a model to keep the earlier behavior.

`OpenAIModel` calls through `TemporalAgent`, stateful or not, currently work only with `@strands-agents/sdk` releases before 1.15.0. From 1.15.0, Strands passes an `AbortSignal` in the model's stream options, the signal reaches the model activity as a plain object, and the OpenAI client fails with `signal.addEventListener is not a function` before it sends a request. Temporal retries that activity until its retry policy or timeout runs out.

Stateful model calls run as their own activities, `strandsInvokeStatefulModel` and `strandsInvokeStatefulModelStreaming`, chosen from the stateful flag that `strandsGetModelConfig` returns, so a run that uses a stateful model needs the workflow and the config activity from the same version. An earlier workflow that reads this version's config schedules `invokeModel`, and this version then fails to replay the run with a nondeterminism error. This version's workflow that reads an earlier config, which has no flag, keeps the stateless path for the rest of the run, and an upgraded worker fails each of its model calls with a non-retryable `StatefulModelOnStatelessPath` error. Changing a worker model between stateful and stateless, such as adding `stateful: true` to a deployed `OpenAIModel`, also applies only to new runs: an open run keeps the mode its config read recorded, and its model calls fail with `StatefulModelOnStatelessPath`, or with `StatelessModelOnStatefulPath` when the model became stateless. None of these runs recovers. The agent's `invoke()` throws an `ActivityFailure` whose cause has that type, so unless the workflow catches it, the run fails; reset a run you want to continue. Roll out with Worker Versioning, keeping each run's workflow and activity workers on one version, including any task queue set in `activityOptions`. Without it, stop the runs that use a stateful model, replace every worker on those task queues, and then start new runs.

## Retries

`TemporalAgent` disables Strands' built-in `ModelRetryStrategy` so retries are handled exclusively by Temporal. Configure retries via `activityOptions.retry` on `TemporalAgent`, and on the activity options accepted by `workflow.activityAsTool`, `workflow.activityAsHook`, and `TemporalMCPClient`:

```ts
new TemporalAgent({
  activityOptions: {
    startToCloseTimeout: '60 seconds',
    retry: { maximumAttempts: 3 },
  },
});
```

Passing `retryStrategy` to `new TemporalAgent(...)` throws; remove the argument and put the retry config on the activity options instead.

The plugin turns Strands' terminal errors (`ContextWindowOverflowError`, `MaxTokensError` and `StructuredOutputError`) into non-retryable `ApplicationFailure`s whose `type` is the error's name:

- A `ContextWindowOverflowError` from the model activity is not retried. It reaches the agent as Strands' own error, so its conversation manager trims the conversation and calls the model again, as it does outside Temporal. Each round is a new model activity with the whole conversation as its input, recorded in history, and the worker logs it at DEBUG without counting it in its activity failure metric. [Proactive compression](https://strandsagents.com/docs/user-guide/sdk/agents/conversation-management/#proactive-context-compression) trims before the call instead.
- An activity of yours that throws one of the three fails without retries.
- The agent throws `MaxTokensError` and `StructuredOutputError` in workflow code. When one of the three ends a `TemporalAgent` turn, `invoke()` and `stream()` throw a non-retryable `ApplicationFailure` of that type, with the original error as its `cause`, so the run, or the update, fails with it instead of the workflow task failing on every attempt.

A model activity that fails for good, times out or is cancelled ends the turn with that activity's `ActivityFailure`, so the run fails or is cancelled. Strands wraps it in a `ModelError`, which on its own would only fail the workflow task. Other errors Strands raises in workflow code, such as a `ModelError` for a model response it cannot parse, still fail the workflow task on every attempt.

Workflow code that catches these errors around `invoke()` by class sees the `ApplicationFailure` or `ActivityFailure` instead: match on the failure's `type`, or read the original error from its `cause`. Your own activity interceptors, and those of plugins listed before `StrandsPlugin`, run outside the plugin's and see the `ApplicationFailure`; those of plugins listed after it run inside and see the original error.

If your worker lists one of these errors in `workflowFailureErrorTypes`, the run used to fail with a retryable failure, so a workflow retry policy started another run. The plugin's failure is non-retryable, so that run is no longer retried.

## Cancellation

The model gets the model activity's cancellation signal as `cancelSignal`. Once the worker learns that the activity was cancelled, the signal is aborted. Only a model that passes `cancelSignal` to its client stops the call; the model activity then ends as cancelled. In `@strands-agents/sdk` 1.19.0, `OpenAIModel` and `BedrockModel` do this, but `AnthropicModel`, `GoogleModel` and `VercelModel` ignore `cancelSignal`. With a model that ignores it, the provider call keeps running after cancellation, and the activity can still complete with the model's result. The worker learns of the cancellation through the activity's heartbeats, which the model activities send only when `activityOptions.heartbeatTimeout` is set. Without a `heartbeatTimeout`, the model call runs to completion after its activity is cancelled. `agent.cancel()` in workflow code does not cancel the model activity.

Through `TemporalAgent`, cancelling the workflow during a model call cancels the model activity and ends the run as cancelled (see [Retries](#retries)). With the default `TRY_CANCEL` cancellation type, the workflow does not wait for the activity: the run is cancelled at once, and the model call is aborted once the worker learns of the cancellation. With `WAIT_CANCELLATION_COMPLETED`, the workflow waits until the model activity ends as cancelled, or completes with a model that ignores `cancelSignal`, before the run is cancelled:

```ts
import { ActivityCancellationType } from '@temporalio/workflow';

new TemporalAgent({
  activityOptions: {
    startToCloseTimeout: '60 seconds',
    heartbeatTimeout: '10 seconds',
    cancellationType: ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
  },
});
```

## Snapshots

`TemporalAgent.takeSnapshot()` and `TemporalAgent.loadSnapshot()` throw. Temporal's event history already persists workflow state durably at a finer granularity than Strands snapshots, so calling either inside a workflow is redundant.

## Structured Output

Like Strands' `Agent`, `TemporalAgent` supports structured output via `structuredOutputSchema`. The values flow through the model activity unchanged; supply any Zod schema:

```ts
import { z } from 'zod';

const PersonInfo = z.object({ name: z.string(), age: z.number() });

export async function myWorkflow(prompt: string) {
  const agent = new TemporalAgent({
    activityOptions: { startToCloseTimeout: '60 seconds' },
    structuredOutputSchema: PersonInfo,
  });
  const result = await agent.invoke(prompt);
  return result.structuredOutput;
}
```

## Streaming

To forward model chunks to external consumers, pass `streamingTopic: '...'` to `TemporalAgent` and host a workflow stream on the workflow via [`@temporalio/workflow-streams`](../workflow-streams). Each model stream event is published on the named topic from inside the model activity; subscribers read via `WorkflowStreamClient`. Chunks are batched on `streamingBatchInterval` (default `'100 milliseconds'`). If a model call fails after it published events, for example on a context overflow the agent then trims, those events stay on the topic and the next call publishes a whole message after them; each message starts with its own `modelMessageStartEvent`.

```ts
import { WorkflowStream } from '@temporalio/workflow-streams/workflow';
import { WorkflowStreamClient } from '@temporalio/workflow-streams/client';

// workflow
export async function streamingWorkflow(prompt: string) {
  new WorkflowStream();
  const agent = new TemporalAgent({
    activityOptions: { startToCloseTimeout: '60 seconds' },
    streamingTopic: 'events',
  });
  await agent.invoke(prompt);
}

// client
const stream = WorkflowStreamClient.create(client, workflowId);
for await (const item of stream.subscribe(['events'])) {
  console.log(item);
}
```

## Tools

Wrap an activity registered on the worker with `workflow.activityAsTool(name, options)`:

```ts
// activities/index.ts
export async function fetchUser(userId: string): Promise<{ name: string }> {
  // ...
}

// workflow.ts
import { workflow as strandsWorkflow, TemporalAgent } from '@temporalio/strands-agents';
import { z } from 'zod';

export async function toolsWorkflow(prompt: string) {
  const agent = new TemporalAgent({
    activityOptions: { startToCloseTimeout: '60 seconds' },
    tools: [
      strandsWorkflow.activityAsTool('fetchUser', {
        description: 'Fetch user by id',
        inputSchema: z.object({ userId: z.string() }),
        activityOptions: { startToCloseTimeout: '30 seconds' },
      }),
    ],
  });
  return await agent.invoke(prompt);
}

// worker.ts
new Worker({
  // ...
  activities: { fetchUser },
  plugins: [new StrandsPlugin({ models })],
});
```

## Hooks

Strands' [hook system](https://strandsagents.com/) lets you subscribe callbacks to events in the agent lifecycle — invocation start/end, model call before/after, tool call before/after, message added. Pass `hooks=[...]` via Strands' standard plugin API to `TemporalAgent`: every hook event fires in workflow context, so deterministic callbacks just work.

Callbacks run in workflow context, so they must be deterministic: no `Date.now()`, `randomUUID()`, or I/O — same rules as workflow code. For callbacks that need I/O (audit logging, metrics, alerting), use `workflow.activityAsHook()` to dispatch the work as a Temporal activity:

```ts
// activities/index.ts
export async function persistToolCall(toolName: string): Promise<void> {
  // I/O safely in an activity.
}

// workflow.ts
import { workflow as strandsWorkflow } from '@temporalio/strands-agents';
import { AfterToolCallEvent } from '@strands-agents/sdk';

const auditCallback = strandsWorkflow.activityAsHook<AfterToolCallEvent, string>('persistToolCall', {
  activityInput: (event) => event.toolUse.name,
  activityOptions: { startToCloseTimeout: '10 seconds' },
});

agent.addHook(AfterToolCallEvent, auditCallback);
```

`activityInput` extracts serializable values from the event to pass as the activity's input. Events themselves are not serializable because they hold references to the `Agent`, `Tool` instances, etc.

## Human-in-the-loop interrupts

Strands supports interrupts via `toolContext.interrupt(...)` and `event.interrupt(...)`. Both work with the plugin: `agent.invoke()` returns `AgentResult` with `stopReason: 'interrupt'` and `interrupts: [...]`. Pair this with a signal handler that supplies responses, then resume by calling `agent.invoke(responses)`.

Interrupts also survive the activity boundary when raised from an `activityAsTool`-wrapped activity. An activity interceptor the plugin adds to the worker packages the interrupt as a typed, non-retryable `ApplicationFailure`, and `TemporalActivityTool` re-raises it through the agent's interrupt machinery on the workflow side, so `AgentResult.interrupts` is populated just like the in-workflow case. From an activity, throw an error with an `interrupts` array, the shape of Strands' `InterruptError`: each entry needs a `toJSON()` that returns an object with a unique string `name` and, optionally, a `reason`, all JSON-encodable, whatever payload converter the worker uses. With several entries, the agent pauses at each one in turn. A malformed interrupt fails the activity without retries, as type `StrandsMalformedInterrupt`, and reaches the model as an error tool result. Interrupts are benign failures: the worker logs them at DEBUG and does not count them in its activity failure metric.

The interceptor runs only on workers that register `StrandsPlugin`, so register it on every worker that runs an `activityAsTool` activity, including activity-only workers on other task queues.

The plugin keeps the failure converter your client and worker set with `dataConverter.failureConverterPath`, for example to [encode failure messages](https://docs.temporal.io/failure-converter). For activity-tool interrupts to work, your converter must decode a failure of type `StrandsInterrupt` to an `ApplicationFailure` with that type and its details, as the SDK's default converter does. Context overflow recovery reads the failure's type from the failure itself, so it works with any converter built on the SDK's default one.

## Continue-as-new

A chat-style workflow accumulates history with every turn and will eventually hit Temporal's per-workflow history limit. `workflowInfo().continueAsNewSuggested` flips true once the server decides history has grown large enough; check it after each turn and hand off to a fresh run, carrying `agent.messages` and `agent.modelState` as input. A stateful model clears the messages after each turn and keeps its context in the model state, such as the last response id; a run that carries only the messages starts the model's conversation over:

```ts
import { defineSignal, setHandler, condition, continueAsNew, workflowInfo } from '@temporalio/workflow';
import { TemporalAgent } from '@temporalio/strands-agents';
import type { JSONValue, Message } from '@strands-agents/sdk';

export const userSays = defineSignal<[string]>('userSays');
export const endChat = defineSignal('endChat');

export interface ChatInput {
  messages?: Message[];
  modelState?: Record<string, JSONValue>;
}

export async function chatWorkflow(input: ChatInput = {}): Promise<void> {
  const agent = new TemporalAgent({
    activityOptions: { startToCloseTimeout: '60 seconds' },
    messages: input.messages,
    modelState: input.modelState,
  });

  const pending: string[] = [];
  let done = false;
  setHandler(userSays, (prompt) => void pending.push(prompt));
  setHandler(endChat, () => void (done = true));

  while (true) {
    await condition(() => pending.length > 0 || done);
    if (done) return;
    await agent.invoke(pending.shift()!);
    if (workflowInfo().continueAsNewSuggested) {
      await continueAsNew<typeof chatWorkflow>({ messages: agent.messages, modelState: agent.modelState.getAll() });
    }
  }
}
```

## MCP

`new StrandsPlugin({ mcpClients })` takes a mapping of `name → McpClient factory`, mirroring the `models` pattern. The plugin registers per-server `{name}-listTools` and `{name}-callTool` activities. Workflow-side, `new TemporalMCPClient({ server: 'name' })` is a thin handle: it references the server by name and carries the per-call activity options.

```ts
import { McpClient } from '@strands-agents/sdk';
import { TemporalMCPClient } from '@temporalio/strands-agents';

// workflow
export async function mcpWorkflow(prompt: string) {
  const echo = new TemporalMCPClient({
    server: 'echo',
    activityOptions: { startToCloseTimeout: '30 seconds' },
  });
  const agent = new TemporalAgent({
    activityOptions: { startToCloseTimeout: '60 seconds' },
    tools: [echo],
  });
  return await agent.invoke(prompt);
}

// worker
new Worker({
  // ...
  plugins: [
    new StrandsPlugin({
      mcpClients: {
        echo: () => new McpClient({ url: 'http://localhost:8765/mcp' }),
      },
    }),
  ],
});
```

Each factory returns a fully configured `McpClient`. The connection is opened lazily on first use rather than at worker startup.

By default, `TemporalMCPClient` re-lists the server's tools (via `{name}-listTools`) on every agent turn, so an MCP server that is restarted or redeployed mid-workflow — with tools added, removed, or renamed — is picked up. To list the tools just once at the beginning of the workflow and reuse that schema for the workflow's lifetime (one fewer activity per turn), set `cacheTools: true`:

```ts
const echo = new TemporalMCPClient({
  server: 'echo',
  cacheTools: true,
  activityOptions: { startToCloseTimeout: '30 seconds' },
});
```

To amortize connection setup, the `{name}-listTools` and `{name}-callTool` activities share one worker-process MCP connection and reuse it across calls. The connection is disconnected after it sits idle for `mcpConnectionIdleTimeout` (default 5 minutes); the timer resets on every reuse. `mcpConnectionIdleTimeout` accepts a millisecond number or a duration string (e.g. `'30 seconds'`), like `startToCloseTimeout`:

```ts
new StrandsPlugin({
  mcpClients: { echo: () => new McpClient({ url: 'http://localhost:8765/mcp' }) },
  mcpConnectionIdleTimeout: '30 seconds',
});
```

## Observability

`StrandsPlugin` composes cleanly with [`OpenTelemetryPlugin`](../interceptors-opentelemetry). Register `OpenTelemetryPlugin` on the client and the worker, and `StrandsPlugin` on the worker. List `OpenTelemetryPlugin` first, so that an activity span ending in an interrupt sees the benign failure and is not marked as an error. You'll get OTel spans around the model, tool, and MCP activities the plugin schedules, plus any spans Strands itself emits inside `invoke`:

```ts
import { OpenTelemetryPlugin } from '@temporalio/interceptors-opentelemetry';
import { Resource } from '@opentelemetry/resources';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-grpc';

const otel = new OpenTelemetryPlugin({
  resource: new Resource({ 'service.name': 'strands-worker' }),
  spanProcessor: new SimpleSpanProcessor(new OTLPTraceExporter()),
});

// client
const client = new Client({ plugins: [otel] });

// worker
const worker = await Worker.create({
  connection,
  taskQueue: 'strands',
  workflowsPath: require.resolve('./workflow'),
  plugins: [otel, new StrandsPlugin({ models })],
});
```

See the [OpenTelemetry interceptor README](../interceptors-opentelemetry) for exporter setup.

## Pre-built workflow bundles

If you build workflow code with `bundleWorkflowCode` instead of giving the worker `workflowsPath`, pass the plugin to the bundler as well as to the worker. Its bundler hook stubs modules that workflow code cannot load.

```ts
import { bundleWorkflowCode } from '@temporalio/worker';
import { StrandsPlugin } from '@temporalio/strands-agents';

const bundle = await bundleWorkflowCode({
  workflowsPath: require.resolve('./workflows'),
  plugins: [new StrandsPlugin()],
});
```

The bundle holds the plugin's workflow code, such as `TemporalAgent`, so rebuild it when you upgrade the plugin: a bundle built with an earlier version keeps that version's behavior in workflow code, and nothing warns about it.

## Upgrading from 1.24 and earlier

This version changes what workflow code does in the cases below, with no version gate. A run that started on an earlier version can fail to replay on this one with a nondeterminism error:

- **Model config read.** Before its first model call, each agent now reads the model's config through the `strandsGetModelConfig` activity, so a run whose agent ran before the upgrade fails to replay.
- **Your failure converter.** Earlier versions replaced the failure converter your client and worker set with the plugin's, so workflow code now decodes the failures of earlier runs with yours. If yours decodes a failure into another error than the SDK's default converter and that changes which commands the workflow issues, the run fails to replay. If it only changes a value the workflow keeps or returns, the run replays without error and continues with the new value, so replay tests do not catch it.
- **Errors from a turn.** Workflow code that catches errors from `invoke()` by class now gets an `ApplicationFailure` or `ActivityFailure` instead of the Strands error. If it acts on the difference, the run fails to replay.
- **Context overflow.** A run whose workflow code caught a context overflow from the agent now trims and calls the model again on replay instead, and fails to replay.

Let those runs finish before you upgrade, or keep them on the earlier version with Worker Versioning. The same applies to rolling back: a run that started on this version can fail to replay on an earlier one, so let it finish before you roll back.

Other changes when you upgrade:

- A run whose workflow task keeps failing on a Strands terminal error, or on a model call that failed or was cancelled, fails or is cancelled once this version runs, and one stuck on a context overflow trims and continues, as long as it started on a build that already reads the model config. A run started on 1.24 and stuck this way fails to replay at the config read instead: reset or terminate it.
- Rebuild [pre-built workflow bundles](#pre-built-workflow-bundles) with this version.
- `StrandsFailureConverter` is removed. Remove imports of it, and any `failureConverterPath` that points at `@temporalio/strands-agents/lib/failure-converter`.
- The plugin no longer needs to be on the client, where it now does nothing, so you can remove it there.
