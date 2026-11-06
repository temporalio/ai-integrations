# LangGraph Plugin for the Temporal TypeScript SDK

Release stage: Pre-release.

This Temporal [Plugin](https://docs.temporal.io/develop/plugins-guide) runs
[LangGraph.js](https://langchain-ai.github.io/langgraphjs/) nodes and tasks as
Temporal Activities, or inline in the Workflow, giving your AI workflows durable
execution, retries and timeouts. It supports the Graph API (`StateGraph`) and
the Functional API (`entrypoint()` / `task()`).

## Installation

```sh
npm install @temporalio/langgraph @langchain/langgraph @langchain/core
```

`@langchain/langgraph` (1.5 or later) and `@langchain/core` (1.1 or later) are
peer dependencies. Node.js 22.12 or later is required.

## Defining graphs

Graphs, entrypoints and tasks are declared once with `defineLangGraph()`, in a
module that both the Worker and the Workflows import. Import it from
`@temporalio/langgraph/workflow`, the entry point that is safe to load in the
Workflow sandbox:

```ts
// graphs.ts
import { Annotation, START, StateGraph } from '@langchain/langgraph';
import { defineLangGraph } from '@temporalio/langgraph/workflow';

const State = Annotation.Root({ value: Annotation<string> });

const agent = new StateGraph(State)
  .addNode('call_model', callModel, { metadata: { executeIn: 'activity', startToCloseTimeout: '1 minute' } })
  .addNode('route', route, { metadata: { executeIn: 'workflow' } })
  .addEdge(START, 'call_model')
  .addEdge('call_model', 'route');

export const langGraph = defineLangGraph({
  graphs: { agent },
  defaultActivityOptions: { startToCloseTimeout: '30 seconds' },
});
```

Workflows compile and run the registered graph:

```ts
// workflows.ts
import { langGraph } from './graphs';

export async function agentWorkflow(input: string): Promise<string> {
  const app = langGraph.graph('agent').compile();
  const result = await app.invoke({ value: input });
  return result.value;
}
```

The Worker registers the plugin with the same definitions:

```ts
// worker.ts
import { Worker } from '@temporalio/worker';
import { LangGraphPlugin } from '@temporalio/langgraph';
import { langGraph } from './graphs';

const worker = await Worker.create({
  taskQueue: 'agent',
  workflowsPath: require.resolve('./workflows'),
  plugins: [new LangGraphPlugin(langGraph)],
});
await worker.run();
```

The plugin registers one Activity per Activity node, named
`<graph name>.<node name>`, and one per Activity task, named after the task.
Keep these names stable: running Workflows replay against them.

A different Activity with one of these names fails with
`Duplicate registration of Activity '<name>'`: a caller Activity, another
integration's Activity, or a `LangGraphPlugin` whose definitions run a different
function under that name. If it is registered before `LangGraphPlugin` (in the
Worker's `activities` or by an earlier plugin), `Worker.create` fails. If a
plugin applied after `LangGraphPlugin` replaces one, `worker.run()` fails before
the Worker polls. The same plugin, or one with the same definitions, may be
registered on both the `NativeConnection` and the Worker.

A Worker that takes a prebuilt `workflowBundle` must build it with
`bundleWorkflowCode({ workflowsPath, plugins: [new LangGraphPlugin(langGraph)] })`,
so the bundle includes the plugin's sandbox setup.

### Functional API

```ts
import { entrypoint, task } from '@langchain/langgraph';
import { defineLangGraph } from '@temporalio/langgraph/workflow';

const research = task('research', async (topic: string) => searchTheWeb(topic));
const write = task('write', async (notes: string) => draft(notes));

export const langGraph = defineLangGraph({
  entrypoints: {
    report: entrypoint('report', async (topic: string) => write(await research(topic))),
  },
  tasks: [research, write],
  activityOptions: {
    research: { executeIn: 'activity', startToCloseTimeout: '2 minutes' },
    write: { executeIn: 'workflow' },
  },
});

// In the Workflow:
const report = await langGraph.entrypoint('report').invoke(topic);
```

Register each entrypoint under the name passed to `entrypoint()`.

## Execution location

Every node and task must set `executeIn` to `'activity'` or `'workflow'`. It
can't be set in `defaultActivityOptions`.

- `'activity'` runs the node or task in an Activity, on any Worker of the task
  queue. Use it for I/O: model calls, tools, network and file access.
- `'workflow'` runs it inline in the Workflow. Its code must be deterministic,
  like any Workflow code.

## Activity options

Any Temporal [`ActivityOptions`](https://typescript.temporal.io/api/interfaces/common.ActivityOptions)
field can be set per node, in its `metadata`, or per task, in
`activityOptions[taskName]`. `defaultActivityOptions` applies to every node and
task that doesn't override it:

```ts
.addNode('call_model', callModel, {
  metadata: {
    executeIn: 'activity',
    startToCloseTimeout: '1 minute',
    retry: { maximumAttempts: 3 },
  },
})
```

The plugin removes its options from the metadata a node sees, so
`config.metadata` holds only your own keys.

LangGraph's own `retryPolicy` (on a node, the graph or a task) and
`errorHandler` nodes are rejected: use Temporal retry options instead.

An Activity node or task can stop when its Activity is cancelled, for example
on a timeout, by listening on `config.signal` (or `getConfig().signal` in a
task), which the plugin sets to the Activity's cancellation signal.

## Context

LangGraph's run context (`invoke(input, { context })`) reaches Activity nodes
and tasks as `config.context` / `getConfig().context`. It travels through the
configured payload converter, so it must be serializable as JSON.

## Values across the Activity boundary

Arguments and results keep their types when they cross the Activity boundary.
The plugin rebuilds LangChain messages, `Date`, `Uint8Array`, and LangGraph's
`Command` and `Send`, wherever they sit inside objects and arrays. Values of
other classes arrive as the payload converter decodes them, usually plain
objects. All Workers must run compatible versions of the plugin: a Worker that
receives a type it doesn't know fails with a non-retryable `ApplicationFailure`
of type `UNKNOWN_VALUE_TYPE_FAILURE_TYPE` (`LangGraphUnknownValueType`, exported
from `@temporalio/langgraph/workflow`).

## Checkpointer and interrupts

If your graph needs a checkpointer, for example for `interrupt()`, use
`MemorySaver`. Temporal provides durability, so external checkpointers are not
needed:

```ts
import { Command, MemorySaver } from '@langchain/langgraph';

const app = langGraph.graph('approval').compile({ checkpointer: new MemorySaver() });
const config = { configurable: { thread_id: workflowInfo().workflowId } };
await app.invoke(input, config);
const state = await app.getState(config);
// ... wait for a Signal with the human's answer ...
const result = await app.invoke(new Command({ resume: answer }), config);
```

An `interrupt()` inside an Activity node or task interrupts the graph in the
Workflow, and `Command({ resume })` resumes it there.

## Continue-as-new

The plugin caches the results of Activity nodes and tasks, keyed by name,
arguments and context. Pass the cache to the next run so completed work isn't
repeated:

```ts
import { continueAsNew } from '@temporalio/workflow';
import { cache } from '@temporalio/langgraph/workflow';

export async function longAgent(input: Input): Promise<Output> {
  const app = langGraph.graph('agent', { cache: input.cache }).compile();
  const state = await app.invoke(input.state);
  if (needsAnotherRun(state)) {
    await continueAsNew<typeof longAgent>({ state, cache: cache() });
  }
  return state;
}
```

## Summaries

`summary` sets a static Activity summary for a node or task. `summaryFn`
derives one from the arguments (`args[0]` is the state of a graph node), or
returns `null` or `''` for none:

```ts
.addNode('plan', plan, {
  metadata: { executeIn: 'activity', summaryFn: ([state]) => `stage=${(state as PlanState).stage}` },
})
```

Either can also go in `defaultActivityOptions`; a node or task that sets one
overrides the default. Setting both at the same level is rejected.

For a Workflow node, the result sets the Workflow's current details
(last writer wins, `null` or `''` clears them), which the
`__temporal_workflow_metadata` query reports. `summaryFn` runs in the Workflow,
on every replay, so it must be deterministic and must not throw.

## Streaming

With `streamingTopic`, values written through LangGraph's `getWriter()`
(stream mode `custom`) are published to that topic of the Workflow's
[`WorkflowStream`](https://github.com/temporalio/sdk-typescript/tree/main/packages/workflow-streams).
The Workflow creates the stream and passes it to `graph()` or `entrypoint()`:

```ts
// graphs.ts
export const langGraph = defineLangGraph({ graphs: { chat }, streamingTopic: 'tokens' });

// workflows.ts
import { WorkflowStream } from '@temporalio/workflow-streams/workflow';

export async function chatWorkflow(input: string): Promise<string> {
  const stream = new WorkflowStream();
  const app = langGraph.graph('chat', { stream }).compile();
  return (await app.invoke({ value: input })).value;
}

// client.ts
import { WorkflowStreamClient } from '@temporalio/workflow-streams/client';

const stream = WorkflowStreamClient.create(client, workflowId);
for await (const item of stream.topic('tokens').subscribe(0)) {
  console.log(item.data);
}
```

Activity nodes publish in batches by Signal (`streamingBatchInterval`, 100 ms by
default); Workflow nodes publish directly. Delivery is at least once per
Activity attempt: a retried node publishes its writes again.

Other stream modes (`updates`, `values`, `messages`, `debug`) come from the
graph loop in the Workflow. Publish them yourself from `app.stream()`:

```ts
const topic = stream.topic('updates');
for await (const chunk of await app.stream(input, { streamMode: 'updates' })) {
  topic.publish(chunk);
}
```

## Stores are not supported

A LangGraph `Store` can't cross the Activity boundary. If you pass one, the
plugin logs a warning and Activity nodes don't receive it. Use Workflow state
for per-run memory, or an external database for shared memory.

## Differences from the Python plugin

- Graphs are registered with `defineLangGraph()`, shared by the Worker and the
  Workflows, instead of a process-wide registry, because TypeScript Workflows
  run in a separate bundle.
- Options use TypeScript SDK names: `executeIn`, `startToCloseTimeout`,
  `retry`, `summaryFn`, `defaultActivityOptions`.
- `summaryFn` receives the arguments array (`args`), with no keyword arguments.
- Workflows pass their `WorkflowStream` to `graph()` / `entrypoint()` when
  streaming is on.
- LangGraph.js has no `version: 'v2'` invoke option; read interrupts with
  `getState()` or the result's `__interrupt__`.
- Contexts are JSON values; there is no class-based context schema rebuilding.

## Development

```sh
make sync
make lint
make test
make build
```

Tests start a local Temporal dev server and need no model credentials.
