# LangGraph Plugin for Temporal Python SDK

> Release stage: [Generally Available](https://docs.temporal.io/develop/python/integrations/langgraph).

This Temporal [Plugin](https://docs.temporal.io/develop/plugins-guide) allows you to run [LangGraph](https://www.langchain.com/langgraph) nodes and tasks as Temporal Activities, giving your AI workflows durable execution, automatic retries, and timeouts. It supports both the LangGraph Graph API (``StateGraph``) and Functional API (``@entrypoint`` / ``@task``).

## Installation

```sh
uv add temporalio-langgraph
```

## Plugin Initialization

### Graph API

```python
from datetime import timedelta

from langgraph.graph import StateGraph
from temporalio.langgraph import LangGraphPlugin

g = StateGraph(State)
g.add_node("my_node", my_node, metadata={"execute_in": "activity"})

plugin = LangGraphPlugin(
    graphs={"my-graph": g},
    default_activity_options={"start_to_close_timeout": timedelta(seconds=30)},
)
```

Graph API node functions may be closures or local functions — e.g. built by a node factory as the worker constructs the graph — because each node's activity is named `<graph>.<node>`:

```python
def make_adder(step: int):
    def add(state: State) -> dict[str, int]:
        return {"value": state["value"] + step}

    return add

g.add_node("add_one", make_adder(1), metadata={"execute_in": "activity"})
```

Functional API `@task` functions must be defined at module level — their `module.qualname` is the activity name registered with the worker — so a task defined inside a function raises `ValueError` when the plugin is created. Node functions defined at module level in `__main__` are likewise rejected; put the graph in an importable module. A closure from `__main__` is accepted like any other closure.

When nodes or tasks use a provider package such as `langchain_openai`, define the models, the node and task functions, and the graph or entrypoint in a module of their own, not in the Workflow's module. A Workflow gets its graph or entrypoint by name, with `graph()` or `entrypoint()`, so its module does not need to import that module. If it does, wrap the import in `workflow.unsafe.imports_passed_through()`. Otherwise the sandbox imports the provider package itself, and `Worker(...)` fails with `Failed validating workflow`, caused by a `RestrictedWorkflowAccessError`.

### Functional API

```python
from datetime import timedelta

from temporalio.langgraph import LangGraphPlugin

plugin = LangGraphPlugin(
    entrypoints={"my_entrypoint": my_entrypoint},
    tasks=[my_task],
    activity_options={"my_task": {"execute_in": "activity"}},
    default_activity_options={"start_to_close_timeout": timedelta(seconds=30)},
)
```

### One plugin per Worker

Give each Worker one `LangGraphPlugin` with all the graphs, entrypoints and tasks that its Workflows use:

```python
from datetime import timedelta

from temporalio.langgraph import LangGraphPlugin
from temporalio.worker import Worker

plugin = LangGraphPlugin(
    graphs={"my-graph": g},
    entrypoints={"my_entrypoint": my_entrypoint},
    tasks=[my_task],
    activity_options={"my_task": {"execute_in": "activity"}},
    default_activity_options={"start_to_close_timeout": timedelta(seconds=30)},
)

worker = Worker(
    client,
    task_queue="my-task-queue",
    workflows=[MyWorkflow],
    plugins=[plugin],
)
```

A Workflow can only use the graphs and entrypoints of one `LangGraphPlugin` instance, so a Worker with a second instance raises `ValueError` when it is created. A Worker also gets the plugins of its Client, and those count: a Worker is refused too when its Client has two instances, or when its Client has one and the Worker has another. The same instance on both is applied once. A `Replayer` with a second instance raises `ValueError` too. Workers in one process can each have their own instance.

**Breaking change:** a Worker or a `Replayer` with two instances used to be created. Its Workflows only found the graphs and entrypoints of the last instance. `graph()` and `entrypoint()` raised `KeyError` for the others, so those Workflow Tasks failed and the runs stayed open. Put the graphs, entrypoints, tasks and `activity_options` of both instances into one `LangGraphPlugin`. Activity names do not change, so open runs continue on the new Worker. If the instances had different `default_activity_options`, set those options on each node or task. If they had different `streaming_topic` or `streaming_batch_interval` values, give each instance its own Worker and task queue. Pass each plugin to its Worker, not to a Client that the Workers share.

## Checkpointer

If your LangGraph code requires a checkpointer (for example, if you're using interrupts), use `InMemorySaver`.
Temporal handles durability, so third-party checkpointers (like PostgreSQL or Redis) are not needed.

```python
import langgraph.checkpoint.memory
import typing

from temporalio.langgraph import graph
from temporalio import workflow

@workflow.defn
class MyWorkflow:
    @workflow.run
    async def run(self, input: str) -> typing.Any:
        g = graph("my-graph").compile(
            checkpointer=langgraph.checkpoint.memory.InMemorySaver(),
        )

        ...
```

Functional API entrypoints registered with `@entrypoint(checkpointer=InMemorySaver())` receive a fresh in-memory saver for each Workflow execution instance. Repeated `entrypoint(name)` lookups, invocations and interrupt resumes within that instance share its checkpoints. Other Workflows, replay after eviction, and a new run after continue-as-new start with an empty saver; replay reconstructs its contents from History. Initialize a new run from Workflow input, including any values carried across continue-as-new.

The registered saver supplies its serializer (`serde`). Its existing checkpoints are not imported. Only the standard `InMemorySaver` class is supported: custom checkpointers and subclasses raise `ValueError` when the plugin is constructed, because the plugin cannot safely reproduce their configuration and resources for each Workflow instance.

**Breaking change:** entrypoint checkpoints are no longer shared across Workflow instances, and custom entrypoint checkpointers are refused. Let runs that relied on that state or those checkpointers finish before upgrading, or use Worker Versioning.

## Interrupts and message ids

An Activity node or task that calls `interrupt()` more than once does not finish: every resume gives the newest answer to the first call and stops at the second call again. In an entrypoint, an `interrupt()` in an Activity task has to be the entrypoint's only `interrupt()` call: when the entrypoint function or another task calls `interrupt()` too, the second call gets the first one's answer without asking. Call `interrupt()` once per Activity node or task, or label the node or task `execute_in="workflow"`.

A message that enters the graph without an id gets a random id in Workflow code, and that id is different after the Workflow replays, for example after a Worker restart. Give every message an id before it enters the graph, for example `HumanMessage(content=text, id=str(workflow.uuid4()))`, and set an id on messages that your nodes return without one, such as the `ToolMessage`s of a `ToolNode`. Otherwise removing or replacing a message by id stops working after a replay. A resume keyed by an interrupt id that was read before a replay does not reach its `interrupt()` either, so keep one interrupt pending at a time and resume it with a plain value, `Command(resume=value)`.

## Execution Location

Every node (Graph API) and task (Functional API) must be labeled with `execute_in`, set to either `"activity"` or `"workflow"`. This is required per node/task; it cannot be set in `default_activity_options`.

```python
# Graph API
graph.add_node("my_node", my_node, metadata={"execute_in": "activity"})
graph.add_node("format_node", format_node, metadata={"execute_in": "workflow"})

# Functional API
plugin = LangGraphPlugin(
    tasks=[my_task, format_task],
    activity_options={
        "my_task": {"execute_in": "activity"},
        "format_task": {"execute_in": "workflow"},
    },
)
```

Routing functions given to `add_conditional_edges` always run in the Workflow, so they must be deterministic. They can be `def` or `async def`. The same holds for sync LangChain callables that Workflow code awaits: a `def` tool or a sync-only model called with `ainvoke` in an `execute_in="workflow"` node runs in the Workflow, and runs again on replay. Keep model and tool calls in Activity nodes.

A `@task` that is not listed in `tasks=[...]` also runs in the Workflow. Write entrypoints as `async def` and `await` their tasks: a `def` entrypoint that calls `.result()` on a task fails the Workflow Task with `Result is not set.`

As an Activity node, LangGraph's prebuilt `ToolNode` runs `def` tools. An `async def` tool fails there with `NotImplementedError: StructuredTool does not support sync invocation`. With `execute_in="workflow"`, the graph's `compile()` does not return and the Workflow Task fails with `[TMPRL1101] Potential deadlock detected`. For async tools, await the `ToolNode` in an `async def` Activity node:

```python
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)

async def call_tools(state: MessagesState) -> dict:
    return await tool_node.ainvoke(state)

g.add_node("tools", call_tools, metadata={"execute_in": "activity"})
```

## Activity Options

Options are passed through to [`workflow.execute_activity()`](https://python.temporal.io/temporalio.workflow.html#execute_activity), which supports parameters like `start_to_close_timeout`, `retry_policy`, `schedule_to_close_timeout`, and more. Every Activity node and task needs a `start_to_close_timeout` or a `schedule_to_close_timeout`, set on the node or task or in `default_activity_options`. Without one, the Workflow Task fails with `Activity must have start_to_close_timeout or schedule_to_close_timeout` and the run stays open.

### Graph API

Pass Activity options as node `metadata` when calling `add_node`:

```python
from datetime import timedelta
from temporalio.common import RetryPolicy

g = StateGraph(State)
g.add_node("my_node", my_node, metadata={
    "execute_in": "activity",
    "start_to_close_timeout": timedelta(seconds=30),
    "retry_policy": RetryPolicy(maximum_attempts=3),
})
```

### Functional API

Pass Activity options to the `LangGraphPlugin` constructor, keyed by task function name:

```python
from datetime import timedelta
from temporalio.common import RetryPolicy
from temporalio.langgraph import LangGraphPlugin

plugin = LangGraphPlugin(
    entrypoints={"my_entrypoint": my_entrypoint},
    tasks=[my_task],
    activity_options={
        "my_task": {
            "execute_in": "activity",
            "start_to_close_timeout": timedelta(seconds=30),
            "retry_policy": RetryPolicy(maximum_attempts=3),
        },
    },
)
```

### Heartbeats and `def` nodes

The plugin does not send Activity heartbeats. An `async def` node or task can call `activity.heartbeat()` itself. A `def` node or task cannot: there the call raises `RuntimeError: no running event loop`, so do not set `heartbeat_timeout` for one. A node or task that does not heartbeat keeps running after its Workflow is cancelled or its Activity times out.

`def` Activity nodes and tasks run in the event loop's default thread pool, which has `min(32, CPU count + 4)` threads. A Worker takes up to 100 Activities at once by default, so `def` nodes beyond the pool size wait for a thread while their `start_to_close_timeout` runs. Give the event loop a larger pool before the Worker starts, with `loop.set_default_executor(ThreadPoolExecutor(max_workers=...))`, or set the Worker's `max_concurrent_activities` to the pool size.

### Runtime Context

LangGraph's run-scoped context is available through `runtime.context` in nodes and tasks. Activity nodes and Functional API tasks restore dataclass and Pydantic v2 contexts from registered `context_schema` classes, so they support attribute access just like native LangGraph:

```python
from dataclasses import dataclass

from langgraph.runtime import Runtime

from temporalio.langgraph import graph

@dataclass
class Context:
    user_id: str

async def my_node(state: State, runtime: Runtime[Context]) -> dict:
    return {"user": runtime.context.user_id}

# In the Workflow:
g = graph("my-graph").compile()
await g.ainvoke({...}, context=Context(user_id="alice"))
```

Your `context` object must be serializable by the configured Temporal payload converter, since it crosses the Activity boundary. A dataclass or Pydantic context must be an instance of the declared `context_schema` class itself, and the class must be defined in an importable module. An instance of an undeclared class, of a subclass of the schema, or of a class defined in the file you run fails the Workflow Task at the first Activity node or task with `Context type ... is not registered`, and the run stays open.

Values in `config["configurable"]`, other than LangGraph's own keys, do not reach Activity nodes and tasks. Pass per-run values through `context`. In an Activity node the `config` parameter is a reduced copy of the run's config, so when the node calls a tool or a subgraph, pass `langgraph.config.get_config()` instead. With the `config` parameter, an `interrupt()` in the tool fails with `KeyError: '__pregel_scratchpad'`, and a subgraph that interrupts asks again on every resume. A `@task` that declares a required `config` parameter fails with a `TypeError` for the missing argument. Call `get_config()` in the task instead.

Functional API entrypoints declare the schema with `@entrypoint(context_schema=Context)` and must be registered with the plugin. Their tasks can read it through `get_runtime().context`. A task shared by several entrypoints receives the context type for the current call.

For Pydantic v2 contexts, configure `temporalio.contrib.pydantic.pydantic_data_converter` on the Client used by the Worker and compatible converters on other Clients communicating with the same executions. The plugin rebuilds models with `model_validate`, including nested BaseModels. Validators run again after decoding and must be safe to repeat. The plugin does not replace your configured converter.

`TypedDict` and plain dictionary contexts still use dictionary access, such as `runtime.context["user_id"]`. Dataclass reconstruction uses the dataclass constructor and preserves saved `init=False` fields; it does not restore nested custom types on its own. Other field types that JSON does not have are not restored either: in Activity nodes and tasks a `datetime` or `UUID` field is a string, and a set or tuple is a list. A `StrEnum` field is a plain string. Use a Pydantic context when Activity code needs these types. Scalar or list RootModel contexts, custom serializers that change the context shape, and Pydantic v1 are outside this support. Functional API support still requires Python 3.11 or later.

**Breaking change:** class-based contexts now reach Activity nodes and tasks as their declared type. Replace dictionary-based workarounds such as `runtime.context["user_id"]` with `runtime.context.user_id`. Passing a dictionary to a declared class schema does not preserve dictionary access: LangGraph converts it to the schema type. Context schema IDs also extend the serialized Activity input. Declare the context's class as `context_schema` before you upgrade: a run that passes an undeclared class fails its Workflow Tasks after the upgrade.

### Node and Task Values

Activity nodes and tasks get their arguments, and return their results, with the same types as in native LangGraph. After a value crosses the Activity boundary, the plugin rebuilds:

- Pydantic models and dataclasses found in a registered graph's schemas or in a node's or task's type hints, including the types those reference and their subclasses
- LangChain messages
- `bytes`
- `Command`s in a list a node returns, as `ToolNode` does when a tool returns a `Command`

The configured payload converter rebuilds each value, as it would an Activity parameter of that class, so it also decides the types of the values inside. For Pydantic models, configure `pydantic_data_converter` as described for contexts above. With that converter, `bytes` that are not valid UTF-8 do not cross the Activity boundary: an Activity that returns them fails with `invalid utf-8 sequence`, and when Workflow code produced them the Workflow Task fails. Encode binary data as text first, for example with base64. The converter rebuilds a field with its declared class, so declare messages in a Pydantic or dataclass state as `AnyMessage`, not `BaseMessage`, to keep each message's class. Dataclasses keep their saved `init=False` fields, but a dataclass inside another value computes them again. Values of other classes, and values the converter can't rebuild, such as a dataclass with a `Protocol` field, still arrive as decoded JSON, usually dictionaries. All Workers must register the same graphs and tasks and import the same message classes. A Worker that gets a value of a class it doesn't know raises a non-retryable error: in Workflow code this fails the Workflow, and in an Activity it fails the Activity without a retry. Deploy a new class to every Worker before any node or task returns it, or use Worker Versioning. Event History records each value's class by module and qualified name, so moving or renaming a class breaks replay of the runs that recorded it. In Workflow code, rebuilt values use the classes the Worker loaded outside the sandbox; import model modules with `workflow.unsafe.imports_passed_through()` to use them in `isinstance` checks or `model_validate`.

**Breaking change:** code that reads these values with dictionary access, such as `state["answer"]["value"]`, must use attribute access, such as `state["answer"].value`. Code that validates values again, such as `Answer.model_validate(value)` or `convert_to_messages(messages)`, keeps working, except `model_validate` in Workflow code with a model class the sandbox loaded itself; import its module with `imports_passed_through()` as above. `convert_to_messages` also stops restoring message classes when a Pydantic or dataclass state declares its messages as `BaseMessage`: nodes get bare `BaseMessage` objects, so declare `AnyMessage` as described above. Activity payloads keep their format and only gain type information, so older Workers read new payloads as before, and histories recorded before the upgrade replay unchanged, apart from the task cache case below. Results recorded before the upgrade still replay as dictionaries, and nodes and tasks that run later in the same run receive them as dictionaries. Deploy Workflow, node and task code that reads Activity results by attribute with Worker Versioning or after open runs finish. Task cache keys for arguments that are models, dataclasses or sets now come from their class and contents, so a run that carries a task cache across continue-as-new can fail replay with a non-determinism error on the other version, after an upgrade or a rollback. Let such runs finish before you upgrade, or upgrade with Worker Versioning. Runs started on this version may not replay on an earlier one, so roll back with Worker Versioning.

## Error Handlers

A default error handler, set with `set_node_defaults(error_handler=...)`, runs when a node fails, as in LangGraph. LangGraph adds the handler as a node when the graph is compiled, so it takes no `execute_in` label and always runs in the Workflow. It must be deterministic.

```python
from langgraph.errors import NodeError

async def recover(state: State, error: NodeError) -> dict[str, str]:
    return {"status": f"{error.node} failed"}

g = StateGraph(State)
g.set_node_defaults(error_handler=recover)
g.add_node("charge", charge, metadata={"execute_in": "activity"})
```

When an Activity node fails, the handler runs after Temporal's retries are used up. `error.error` is then the `ActivityError`. Its `cause` is an `ApplicationError` if the node raised an exception, or a `TimeoutError` if the Activity timed out.

A Workflow cancellation that arrives while an Activity node runs can reach the handler the same way, with a `CancelledError` as the `cause`. If the handler returns, the Workflow completes instead of closing as cancelled. To keep the cancellation, raise `error.error` again when `temporalio.exceptions.is_cancelled_exception(error.error)` is true.

With LangGraph before 1.2.18, `ainvoke` still raises a handled error if the failed node ran in parallel with another node.

A handler for a single node, set with `add_node(..., error_handler=...)`, is not supported: `LangGraphPlugin` raises `ValueError` for it.

## Continue-as-new

`cache()` returns the results of the Activity nodes and tasks called so far, as a dictionary. Pass it to the next run through `workflow.continue_as_new()` and give it to `graph(name, cache=...)` or `entrypoint(name, cache=...)` there. When the new run calls a node or task with the same input as a call in the cache, it gets the cached result and no Activity runs.

```python
from typing import Any

from temporalio import workflow
from temporalio.langgraph import cache, graph


@workflow.defn
class SummarizeWorkflow:
    @workflow.run
    async def run(
        self, documents: list[str], carried: dict[str, Any] | None = None
    ) -> list[str]:
        app = graph("summarize", cache=carried).compile()
        summaries = []
        for document in documents:
            if workflow.info().is_continue_as_new_suggested():
                workflow.continue_as_new(args=[documents, cache()])
            result = await app.ainvoke({"document": document})
            summaries.append(result["summary"])
        return summaries
```

The next run starts the loop again. The documents an earlier run finished come from the cache, and Activities run for the rest. A document that is in the list twice is an exception: see "What comes from the cache" below.

The cache belongs to the Workflow run. `__init__`, `run`, Signal and Update handlers, and nodes and tasks that run in the Workflow share it, so you can call `graph()` and `entrypoint()` in any of them, and no other Workflow sees it. Every graph and entrypoint of the run uses it, and a call without `cache` keeps what an earlier call was given.

### What comes from the cache

Only results from an earlier run come from the cache, each once. A node or task that a run calls again with the same input runs its Activity again, as in LangGraph, so a poll loop sees its result change. When a run made the same call several times, the next run gets the latest result for the first such call and runs the others, and so does every run after it. The same holds for calls made in parallel, such as two `Send`s with the same argument. To get every result back, give each call an input of its own, for example an item's position in a list.

Two calls are the same when the node or task, its arguments and the run `context` are equal. A Graph API node is identified by its graph and node name and a task by its module and function name, so two nodes that share a function keep separate results. The rest of the run's config, such as `thread_id`, `configurable` values and tags, is not compared.

A cached result is returned whatever it holds, a human's answer to an `interrupt()` included. The cache cannot tell a call that the next run repeats from a new call with the same input, so pass it only to a run that starts the same work again. If the same request has to be asked again in the next run, put a value that tells the requests apart into the node's input.

`cache()` holds the latest result of every different call the runs have made so far. It travels in the continue-as-new input, so it counts toward the payload size limit.

### Values you carry yourself

Equal arguments have equal values of equal types. Results that come from the cache keep their types. A message list or a Pydantic model that you carry through `workflow.continue_as_new()` yourself comes back as plain JSON unless the Workflow declares its type. A dictionary is not the same input as the object an Activity returned, so the node runs again, and it gets a dictionary where it got an object before.

Carry such values with their types. One way is a typed Workflow parameter, for example a Pydantic model with `messages: list[AnyMessage]`, together with `pydantic_data_converter`. The other is to rebuild them before you call the graph: `TypeAdapter(list[AnyMessage]).validate_python(...)` for messages, `Answer.model_validate(...)` for a model. A graph whose state schema is a Pydantic model rebuilds its own fields. `convert_to_messages()` is not enough, and neither is passing the carried dictionaries into an `add_messages` state: LangChain rebuilds an `AIMessage` from a dictionary with `invalid_tool_calls` and `usage_metadata` in its `additional_kwargs`, which the message the Activity returned does not have.

A message that a node returns without an `id` gets a random one in the Workflow, new in every run, so the nodes that get that message as input run again after continue-as-new. `ToolNode` returns its tool messages without an `id`.

### Upgrading

**Breaking change:** earlier versions returned a call's first result to every later call with the same input, in the same run too. They kept the cache in a context variable that Workflows on one Worker thread could share, and they identified a Graph API node by its function. A Workflow that makes the same call several times now runs an Activity for each one, and after continue-as-new it runs all but the first again: see "What comes from the cache". A run recorded on an earlier version fails replay on this one with a non-determinism error if the cache gave it a result for a call the same run had made before, a result of another Workflow, or, after continue-as-new, the result of a Graph API node. Let such runs finish before you upgrade, or upgrade with Worker Versioning. A chain of runs that continues as new across the upgrade runs its finished Graph API nodes once more, because the cache it carries identifies them by function. It still finds its task results. Runs started on this version may not replay on an earlier one, so roll back with Worker Versioning. Calling `graph()` or `entrypoint()` again no longer empties the cache, and `graph(name, cache=carried)` no longer adds results to `carried`: read them with `cache()`.

## Summaries

Summaries are short, human-readable labels that show up in the Temporal UI and CLI, making it easier to see what each step of a run is doing.

### Static summary

`summary` is an ordinary Activity option, so a fixed per-node label works today — pass it like any other option:

```python
g.add_node("plan", plan, metadata={"execute_in": "activity", "summary": "Planning step"})
```

For an `execute_in="activity"` node it is attached to the node's scheduled-activity event. For an `execute_in="workflow"` node it sets the workflow's current details, as a `summary_fn` does (see below).

### Dynamic summary (`summary_fn`)

To derive the label from the node's input at runtime, supply a `summary_fn`. It receives the node's `(args, kwargs)` and returns a summary string, or `None`/`""` for no summary. For a `StateGraph` node `args[0]` is the state; for a Functional `@task` it is the task's arguments.

```python
def summarize(args, kwargs) -> str | None:
    state = args[0]
    return f"stage={state['stage']} doc={state['doc_id']}"

# Graph API: per-node
g.add_node("plan", plan, metadata={"execute_in": "activity", "summary_fn": summarize})

# Functional API: per-task
plugin = LangGraphPlugin(
    tasks=[plan],
    activity_options={"plan": {"execute_in": "activity", "summary_fn": summarize}},
)
```

`summary_fn` is set per node/task (like the static `summary`), so different nodes — which receive different inputs — can compute their summaries independently. You can also put a `summary` or `summary_fn` in `default_activity_options` as a fallback for every node; a node/task that sets either form overrides the inherited default (you just can't set both forms at the same level).

- For `execute_in="activity"` nodes the result sets the activity `summary` (one per scheduled-activity event, visible in history).
- For `execute_in="workflow"` nodes there is no activity, so the result updates the workflow's current details via [`workflow.set_current_details()`](https://python.temporal.io/temporalio.workflow.html#set_current_details). This is a single workflow-level slot (last-writer-wins) reflecting the most recent workflow-bound node that defines a `summary_fn`; a `None`/`""` result clears it. It is queryable via `__temporal_workflow_metadata`.

`summary_fn` runs in workflow context on every replay, so it **must be deterministic and must not raise** (an exception fails the workflow task). Setting both a static `summary` and a `summary_fn` on the same node raises `ValueError`.

## Streaming

Streaming support is experimental and may change in future versions.

When `streaming_topic` is set on `LangGraphPlugin`, calls to `langgraph.config.get_stream_writer()` inside a node publish to the named topic on the workflow's [`WorkflowStream`](https://github.com/temporalio/sdk-python/tree/1.36.0/temporalio/contrib/workflow_streams). Activity-side nodes publish via `WorkflowStreamClient` (a signal carrying batched items, controlled by `streaming_batch_interval`); workflow-side nodes publish synchronously to the in-workflow stream (no signal). External subscribers consume the stream with `WorkflowStreamClient.create(...).topic(...).subscribe(...)`. A node that declares a required `writer` parameter fails with a `TypeError` for the missing argument, so call `get_stream_writer()` inside the node.

The workflow **must** construct `WorkflowStream()` in its `@workflow.init` (i.e. `__init__`). This applies to every workflow type on the worker, including one that uses no graph: without it, the workflow's first workflow task fails.

```python
from datetime import timedelta
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

from temporalio import workflow
from temporalio.client import Client
from temporalio.langgraph import LangGraphPlugin, graph
from temporalio.contrib.workflow_streams import WorkflowStream, WorkflowStreamClient
from temporalio.worker import Worker


class State(TypedDict):
    value: str


async def token_node(state: State) -> dict[str, str]:
    writer = get_stream_writer()
    for token in ["hello", " ", "world"]:
        writer({"token": token})
    writer({"done": True})
    return {"value": "hello world"}


@workflow.defn
class StreamingWorkflow:
    def __init__(self) -> None:
        # Required when streaming_topic is set on the plugin.
        _ = WorkflowStream()
        self.app = graph("streaming").compile()

    @workflow.run
    async def run(self) -> str:
        result = await self.app.ainvoke({"value": ""})
        return result["value"]


async def main(client: Client) -> None:
    g = StateGraph(State)
    g.add_node("token_node", token_node, metadata={"execute_in": "activity"})
    g.add_edge(START, "token_node")

    async with Worker(
        client,
        task_queue="streaming-tq",
        workflows=[StreamingWorkflow],
        plugins=[
            LangGraphPlugin(
                graphs={"streaming": g},
                default_activity_options={
                    "start_to_close_timeout": timedelta(seconds=10)
                },
                streaming_topic="tokens",
            )
        ],
    ):
        handle = await client.start_workflow(
            StreamingWorkflow.run, id="streaming-wf", task_queue="streaming-tq"
        )

        ws_client = WorkflowStreamClient.create(client, handle.id)
        async for item in ws_client.topic("tokens", type=dict).subscribe(from_offset=0):
            print(item.data)
            if item.data.get("done"):
                break

        print(await handle.result())
```

### What's covered, and what isn't

`streaming_topic` wires up exactly **one** LangGraph stream mode: `stream_mode="custom"`, i.e. values written through `get_stream_writer()`. The other modes — `"messages"`, `"values"`, `"updates"`, `"debug"` — are **not** captured by `streaming_topic`. They aren't produced by node-side writers; LangGraph's orchestrator emits them as it walks the graph. The documented pattern is to **bridge `astream()` in the workflow** and republish each yielded chunk to a `WorkflowStream` topic yourself:

```python
@workflow.defn
class AstreamBridge:
    def __init__(self) -> None:
        self.stream = WorkflowStream()
        self.app = graph("g").compile()

    @workflow.run
    async def run(self) -> None:
        topic = self.stream.topic("astream")
        async for chunk in self.app.astream({...}, stream_mode="updates"):
            topic.publish(chunk)
        topic.publish({"done": True})
```

Bridge `"updates"`, `"values"` or `"debug"`. Model tokens emitted inside an
Activity node are not forwarded through the Workflow's `"messages"` mode; publish
them from the Activity to a topic when external subscribers need live tokens.

`app.astream_events(..., version="v2")` also works in Workflow code. Events are
local to the Workflow; publish them to a `WorkflowStream` topic when external
subscribers need them. The plugin schedules stream callbacks only from the same
Workflow event loop; cross-thread callbacks remain unsupported. Existing histories
retain their previous behavior through the `langchain-same-loop-streams` patch.

The shim is also installed before Signal-With-Start and Update-With-Start handlers
run. Generated event `run_id` and `parent_ids` are replay-stable in Workflows;
Activities and ordinary clients retain LangChain's native IDs. This uses the
separate `langchain-replay-stable-run-ids` patch, so existing histories retain
their original ID generation. Explicitly supplied run IDs are preserved. Read-only
Queries and Update validators retain native ID generation.

When installing this plugin alongside `temporalio-langsmith`, upgrade both packages together.
Older LangSmith plugins (including 1.0.0) lack the workflow-local streaming-ID
capability and are rejected before Worker initialization or Replayer history processing, in either
registration order, including plugins inherited from a Client. The check also
applies when the older package is installed but configured on a different Worker,
because its UUID hooks are process-wide. This prevents
recording histories whose child IDs would change after upgrading LangSmith.

This guarantee covers event trace IDs, not every ID inside a message payload.
In particular, LangGraph can generate a new `ToolMessage.id` when a tool returns
a string. Do not use that generated message ID to acknowledge or deduplicate
across Workflow replay. Use `ToolMessage.tool_call_id` to correlate a result with
its originating tool call (for model calls executed in Activities, that call ID
is recorded in the Activity result), or assign your own replay-stable message ID.

### Retry semantics

Streaming has **at-least-once** delivery per activity attempt. When an activity-wrapped node retries (transient failure, worker crash, etc.), the user function re-runs from scratch and re-publishes its writes — earlier publishes from the failed attempt are not rolled back. Subscribers should be ready to see duplicates and recover idempotently (e.g. dedupe on a sequence id you include in each chunk, or treat the stream as advisory and rely on the workflow's final result for state).

## Tracing

We recommend the [Temporal LangSmith Plugin](https://github.com/temporalio/ai-integrations/tree/main/python/langsmith) to trace your LangGraph Workflows and Activities.

## Unsupported LangGraph options

Temporal retries Activities, and Workflow code must be deterministic, so the plugin refuses the LangGraph options below. `LangGraphPlugin(...)` raises `ValueError` when a graph, task or entrypoint sets one.

- `retry_policy` on `add_node`, `set_node_defaults`, `@task` and `@entrypoint`. LangGraph would schedule a failed node's Activity again from the Workflow, on top of Temporal's retries. Set a Temporal `retry_policy` in the [Activity options](#activity-options) instead.
- `TimeoutPolicy(idle_timeout=...)` on `add_node`, `set_node_defaults`, `@task`, `@entrypoint` and `Send`. LangGraph times it with the system clock, so a run can't replay after the timeout fires. Use `run_timeout` instead, or a Temporal `heartbeat_timeout` with `activity.heartbeat()` calls in an `async def` node that runs as an Activity.
- `CachePolicy(ttl=...)` on `add_node`, `set_node_defaults`, `@task` and `@entrypoint`. LangGraph's `InMemoryCache` expires entries by the system clock, so a run may not replay after an entry expires. Use a cache policy without `ttl`.
- `cache` on `@entrypoint`. Every Workflow run on the Worker gets the same entrypoint object, so one run would get another run's results. An entrypoint then has no LangGraph cache, so a cache policy on it or on its tasks does nothing.

A `Send` is created while the graph runs, so the plugin checks its `timeout` then: a node that a `Send` runs under an `idle_timeout` fails with a non-retryable `ApplicationError`, which fails the Workflow unless your code catches it.

A `run_timeout` works, and a number or a `timedelta` passed as `timeout` sets one. In a Workflow its timer is a Temporal timer. When it fires, the Workflow gets LangGraph's `NodeTimeoutError` and asks Temporal to cancel the node's Activity. Don't branch on the error's `elapsed` or on its message: LangGraph measures the elapsed time with the system clock, so it differs on replay.

A cache policy without `ttl` works with a cache that the Workflow creates, such as `graph("my-graph").compile(cache=InMemoryCache())`. The plugin doesn't see the cache you pass to `compile()`. A cache that outlives a Workflow run, such as one from a module imported with `workflow.unsafe.imports_passed_through()`, is shared by every run on the Worker: a run gets another run's results and its History then fails replay.

**Breaking change:** the plugin accepted these options before, except `retry_policy` on `add_node` and `@task`. Building the plugin with one of them now raises `ValueError`, so the Worker doesn't start, and a `Send` with an `idle_timeout` fails its Workflow. Remove the option or replace it as described above. That can change the commands of a run that used the option: a LangGraph retry leaves a timer and another Activity in the History, and a LangGraph timeout leaves a timer. Let such runs finish before you upgrade, or upgrade with Worker Versioning.

## Stores are not supported

LangGraph's `Store` (e.g. `InMemoryStore` passed via `graph.compile(store=...)` or `@entrypoint(store=...)`) isn't accessible inside Activity-wrapped nodes: the Store holds live state that can't cross the Activity boundary, and Activities may run on a different worker than the Workflow. If you pass a store, the plugin logs a warning on first use and `runtime.store` is `None` inside nodes. A node that declares a required `store` parameter fails with a `TypeError` for the missing argument.

Use Workflow state for per-run memory, or an external database (Postgres/Redis/etc.) configured on each worker if you need shared memory across runs.

## Errors raised in the Workflow

An error that LangGraph raises in Workflow code, such as `langgraph.errors.GraphRecursionError`, fails the Workflow Task and not the Workflow: the run stays open and the task is retried. To fail the Workflow instead, name the error types on the Worker with `Worker(..., workflow_failure_exception_types=[GraphRecursionError])`.

## Payload size

The Activity input of a node holds the node's whole input state and the run `context`. The Activity input of a task holds its arguments, the run `context` and the entrypoint's `previous` value. The Activity result holds what the node or task returns. Each of these can pass Temporal's payload size limit, so store large values elsewhere and pass a reference.

## Running Tests

Install dependencies:

```sh
make sync
```

Run the test suite:

```sh
make test
```

Tests start a local Temporal dev server automatically — no external server needed.
