# DeepAgentsPlugin — Temporal plugin for LangChain Deep Agents

Make a [Deep Agent](https://github.com/langchain-ai/deepagents) durable by adding
one plugin. Build your agent with `create_temporal_deep_agent(...)` (or vanilla
`create_deep_agent(...)`) inside a `@workflow.defn`, add
`plugins=[DeepAgentsPlugin(...)]` to your Client (or Worker), and each LLM call
and each I/O tool call becomes a Temporal Activity — while the agent's control
loop runs, and deterministically replays, inside the Workflow.

The code you already wrote against `deepagents` does not change: sub-agents,
planning/todo state, the filesystem middleware, human-in-the-loop interrupts, and
`agent.ainvoke(...)` all keep working. You get crash-durability, resumable
human-in-the-loop, and bounded history on top.

> Release stage: [Pre-release](https://docs.temporal.io/develop/python/integrations/deepagents).

## Install

```bash
uv add "temporalio-deepagents"
```

(or `pip install "temporalio-deepagents"`). Requires Python ≥ 3.11 (the same
floor `deepagents` sets).

## Hello world

```python
import asyncio
from datetime import timedelta

from deepagents import create_deep_agent  # no import guard needed; see below
from temporalio import workflow
from temporalio.client import Client
from temporalio.deepagents import (
    DeepAgentsPlugin,
    create_temporal_deep_agent,
)
from temporalio.worker import Worker


@workflow.defn
class ResearchAgent:
    @workflow.run
    async def run(self, question: str) -> str:
        # create_temporal_deep_agent wraps deepagents' create_deep_agent and
        # scopes this agent's model-call activity options explicitly.
        agent = create_temporal_deep_agent(
            model="anthropic:claude-sonnet-5-5",
            system_prompt="You are a careful research assistant.",
            activity_options={"start_to_close_timeout": timedelta(minutes=5)},
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        return result["messages"][-1].content


async def main() -> None:
    # API keys live on the worker via the model provider, never in workflow
    # inputs or history. The default provider is LangChain's init_chat_model.
    plugin = DeepAgentsPlugin()
    # Add the plugin on ONE side. The SDK propagates a Client plugin to any
    # Worker built from that Client, so the Worker below inherits it.
    client = await Client.connect("localhost:7233", plugins=[plugin])
    worker = Worker(
        client,
        task_queue="deepagents-task-queue",
        workflows=[ResearchAgent],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```

Two things worth noticing:

- **No `workflow.unsafe.imports_passed_through()` guard.** The plugin
  configures the workflow sandbox to pass the `deepagents` / LangChain import
  tree through, so workflow files import them like any other module.
- **Vanilla `create_deep_agent(...)` also works.** The plugin substitutes the
  durable model automatically whenever `model=` is a name string; use
  `create_temporal_deep_agent` when you want to scope `activity_options` to one
  agent instead of configuring plugin-wide defaults.

## What this plugin gives you

- **Drop-in durability.** `create_deep_agent(...).ainvoke(...)` runs unchanged
  inside a Workflow. The loop replays deterministically; every nondeterministic
  step (LLM, I/O tool, real filesystem/shell op) is an Activity.
- **One LLM call per Activity.** The Workflow ships only the model *name*; the
  worker's `model_provider` builds the real client. Temporal owns retries and
  timeouts (LLM-SDK retries are disabled).
- **Sub-agents inherit durability.** Because sub-agents inherit the parent's
  `model` object and tools, substituting them once propagates to the whole agent
  tree — no per-sub-agent wiring.
- Tools, real-I/O backends, human-in-the-loop, streaming, and continue-as-new
  each get a section below.

## Configuring activity options

Per agent (recommended): `create_temporal_deep_agent(..., activity_options=...)`
as in Hello world above. Plugin-wide defaults use two keyed maps, because model
calls and tool calls have different timeout profiles:

```python
from datetime import timedelta

from temporalio.deepagents import DeepAgentsPlugin

plugin = DeepAgentsPlugin(
    # A single config, or a map keyed by MODEL name (thinking-mode models get
    # longer timeouts than fast ones).
    model_activity_options={"start_to_close_timeout": timedelta(minutes=5)},
    # A single config, or a map keyed by TOOL name.
    tool_activity_options={"start_to_close_timeout": timedelta(seconds=30)},
)
```

## Tools: the Workflow-vs-Activity choice, made explicit

A tool that only mutates agent state can run in-workflow; a tool that does real
I/O must not. Both directions are one call:

```python
from datetime import timedelta

from langchain_core.tools import tool
from temporalio import activity
from temporalio.deepagents import activity_as_tool, tool_as_activity


@activity.defn
async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"It is sunny and 22C in {city}."


@tool
def web_search(query: str) -> str:
    """Search the web for a query."""
    return f"Top result for {query!r}: ..."


# An existing Temporal activity, exposed to the agent as a tool:
weather_tool = activity_as_tool(
    get_weather, start_to_close_timeout=timedelta(seconds=30)
)
# A LangChain tool whose body does I/O, moved into an activity:
search_tool = tool_as_activity(
    web_search, start_to_close_timeout=timedelta(seconds=30)
)
```

Pass both to `create_temporal_deep_agent(..., tools=[weather_tool,
search_tool])`. An unwrapped, non-builtin tool runs in-workflow and the plugin
warns at construction, so the choice is never silent. Deep Agents' pure
built-ins (`write_todos`, state-backed file tools) stay in-workflow by design.

Call `tool_as_activity(...)` at module import or worker startup on **every
activity worker** that can receive `deepagents.invoke_tool`. Registration is
process-local; calling it only inside a workflow does not register the tool in
another process. With `activity_as_tool(...)`, register the underlying Temporal
activity on every worker polling its task queue.

A wrapped tool gets LangChain messages in its arguments as message objects, as
it does without Temporal. For example, `state["messages"]` from `InjectedState`
holds `HumanMessage` and `AIMessage` objects, not dicts. This covers the message
classes in `langchain_core.messages` and their chunks; a subclass you define
still arrives as a dict. Values inside a message, such as a `ToolMessage`
artifact, keep their JSON form, so a `Document` artifact arrives as a dict.
Update activity workers before or together with workflow workers: an older
activity worker still passes the messages to the tool as dicts.

## Durable file and shell backends

Use `TemporalBackend("name", ...)` for a real-I/O backend
(`FilesystemBackend` / `LocalShellBackend` / a remote sandbox). The agent's
*built-in* file and shell tools execute as durable `deepagents.backend_op`
Activities. Each activity worker constructs the backend from the configuration
sent with the operation, so multiple worker processes and serverless workers
can execute it independently:

```python
from datetime import timedelta

from deepagents.backends import FilesystemBackend
from temporalio import workflow
from temporalio.deepagents import (
    DeepAgentsPlugin,
    TemporalBackend,
    create_temporal_deep_agent,
)
from temporalio.workflow import ActivityConfig


# Configure this plugin on every worker polling backend activities.
plugin = DeepAgentsPlugin(
    backends={
        "filesystem": lambda root_dir: FilesystemBackend(
            root_dir=root_dir, virtual_mode=True
        ),
    }
)


@workflow.defn
class FilesystemAgent:
    @workflow.run
    async def run(self, root_dir: str) -> str:
        backend = TemporalBackend(
            "filesystem",
            factory_argument=root_dir,
            backend_type=FilesystemBackend,
            activity_options=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=30)
            ),
        )
        agent = create_temporal_deep_agent(
            model="anthropic:claude-sonnet-5-5",
            backend=backend,
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Take notes as you work."}]}
        )
        return result["messages"][-1].content
```

Add `plugin` to the Client or Worker as in the Hello world example.
`activity_options` accepts the SDK's `ActivityConfig`, a typed dictionary for
timeouts, retry policy and task queue routing. Existing dictionary literals also
work.
`backend_type` advertises the backend's delete and shell capabilities and method
signatures without constructing it in the workflow. For a sandbox, pass
`metadata={"id": sandbox_id}` if workflow code needs its `id`.

For a remote sandbox, register a factory that reconnects using a serializable
sandbox ID, and pass that ID as `factory_argument`. This follows the MCP plugins'
named factory pattern: the lambda stays on the activity worker; only its name
and argument cross the activity boundary. A factory can accept one positional
argument (`None` by default), or no arguments when everything is configured on
the worker, for example
`{"filesystem": lambda: FilesystemBackend(root_dir="/shared")}`. Factories may be
synchronous or asynchronous and run once per activity attempt. Operations and
retries must reconnect to the same external state. Synchronous factory
construction runs in a thread with the activity context preserved; returned
awaitables run on the activity event loop. Changes held only in a
backend instance's memory do not survive between calls. A local filesystem
path must be accessible to every activity worker that can receive the operation,
or the backend activities must be routed to a task queue whose workers share
that filesystem. The factory argument is recorded in workflow history;
credentials and live connections belong in the worker-side factory.

`TemporalBackend(existing_backend, ...)` also works when the workflow and its
backend activities execute in the same process. It holds the backend in a
process-local registry, so use a named backend factory when activities can run
in another process.

State-only backends (the default) need no wrapping — they are pure workflow
state, replayed deterministically.

## Human-in-the-loop

With `interrupt_on=...`, the agent pauses before a guarded tool and
`ainvoke(...)` returns the pending approval under the SDK-native
`__interrupt__` key — directly in your workflow. Expose it via a Query and
resume with an Update; no shim exception, the native LangGraph resume protocol
is used as-is:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from temporalio import workflow
from temporalio.deepagents import create_temporal_deep_agent


@workflow.defn
class ApprovalAgent:
    def __init__(self) -> None:
        self._pending: str | None = None
        self._decision: str | None = None

    @workflow.run
    async def run(self, request: str) -> str:
        agent = create_temporal_deep_agent(
            model="anthropic:claude-sonnet-5-5",
            interrupt_on={"book_trip": True},
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": workflow.info().workflow_id}}
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": request}]}, config=config
        )
        if result.get("__interrupt__"):
            self._pending = str(result["__interrupt__"][0].value)
            await workflow.wait_condition(lambda: self._decision is not None)
            result = await agent.ainvoke(
                Command(resume={"decisions": [{"type": self._decision}]}),
                config=config,
            )
        return result["messages"][-1].content

    @workflow.query
    def pending_approval(self) -> str | None:
        return self._pending

    @workflow.update
    async def resume(self, decision: str) -> None:
        self._decision = decision
```

A client polls `pending_approval`, shows it to a person, and calls the
`resume` update with `"approve"` / `"reject"`.

## Streaming

Set `streaming_topic=` on the plugin and model dispatch switches to a streaming
Activity that publishes chunk batches to a
`temporalio.contrib.workflow_streams` topic for live subscribers — while the
aggregated final message still returns to the workflow, so the durable result
is identical to the non-streaming path:

```python
from temporalio.deepagents import DeepAgentsPlugin

plugin = DeepAgentsPlugin(streaming_topic="agent-stream")
```

Subscribers read the topic with `WorkflowStreamClient`; each item is an
`AIMessageChunk` in `langchain_core.load.dumpd` form.

## Continue-as-new: what carries and what does not

Long agent invocations grow workflow history. `run_deep_agent(agent, input,
state_snapshot=...)` pauses the graph before its next model step, after the
preceding tools have finished, when the server recommends continuing
(`workflow.info().is_continue_as_new_suggested()`, the default and recommended
mode, which accounts for history length and size). It carries the current graph
checkpoint into a fresh Temporal run and resumes that pending step.

```python
from deepagents import create_deep_agent
from temporalio import workflow
from temporalio.deepagents import run_deep_agent


@workflow.defn
class LongResearchAgent:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
        agent = create_deep_agent(model="anthropic:claude-sonnet-5-5")
        return await run_deep_agent(
            agent,
            input,
            state_snapshot=state_snapshot,
        )
```

Pass `continue_as_new_after=N` instead to trigger on a fixed history-event
count.

- **Carries forward:** messages, state-backed files, todos (if present), and
  serializable middleware state, including fields marked `OmitFromInput`.
  Invocation-local `UntrackedValue` fields, such as model-call counters and
  filesystem caches, remain live between model steps and are restored when that
  same invocation continues in a new run. A new invocation resets them normally.
  The snapshot retains the graph's execution position and remaining recursion
  budget. Agent lifecycle hooks therefore run once per logical invocation;
  `after_agent` runs when the agent finishes, rather than at every rollover.
  Checkpoint values are materialized, including delta-backed messages and files,
  so earlier checkpoint history is unnecessary. Message classes and types
  declared in the graph's state schema are preserved, including nested types.
  Types used in a snapshot must be importable from their defining module;
  classes defined inside a workflow method cannot be restored. Unsupported
  state fails checkpoint creation explicitly instead of silently changing type.
- **Checkpoint storage:** the driver uses an in-workflow `InMemorySaver`, creating
  one when the agent has none. An existing saver must also be an `InMemorySaver`.
  Temporal replay reconstructs it within a run; the checkpoint snapshot crosses
  continue-as-new as input. Large snapshots count toward payload limits, so keep
  large files in a durable backend such as `TemporalBackend`. The snapshot uses
  LangGraph's typed serialization with types allowed by the graph's schema and
  must be resumed with the same graph and middleware definitions.
  Mapping inputs omit `messages` and `files` on rollover because the checkpoint
  already contains them; other input fields remain available to reconstruct the
  agent. The driver does not mutate the caller's original input.
- **Completion and interrupts:** completed invocations return their result.
  Application interrupts, including human approvals, return to the caller.
  For these flows, supply `checkpointer=InMemorySaver()` and configure a stable
  `thread_id` with `agent.with_config(...)` so the caller can resume the agent.
  Explicit static debugging breakpoints disable the driver's automatic rollover.
  Sub-agent tool calls and a parallel batch of tools finish before a boundary;
  they are not split. The snapshot covers the parent graph's channels, including
  completed sub-agent results in its transcript. Sub-agent-local checkpoint
  history and objects held outside the parent graph state are not carried.
- **Recursion limits:** the remaining budget applies to the entire unfinished
  invocation, across all continued runs. Continue-as-new does not reset it.
  Deep Agents defaults to 9,999 graph steps; configure a larger budget with
  `agent.with_config({"recursion_limit": 50_000})` for longer invocations.
  Exhaustion and invalid driver configuration raise `DeepAgentsWorkflowError`
  and fail the workflow instead of repeatedly failing Workflow Tasks.
- **Conversation workflows:** your workflow owns its input queue and between-turn
  rollover policy. This helper resumes an unfinished invocation and expects the
  workflow signature `(input, state_snapshot=None)` shown above. It does not
  restart a completed answer to keep a conversation open. Plain agents and
  wrappers without a compiled graph run once through `ainvoke`.
- **Replay compatibility:** histories predating
  `deepagents.checkpoint-continue-as-new` keep the original transcript/todo-based
  behavior. Old transcript snapshots can seed a fresh graph, but only a graph
  checkpoint restores its execution position. Repeated identical model, tool,
  and backend requests continue to run their own Activities; results are not
  memoized across calls.
  Use Worker Versioning when changing graph or middleware definitions for
  chains that will resume saved checkpoints.

The approval example above uses direct `ainvoke`. To combine approvals with
automatic rollover, use a workflow with the `(input, state_snapshot=None)`
signature and route both the initial call and the approval through the driver:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


@workflow.run
async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
    agent = create_deep_agent(
        model="anthropic:claude-sonnet-5-5",
        checkpointer=InMemorySaver(),
        interrupt_on={"write_file": True},
    ).with_config({"configurable": {"thread_id": workflow.info().workflow_id}})

    result = await run_deep_agent(agent, input, state_snapshot=state_snapshot)
    if result.get("__interrupt__"):
        # Expose the pending approval and wait for the caller's decision, as above.
        await workflow.wait_condition(lambda: self._decision is not None)
        result = await run_deep_agent(
            agent,
            Command(resume={"decisions": [{"type": self._decision}]}),
        )
    return result
```

After rollover the driver restores the checkpoint into the supplied saver under
the same thread ID, so the caller can resume the interrupt through that agent.
Known human-in-the-loop middleware is validated before the first model call;
custom dynamic interrupts are checked when they occur.

For between-turn rollover, a completed result can seed the next invocation as a
transcript/files snapshot. Serialize messages with public LangChain `dumpd` to
preserve their classes through the payload converter:

```python
from langchain_core.load import dumpd

snapshot = {
    "messages": [dumpd(message) for message in result["messages"]],
    "files": result.get("files", {}),
}
workflow.continue_as_new(args=[{"messages": []}, snapshot])
```

The workflow must pass that snapshot into `run_deep_agent` on its next run.
This starts a new invocation with the carried transcript and files; it does not
restore other middleware fields or an unfinished graph's execution position.

## Runtime behavior

While a worker built with this plugin is running, the plugin wraps
`deepagents.create_deep_agent` so a bare `model="provider:name"` string is
auto-routed through an Activity. The wrapper only rewrites arguments when called
*inside a workflow*, so importing `deepagents` on a plain client or activity
worker is unaffected, and the original function is restored when the worker
stops. If you would rather be explicit, use `create_temporal_deep_agent` or
pass `TemporalModel("provider:name")` yourself.

Each plugin instance registers four Activities: `deepagents.invoke_model`,
`deepagents.invoke_model_streaming`, `deepagents.invoke_tool` and
`deepagents.backend_op`. Use one `DeepAgentsPlugin` per Worker: one
`model_provider` receives every model name, so it can serve several models, and
`backends` takes every factory. The same instance on both the Client and the
Worker works. So do one instance on the Client and one on the Worker built with
the same `model_provider` and `backends` factories (the same objects) and the
same `streaming_batch_interval`; two default instances match. Any other second
instance fails at Worker start with a `ValueError` that names
`DeepAgentsPlugin`. With one instance on each side, the Worker instance does
not add an Activity whose name is already
registered, and still registered after its own step, by the same function, the
Client instance's copy of the same built-in Activity, or a wrapper of either
that sets `__wrapped__` (as `functools.wraps` does).
A dynamic Activity counts the same way, with the dynamic slot as its name.
An Activity registered under another name, such as a wrapper with an alias,
does not count. So the Worker keeps what the Client side
registered, including a wrapper a later Client plugin put around it if that
wrapper sets `__wrapped__`, and any
other Activity given to the Worker instance is still added. Any other repeated
Activity name fails at Worker start with `ValueError: More than one activity
named ...`, as it does without the plugin. That includes a caller Activity
with one of these names. A second, different
dynamic Activity fails with `TypeError: More than one dynamic activity`.

## Composing with other plugins

This plugin carries no tracing context of its own. For observability, compose it
with `temporalio.langsmith` or `temporalio.contrib.opentelemetry` —
registration order does not matter:

```python
from temporalio.client import Client
from temporalio.deepagents import DeepAgentsPlugin


async def connect():
    return await Client.connect(
        "localhost:7233",
        plugins=[
            # LangSmithPlugin(),  # or OpenTelemetryPlugin(), in either order
            DeepAgentsPlugin(),
        ],
    )
```

For agents built directly as LangGraph graphs (rather than a compiled Deep
Agent), see `temporalio.langgraph`.
