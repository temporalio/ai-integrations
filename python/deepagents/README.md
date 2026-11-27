# DeepAgentsPlugin — Temporal plugin for LangChain Deep Agents

Make a [Deep Agent](https://github.com/langchain-ai/deepagents) durable by adding
one plugin. Build your agent with `create_temporal_deep_agent(...)` (or vanilla
`create_deep_agent(...)`) inside a `@workflow.defn`, add
`plugins=[DeepAgentsPlugin(...)]` to your Client (or Worker), and each LLM call
and each I/O tool call becomes a Temporal Activity — while the agent's control
loop runs, and deterministically replays, inside the Workflow.

The code you already wrote against `deepagents` does not change: sub-agents,
the filesystem middleware, human-in-the-loop interrupts, and
`agent.ainvoke(...)` all keep working. You get crash-durability and resumable
human-in-the-loop on top. The Limits section lists what does not carry over.

> Release stage: [Public Preview](https://docs.temporal.io/develop/python/integrations/deepagents).

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

- **No `workflow.unsafe.imports_passed_through()` guard for `deepagents` and
  LangChain.** The plugin configures the workflow sandbox to pass the
  `deepagents` / LangChain import tree through, so workflow files import them
  like any other module. A module of your own that wraps tools with
  `tool_as_activity` does need the guard: see the Tools section.
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
- **Sub-agents inherit durability.** A sub-agent you declare as a dict runs its
  model calls as Activities too, with the parent's model or with its own
  `model` name. A `CompiledSubAgent` is a graph you built yourself, so build
  its runnable with `TemporalModel(...)`: a provider model there is called from
  Workflow code, not from an Activity.
- Tools, real-I/O backends, human-in-the-loop, streaming, and continue-as-new
  each get a section below.

## Model profiles

Deep Agents sizes summarization from the model's LangChain profile
(`max_input_tokens`), and LangChain reads the same profile to decide how to
send `response_format=<schema>`. Only the model name reaches the Workflow, so
fetch the profile of the Worker's model before building the agent:

```python
from temporalio import workflow
from temporalio.deepagents import (
    TemporalModel,
    create_temporal_deep_agent,
    get_model_profile,
)


@workflow.defn
class ResearchAgent:
    @workflow.run
    async def run(self, question: str) -> None:
        model = "anthropic:claude-haiku-4-5"
        profile = await get_model_profile(model)
        agent = create_temporal_deep_agent(
            model=TemporalModel(model=model, profile=profile),
            tools=[],
        )
        ...
```

`get_model_profile` runs one Activity, registered by `DeepAgentsPlugin`, that
returns `model_provider(name).profile`, and History records the result, so a
later change on the Worker does not affect runs in flight.
`create_temporal_deep_agent` keeps a `TemporalModel`'s profile when it applies
`activity_options`.

Without the profile, LangChain picks the structured-output strategy from its
own list of model names. A model that is not on the list gets the schema as one
more tool and is forced to call a tool (`tool_choice="any"`), instead of the
provider's native structured output. `claude-sonnet-5-5` is not on the list as
of langchain 1.5.0, and the profile langchain-anthropic ships for it says it
does not support a forced tool choice. Fetch the profile before you pass
`response_format` with that model.

## Configuring activity options

Per agent (recommended): `create_temporal_deep_agent(..., activity_options=...)`
as in Hello world above. Plugin-wide options use two keyed maps, because model
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

`model_activity_options` is a default: an agent's own `activity_options` win.
`tool_activity_options` works differently. It applies to tools wrapped with
`tool_as_activity`, and its `start_to_close_timeout` replaces the one you pass
to `tool_as_activity`. It is read when `tool_as_activity(...)` runs, not when
the tool is called, so a tool wrapped before the plugin is constructed does not
get it. That is the usual order for a tools module imported under
`workflow.unsafe.imports_passed_through()`, as the Tools section recommends.
`activity_options` passed to `tool_as_activity` win in either case, so set a
tool's options there.

When you set nothing, a model call and `get_model_profile` get a 5-minute
start-to-close timeout and a 10-second heartbeat timeout. A backend operation
gets 1 minute and no heartbeat timeout.

Model calls — `invoke_model` and `invoke_model_streaming` — heartbeat on a
background task and are scheduled with a default `heartbeat_timeout` of 10
seconds, as in the other plugins in this repository, never exceeding the schedule's own start-to-close (or
schedule-to-close) timeout. The heartbeater beats at half the timeout, and LangChain
runs sync models on worker threads, so slow model calls of any length keep
heartbeating. A synchronous `model_provider` is called on the activity's event
loop, because providers may create loop-bound objects such as an `aiohttp`
session; a provider that blocks longer than the heartbeat timeout (loading a
local model) should set `DeepAgentsPlugin(model_provider_in_thread=True)` so the
heartbeater keeps beating while it builds — it then must not depend on the
running loop. The heartbeat default changes how a dead worker or a
cancellation is noticed, not how long a healthy call may run. It buys two
things: a crashed worker's model call is timed out within the heartbeat timeout
and retried per its retry policy instead of waiting out
`start_to_close_timeout`, and workflow cancellation — which a Python activity
only observes through a heartbeat — reaches a running model call instead of
never. The SDK sends at most one heartbeat per 80% of the heartbeat timeout,
so cancellation can take up to about the whole heartbeat timeout to arrive
(10 s with the default), not half of it. Override or disable it with
`heartbeat_timeout` (including `None`) in the model activity options above.
`get_model_profile` gets the same default and heartbeats the same way.

Tools and backend operations get no default heartbeat timeout. They run
arbitrary application code with their own execution, progress-reporting and
cancellation behavior, so the application decides when background heartbeats
are appropriate: set `heartbeat_timeout` in `tool_activity_options`, a tool's
`activity_options`, or `TemporalBackend(activity_options=...)`, and the
plugin's `invoke_tool` and `backend_op` activities heartbeat in the background
for you. Those background heartbeats repeat the details of the latest
`activity.heartbeat(...)` call your code made (or, on a retry, the previous
attempt's), so they never replace a saved checkpoint with empty details.

`backend_op` calls a sync backend method on the activity's event loop, because
a method may need the running loop (to create an `aiohttp` session, say), and
awaits whatever it returns. A sync method that blocks, such as a shell command,
then also blocks the heartbeater. Set
`DeepAgentsPlugin(backend_ops_in_thread=True)` to run sync backend methods on a
worker thread instead, so they keep heartbeating; they then must not depend on
the running loop. Their `activity.heartbeat(...)` calls still work. A thread
cannot be interrupted, so when the activity is cancelled during such a call it
reports the cancellation only after the method returns. Async backend methods
run on the loop either way, and a sync backend factory always runs on a worker
thread.

Activities you expose with `activity_as_tool(...)` are your own; the plugin
never adds a heartbeat timeout to their schedule.

## Tools: the Workflow-vs-Activity choice, made explicit

A tool that only mutates agent state can run in-workflow; a tool that does real
I/O must not. Both directions are one call. Keep these tools in a module of
their own, for example `myapp/tools.py`:

```python
from datetime import timedelta

from langchain_core.tools import tool
from temporalio import activity
from temporalio.deepagents import activity_as_tool, tool_as_activity
from temporalio.workflow import ActivityConfig


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
    get_weather,
    activity_options=ActivityConfig(start_to_close_timeout=timedelta(seconds=30)),
)
# A LangChain tool whose body does I/O, moved into an activity:
search_tool = tool_as_activity(
    web_search, start_to_close_timeout=timedelta(seconds=30)
)
```

Pass both to `create_temporal_deep_agent(..., tools=[weather_tool,
search_tool])`. An unwrapped tool, sync or async, runs in-workflow, so it has to
be deterministic. The plugin warns about an unwrapped tool passed in the
`tools=` keyword. It does not check `tools` passed by position, a sub-agent's
tools, or tools added by middleware.
Deep Agents' built-in file tools on the default state backend stay in-workflow
by design.

Sync tools, sync graph nodes and routers, LangChain output parsers, and sync
`RunnableLambda`s (or other runnables with only a sync `invoke`) run directly in
the workflow, not in a separate thread as LangChain runs them outside Temporal.
This applies to every workflow in a process that has a worker with the plugin,
not only Deep Agent runs. LangGraph picks how to run a sync node or router when
it builds the graph, so build graphs in workflow code. A graph built before the
worker, outside the sandbox, still fails with `NotImplementedError`. This
includes a graph built at import in a module imported with
`workflow.unsafe.imports_passed_through()`.

Sync functions can compute a result, but cannot wait for a LangGraph task with
`task(...).result()`: the task needs the Workflow event loop to run, and the
call raises `InvalidStateError`. Use an async entrypoint or node and
`await task(...)` instead.

Call compiled graphs with `await graph.ainvoke(...)` or `graph.astream(...)`.
A sync call such as `graph.invoke(...)` or `graph.stream(...)`, including one
from a sync tool, runs in the workflow only while each step runs one node.
LangChain runs a step with several nodes, `batch(...)` and `RunnableParallel`
on its own thread pool, outside the workflow, so in a workflow they fail the
Workflow Task with `RestrictedWorkflowAccessError` before any of that work
starts. Use the async form, or move the call into an activity.

Like other workflow code, sync calls run again on every replay. Their combined
work in one Workflow Task can trigger deadlock detection (two seconds by
default), even if each call takes less than two seconds. Move slow work into
activities.
While one runs, LangChain runs async callback handlers on a separate thread, as
it does outside Temporal, so keep them quick and don't call workflow APIs from
them. Other sync LangChain calls that it would run in a thread, such as a
sync-only cache, retriever or vector store, or `ShellToolMiddleware`, still fail
with `NotImplementedError`; run that work in activities. If the same Worker
also uses `LangGraphPlugin`, its executor override currently runs those calls
inline too.

**Upgrading:** a run that made one of these sync calls on an earlier version
keeps that version's behavior, so its history still replays: the call fails
with `NotImplementedError` again, and LangChain's thread pool stays open to the
run. A run stuck on that error has no recorded call yet, so a worker with this
version runs the call in the workflow, and the run completes. A patch marker,
`deepagents-sync-calls-inline`, records which way each run went.

`activity_as_tool` takes every activity option through `activity_options`, an
`ActivityConfig`: timeouts (`start_to_close_timeout` or
`schedule_to_close_timeout` is required), `retry_policy`, `summary` (default
`"tool:<name>"`), `activity_id`, and routing such as `task_queue`. Route a slow
tool to its own workers with `task_queue="weather-tools"` (a worker registered
for `get_weather` must then poll that queue), or set `heartbeat_timeout` for an
activity that heartbeats, so a lost worker is noticed at the heartbeat instead
of at start-to-close. `arg` and `args` are reserved: the tool builds them from
the model's call.

Call `tool_as_activity(...)` at the top level of the tools module, and import
that module in the Workflow file under
`workflow.unsafe.imports_passed_through()`:

```python
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from myapp.tools import search_tool, weather_tool
```

Registration is process-local, and `deepagents.invoke_tool` runs the tool
object registered last under the tool's name. Without the guard the sandbox
imports the tools module again for each run, `tool_as_activity(...)` runs
again inside it, and the Activity then runs the sandbox's copy of your tool.
In that copy, calls the sandbox restricts raise `RestrictedWorkflowAccessError`
on every attempt: `datetime.now()`, `random`, `pathlib` reads, `subprocess` and
`urllib.request` among them. The same happens to a tool defined in the Workflow
file itself. Do not call `tool_as_activity(...)` inside `run` either. The tool
is then registered only once a run gets there. A Worker that has not run the
Workflow yet, such as one started after a crash, fails the tool Activity with
the non-retryable `DeepAgentsUnknownTool`.

Import the tools module on **every activity worker** that can receive
`deepagents.invoke_tool`. With `activity_as_tool(...)`, register the underlying
Temporal activity on every worker polling its task queue.

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
from deepagents.backends import FilesystemBackend
from temporalio.deepagents import DeepAgentsPlugin

# Configure this plugin on every worker polling backend activities.
plugin = DeepAgentsPlugin(
    backends={
        "filesystem": lambda root_dir: FilesystemBackend(
            root_dir=root_dir, virtual_mode=True
        ),
    }
)
```

Construct the plugin next to the Client or Worker, as in the Hello world
example. Do not construct it at the top level of a Workflow file. The sandbox
imports a Workflow file again for each run, so a `DeepAgentsPlugin(...)` call
there runs again and replaces the model and tool activity options and the
streaming topic of the plugin you configured. The Workflow names the factory:

```python
from datetime import timedelta

from deepagents.backends import FilesystemBackend
from temporalio import workflow
from temporalio.deepagents import TemporalBackend, create_temporal_deep_agent
from temporalio.workflow import ActivityConfig


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
from datetime import timedelta

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from temporalio import activity, workflow
from temporalio.deepagents import activity_as_tool, create_temporal_deep_agent
from temporalio.exceptions import ApplicationError
from temporalio.workflow import ActivityConfig


@activity.defn
async def book_trip(city: str) -> str:
    """Book a trip to a city."""
    return f"Booked a trip to {city}."


book_trip_tool = activity_as_tool(
    book_trip,
    activity_options=ActivityConfig(start_to_close_timeout=timedelta(seconds=30)),
)


@workflow.defn
class ApprovalAgent:
    def __init__(self) -> None:
        self._pending: list[dict] = []
        self._decisions: list[str] | None = None

    @workflow.run
    async def run(self, request: str) -> str:
        agent = create_temporal_deep_agent(
            model="anthropic:claude-sonnet-5-5",
            tools=[book_trip_tool],
            interrupt_on={"book_trip": True},
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": workflow.info().workflow_id}}
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": request}]}, config=config
        )
        # One model turn can ask for several guarded calls, and the agent can
        # pause again after a resume.
        while result.get("__interrupt__"):
            self._pending = result["__interrupt__"][0].value["action_requests"]
            await workflow.wait_condition(lambda: self._decisions is not None)
            decisions = [{"type": decision} for decision in self._decisions]
            self._decisions = None
            result = await agent.ainvoke(
                Command(resume={"decisions": decisions}), config=config
            )
        return result["messages"][-1].content

    @workflow.query
    def pending_approvals(self) -> list[dict]:
        return self._pending

    @workflow.update
    async def resume(self, decisions: list[str]) -> None:
        if not self._pending or len(decisions) != len(self._pending):
            raise ApplicationError(
                f"Expected {len(self._pending)} decisions, got {len(decisions)}."
            )
        # With nothing pending, a second answer fails the check above.
        self._pending, self._decisions = [], decisions

    @resume.validator
    def check_resume(self, decisions: list[str]) -> None:
        if not set(decisions) <= {"approve", "reject"}:
            raise ValueError('Each decision must be "approve" or "reject".')
```

A client polls `pending_approvals` and shows the calls to a person. It then
calls the `resume` update with one `"approve"` or `"reject"` per call, in the
same order. LangChain needs exactly one decision for each guarded call of a
turn. Passed on to LangChain, a different number of decisions or any other
value would raise in the Workflow and fail the Workflow Task. The Update
refuses both: the validator checks the values, and the handler checks the
number against what is pending and fails the Update with an
`ApplicationError`. Register `book_trip` on the Worker with
`activities=[book_trip]`.

## Streaming

Set `streaming_topic=` on the plugin and a model call that LangChain streams
runs in a streaming Activity, which publishes chunk batches to a
`temporalio.contrib.workflow_streams` topic for live subscribers. The
aggregated final message still returns to the workflow, so the durable result
is identical to the non-streaming path:

```python
from temporalio.deepagents import DeepAgentsPlugin

plugin = DeepAgentsPlugin(streaming_topic="agent-stream")
```

No chunk reaches a subscriber unless the Workflow does two things:

- Host the stream: construct `WorkflowStream()` in `@workflow.init`.
- Drive the agent with `agent.astream(...)` and a `stream_mode` that includes
  `"messages"`: that makes LangChain call the model's streaming path.
  `agent.ainvoke(...)` and `astream` with other stream modes run the
  non-streaming Activity.

```python
from temporalio import workflow
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.deepagents import create_temporal_deep_agent


@workflow.defn
class StreamingAgent:
    @workflow.init
    def __init__(self, question: str) -> None:
        # Receives the chunks the model Activity publishes and serves them to
        # subscribers.
        self.stream = WorkflowStream()

    @workflow.run
    async def run(self, question: str) -> str:
        agent = create_temporal_deep_agent(model="anthropic:claude-sonnet-5-5")
        final = None
        async for mode, chunk in agent.astream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode=["messages", "values"],
        ):
            if mode == "values":
                final = chunk
        return final["messages"][-1].content
```

In the Workflow, read the final state from `"values"`. The `"messages"` items
do not arrive there: LangGraph hands them over with
`loop.call_soon_threadsafe`, which the Workflow event loop does not implement,
so LangChain logs `NotImplementedError` warnings from `StreamMessagesHandler`
and the tokens go to the topic only.

Subscribers read the topic with `WorkflowStreamClient`; each item is an
`AIMessageChunk` in `langchain_core.load.dumpd` form:

```python
from langchain_core.load import load
from temporalio.client import Client
from temporalio.contrib.workflow_streams import WorkflowStreamClient


async def print_tokens(client: Client, workflow_id: str) -> None:
    stream = WorkflowStreamClient.create(client, workflow_id)
    async for item in stream.subscribe("agent-stream"):
        chunk = load(item.data, allowed_objects="messages")
        print(chunk.text, end="", flush=True)
```

Streaming adds to History. Each published batch is a Signal with its own
Workflow Task, and each subscriber poll is an Update. With the default
`streaming_batch_interval` of 100 ms, a model call that streamed for 10 seconds
added about 400 Events, and about 900 with one subscriber. With
`streaming_batch_interval=timedelta(seconds=1)` it added about 60 and about
110. The chunks are the model's output as the provider sent it. They are
recorded in History with the Signals, as the full reply is with the Activity
result, and middleware that edits the reply in the Workflow, such as
`PIIMiddleware(apply_to_output=True)`, changes neither.

## Continue-as-new: what carries and what does not

Continue-as-new through `run_deep_agent` is experimental: when it triggers and
what the snapshot holds can change in a later release.

Long conversations bloat workflow history. `run_deep_agent(agent, input,
state_snapshot=...)` snapshots state and continues into a fresh run when the
turn ends with pending todos and the server recommends continuing
(`workflow.info().is_continue_as_new_suggested()`, the default and recommended
mode — it accounts for history length *and* size):

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

`run_deep_agent` has these limits:

- **It needs todos.** It reads `todos` from the turn's result. LangChain's
  `TodoListMiddleware` puts them there, and `create_deep_agent` on
  deepagents 0.7 does not add that middleware by default. The agent above
  therefore returns after each turn whatever the history size, and nothing is
  logged.
- **The continued run calls the model with no new message.** It invokes the
  agent on the carried transcript alone, so the first request ends with the
  model's own last answer. If the provider rejects a conversation that ends in
  an assistant message with an HTTP 400, the Workflow fails.
- **Summarization state is not carried.** The snapshot holds every raw message,
  not the summary Deep Agents was working from. A continued run of a
  summarized conversation therefore summarizes it again from the first
  message, in one model call. If the conversation no longer fits the model's
  context window, that call fails and so does the Workflow.
- **Not from an Update handler.** Temporal does not allow continue-as-new
  there. When `run_deep_agent` decides to continue inside a handler, the
  Workflow Task fails and the Update never completes.
- **A pending approval is dropped.** `run_deep_agent` does not look at
  `__interrupt__`. A turn that pauses for an approval with a todo still open
  continues as new once history is past the limit, and the new run tells the
  model that the tool call did not complete.

When a run does continue:

- **Carries forward:** the accumulated messages and the agent's state files
  (the graph's `files` state, where the default `StateBackend` keeps what
  `write_file` writes). Messages of the classes in `langchain_core.messages`
  come back in the next run as the same class (a `ToolMessage` stays a
  `ToolMessage`). Both ride in the continue-as-new input, so large files count
  toward its payload size limit; keep large files in a durable backend such as
  `TemporalBackend`. Repeated identical calls are
  NOT deduplicated: a call the agent re-issues after the boundary runs its own
  Activity, so a genuinely new identical request is never served a stale prior
  result. Your `@workflow.run` must accept `state_snapshot=None` as shown.
- **Does not carry forward:** todos as state. `TodoListMiddleware` marks
  `todos` as `OmitFromInput`, so the graph drops it from input and the next
  run starts with none; the agent's earlier `write_todos` calls are still in
  the carried messages. Nor does anything
  held only in an in-memory checkpointer's own structures beyond the messages
  and files snapshot. The default in-workflow
  `InMemorySaver` is rehydrated for free by deterministic replay; a durable
  checkpointer that does its own I/O is not replay-safe from inside a workflow,
  and the plugin warns if you pass one — prefer the snapshot + continue-as-new
  path above.

## Call configuration and tool results

`TemporalModel.ainvoke` and `TemporalModel.astream` forward the call's
`RunnableConfig` to the model Activity. Tools wrapped with `tool_as_activity`
forward it too. The payload keeps tags, run name and ID, recursion limit,
JSON-compatible metadata and non-internal `configurable` values. Callbacks,
live objects and `configurable` keys beginning with `__` stay in the Workflow.
Forwarded values are recorded in History; keep credentials in the Worker's
`model_provider` and tool implementation.

When called with a model's tool-call ID, a wrapped tool returns its complete
`ToolMessage`, including error status and artifact, with that same ID. Calling
it with bare arguments returns content, as LangChain does. An artifact must be
serializable by LangChain. `ToolRuntime` and `Command` are still unsupported.
New calls to a wrapped `return_direct=True` tool fail with
`DeepAgentsWorkflowError` before the Activity runs. Return from the Workflow
explicitly after calling the tool instead.

Runs recorded before `deepagents.wrapped-tool-call-context` retain the earlier
wrapped-tool behavior during replay and subsequent calls: content-only
results, generated tool-call IDs and empty configuration. New runs record
the patch and use the complete results and call configuration.

## Sandbox passthrough

The Workflow sandbox imports a module again for each Workflow run, and it
refuses some things a module does at import. During a run it refuses more, such
as reading the environment or the clock. A module that is passed through is
imported once, by the Worker process, and every run shares it. On top of the
SDK's own list, the plugin passes these modules through:

`langchain`, `langchain_core`, `langchain_anthropic`, `langchain_aws`,
`langchain_fireworks`, `langchain_openai`, `langgraph`, `deepagents`,
`langsmith`, `numpy`, `pydantic`, `pydantic_core`, `anthropic`, `tiktoken`,
`jsonpatch`, `jsonpointer`, `tenacity`, `orjson`, `httpx` and `httpcore`.

The list covers `deepagents`, LangChain, LangGraph, LangSmith and libraries
they import, so a Workflow file imports them without an import guard. Three
entries are provider packages that these load from Workflow code only if they
are installed: `deepagents` loads `langchain_aws` and `langchain_fireworks`
when it builds an agent, and LangChain loads `langchain_openai` for
provider-native structured output.

To pass another module through, name it in `passthrough_modules`, which adds to
the list:

```python
from temporalio.deepagents import DeepAgentsPlugin

plugin = DeepAgentsPlugin(passthrough_modules=["my_company.schemas"])
```

Do this when Workflow code imports a module that the sandbox refuses. The error
is `RestrictedWorkflowAccessError: Cannot access ... from inside a workflow`,
and it ends with "mark the import as pass through". When the import is at the
top of the Workflow file, creating the Worker fails with
`RuntimeError: Failed validating workflow`. When it happens during a run, the
Workflow Task fails and is retried, so the Workflow stays Running and makes no
progress. For an import in your own Workflow file,
`with workflow.unsafe.imports_passed_through():` around the import does the
same.

The sandbox passes an `importlib.import_module` call through only for a module
the Worker process has already imported. That is how `deepagents` and LangChain
load the three provider packages, so the plugin imports the installed ones when
a Worker or Replayer that uses the sandbox is created. If a library loads your
module that way, name it in `passthrough_modules` and also import it in the
Worker process before you create the Worker.

The sandbox does not restrict most of what a passed-through module does. Called
from a Workflow, its code can read the clock or open a socket. Pass a module
through only if Workflow code uses it deterministically.

Earlier versions did not pass `langchain_aws`, `langchain_fireworks` and
`langchain_openai` through, so with one of them installed the import failed
during a run with `RestrictedWorkflowAccessError`. A run that caught that error
and went on fails replay on this version with a `[TMPRL1100]` nondeterminism
error, because the model Activity it skipped is now scheduled. `deepagents`'
`RubricMiddleware` catches the error itself. On a Worker with `langchain_openai`
installed, its grader recorded `grader_error` instead of a verdict when it asked
for provider-native structured output, so that run fails replay too. Let such
runs finish on the earlier version, or keep them there with Worker Versioning.

## Runtime behavior

While a worker built with this plugin is running, the plugin wraps
`deepagents.create_deep_agent` so a bare `model="provider:name"` string is
auto-routed through an Activity. The wrapper only rewrites arguments when called
*inside a workflow*, so importing `deepagents` on a plain client or activity
worker is unaffected, and the original function is restored when the worker
stops. If you would rather be explicit, use `create_temporal_deep_agent` or
pass `TemporalModel("provider:name")` yourself.

Each plugin instance registers five Activities: `deepagents.invoke_model`,
`deepagents.invoke_model_streaming`, `deepagents.invoke_tool`,
`deepagents.backend_op` and `deepagents.get_model_profile`. Use one
`DeepAgentsPlugin` per Worker: one
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

Cancellation escaping `run_deep_agent` closes the Workflow as `CANCELED`.
Use `temporalio.exceptions.is_cancelled_exception` to recognize cancellation.
Other Temporal failures remain `DeepAgentsWorkflowError`. Histories recorded
before the `deepagents.raise-cancellation` patch retain the earlier failure
and cleanup behavior during replay.

These calls can pass Temporal's payload size limit (2 MiB per payload by
default):

- A model call. Its input holds every message of the request and grows with
  the conversation. Its result is the model's reply.
- A tool wrapped with `tool_as_activity`. Its input holds the arguments,
  including injected ones such as `InjectedState("messages")`. Its result is
  the tool's message.
- An Activity exposed with `activity_as_tool`. Its input holds the arguments of
  the model's tool call, and its result is what the Activity returns.
- A `TemporalBackend` operation. Its input holds what the agent writes or
  uploads, and its result what it reads, greps or downloads.
- The continue-as-new input of `run_deep_agent`, described above.

An input over the limit fails the Workflow Task and the run stays open. A
result over the limit fails the Activity attempt, and the retry runs the call
again.

## Errors and retries

A model call that fails with HTTP 408, 409, 429 or a 5xx status is retried,
after the provider's `retry-after` when it sends one. Any other HTTP status
fails the Activity without a retry. The provider's `x-should-retry` header
overrides both. An error with no HTTP status is left to the Activity's retry
policy. Examples are a missing Anthropic API key, a provider package that is
not installed, and an exception that a wrapped tool raises. Temporal's default
retry policy has no attempt limit, so these keep retrying and the Workflow
stays open: set `retry_policy` or `schedule_to_close_timeout` in the activity
options to bound them.

When an Activity fails for good, the Workflow fails. With `agent.ainvoke(...)`
the failure is the `ActivityError`. A context overflow that summarization
cannot recover from fails the Workflow with LangChain's `ContextOverflowError`.
`run_deep_agent` wraps both in `DeepAgentsWorkflowError`, whose
`ApplicationError.type` is `deepagents.DeepAgentsWorkflowError`. Passing a live
chat model instead of a model name or a `TemporalModel` fails the Workflow with
that type too. `create_temporal_deep_agent` with `activity_options` raises a
`ValueError` for it instead, which fails the Workflow Task.

Any other exception raised in Workflow code fails the Workflow Task, not the
Workflow, and Temporal retries the task until the code changes or the run is
terminated. LangGraph's `GraphRecursionError` and an exception from an
unwrapped tool are two examples. To fail the Workflow instead, add the
exception type to the Worker's `workflow_failure_exception_types`.

## Limits

- `tool_as_activity` does not support `return_direct=True`. New calls fail
  with `DeepAgentsWorkflowError` before the Activity runs. Return from the
  Workflow explicitly after calling the tool instead.
- A tool that takes `ToolRuntime` or returns a `Command` cannot be wrapped
  with `tool_as_activity`: the Workflow Task fails, for a `Command` after
  the tool has run.
- Tool `extras` are not sent to the model provider, for wrapped and unwrapped
  tools.
- `agent.astream(..., subgraphs=True)` fails the Workflow Task when the agent
  delegates to a sub-agent. Stream with `subgraphs=False`.
- A model you give to middleware, such as the fallback of
  `ModelFallbackMiddleware`, has to be a `TemporalModel`, as does the model of
  a `CompiledSubAgent`'s runnable. A model given there by name is called from
  Workflow code.

## Testing

`temporalio.deepagents.testing` has a scripted model, so a Workflow test needs
no provider and no API key:

```python
from langchain_core.messages import AIMessage
from temporalio.deepagents import DeepAgentsPlugin
from temporalio.deepagents.testing import mock_model_provider

call_weather = AIMessage(
    content="",
    tool_calls=[
        {"name": "get_weather", "args": {"city": "Paris"}, "id": "call-1"}
    ],
)
plugin = DeepAgentsPlugin(
    model_provider=mock_model_provider([call_weather, "It is sunny in Paris."])
)
```

Each model call takes the next reply, and the list starts over when it runs
out. A reply is a string or an `AIMessage`, so a script can include tool calls.
`get_model_profile` takes a reply too. The module also has `FakeModel`,
`fake_model_factory` and `MockTool`.

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

## Data converter

On a Client, the plugin replaces the SDK's default payload converter with a
Pydantic one and keeps your payload codec and failure converter. A `Replayer`
with the plugin gets the same converter. A Worker takes its converter from its
Client, so a plugin passed to the Worker alone does not change it. Passing a
`DataConverter` that sets a payload converter class of its own raises
`ValueError`: leave the class at its default and the plugin replaces it.

The converter is used for every payload of that Client, in Deep Agents
Workflows and in any other. It writes Pydantic values as the SDK's
`pydantic_data_converter` does, with the fields that came from a default
included, so a LangChain message keeps its `type` in Workflow inputs and
results, Signals, Updates and Queries. To get message objects on the receiving
side, use a Pydantic model with a `list[AnyMessage]` field as the parameter or
result type. A bare `list[AnyMessage]` annotation is not enough: the SDK reads
it without the discriminator that tells the message classes apart, so most
messages fail to decode. Under `dict` or `Any`, a message arrives as a dict of
its fields.

**Breaking change:** earlier versions left out the fields the sender had not
set. A message lost its `type`, and the receiver generated a `default_factory`
value again. The payload now carries those fields, and the receiver reads them
instead of filling in its own defaults. For your own models that means:

- A default that cannot be written as JSON, such as a function or a class, no
  longer encodes: sending the model raises `PydanticSerializationError`. In a
  Workflow this fails the Workflow Task, also when it replays a History
  recorded before the upgrade. A `SecretStr` default is written masked, so the
  receiver gets `**********`. Mark such a field `Field(exclude=True)`, and the
  receiver uses its own default as before.
- A default that does not pass the field's own validation, such as
  `name: str = None`, no longer decodes. Declare the field `str | None`. A
  float default of `inf` or `nan` fails the same way, because Pydantic writes
  it as `null`: set `ser_json_inf_nan="constants"` in the model's config.
- `model_fields_set` of a received model includes the fields that came from a
  default, so `model_dump(exclude_unset=True)` on it returns them too.
- A model with `extra="forbid"` rejects a payload whose sender has a field the
  model lacks, even when the sender left that field unset. Add the field to
  the receiving model first.

Payloads grow by the fields that were left out. A LangChain message that your
own code sends, or that is in a wrapped tool's arguments, takes up to about
200 bytes more, depending on its class and on how many of its fields were set.
Model Activity payloads do not change. Histories recorded before the upgrade
replay as before, unless the Workflow sends a model with a default that cannot
be written as JSON. Earlier versions of the plugin read the new payloads, so
Clients and Workers can be upgraded and rolled back in any order. A payload
keeps the shape its writer gave it, also in History. A Client on an earlier
version still starts Workflows and sends Signals and Updates without those
fields, so upgrade the Clients as well as the Workers.
