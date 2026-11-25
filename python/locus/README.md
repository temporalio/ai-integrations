# Locus

> Release stage: Pre-release.

This Temporal [Plugin](https://docs.temporal.io/develop/plugins-guide) runs [Locus](https://locusagents.ai/) agents inside Temporal Workflows. The agent's loop runs in the Workflow; every model call and every tool you register with `activity_as_tool` runs as a Temporal Activity, so it is retried by Temporal, recorded in Event History, and not repeated when a Worker restarts.

Published as [`temporalio-locus`](https://pypi.org/project/temporalio-locus/) and imported as `temporalio.locus`.

## Install

```bash
uv add temporalio-locus
```

Add the Locus extra for your model provider, for example `locus-sdk[oci]` for OCI Generative AI, and `temporalio-locus[mcp]` to use MCP servers.

## Quickstart

```python
import asyncio
from datetime import timedelta

from locus.agent import Agent
from locus.models import get_model

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.locus import LocusPlugin, TemporalModel, activity_as_tool
from temporalio.worker import Worker


@activity.defn
async def lookup_order(order_id: str) -> str:
    """Return the current status of an order."""
    return f"order {order_id}: shipped"


@workflow.defn
class SupportWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            model=TemporalModel("default", start_to_close_timeout=timedelta(seconds=60)),
            tools=[
                activity_as_tool(lookup_order, start_to_close_timeout=timedelta(seconds=30))
            ],
            system_prompt="Answer support questions using the order lookup.",
        )
        result = await agent.arun(prompt)
        return result.message or ""


async def main() -> None:
    plugin = LocusPlugin(models={"default": lambda: get_model("oci:openai.gpt-5.5")})
    client = await Client.connect("localhost:7233", plugins=[plugin])
    async with Worker(
        client,
        task_queue="locus",
        workflows=[SupportWorkflow],
        activities=[lookup_order],
    ):
        print(
            await client.execute_workflow(
                SupportWorkflow.run,
                "What happened to order ORD-7842?",
                id="support-1",
                task_queue="locus",
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
```

Build the `Agent` and its `TemporalModel` in Workflow code. Build the real model in a `LocusPlugin` factory: it runs on the Worker, so credentials never enter Event History.

## What runs where

| Locus | In a Workflow | Runs as |
| --- | --- | --- |
| Model call | `Agent(model=TemporalModel(name, ...))` | `locus_model_complete` Activity |
| Streamed model call | `agent.run(prompt, stream_tokens=True)` | `locus_model_stream` Activity; the Workflow gets the chunks when the call ends, and subscribers get them live with `streaming_topic` |
| Tool that does I/O | `activity_as_tool(fn, ...)` | the Activity `fn` |
| MCP server's tools | `Agent(mcp_servers=[TemporalMCPClient(name, ...)])` | `temporalio.mcp.{name}.list-tools` and `temporalio.mcp.{name}.call-tool` Activities |
| `@tool` function, sync or async | passed to `Agent(tools=[...])` | Workflow code; must be deterministic |

## Models

`LocusPlugin(models=...)` maps a name to a factory returning a Locus model. A factory is called on first use and the model is kept for the Worker's lifetime. `TemporalModel(name, ...)` selects the factory and carries the Activity options of that agent's model calls. `TemporalModel` needs `start_to_close_timeout` or `schedule_to_close_timeout`. `heartbeat_timeout` defaults to 10 seconds, and the model Activities heartbeat at half of it.

## Retries

Temporal retries a failed model call with the Activity's `retry_policy`. An error Locus classifies as final (authentication, billing, a malformed request, a context overflow) fails the Activity without a retry, with type `Locus.<reason>`. Locus's own model retry (`model_retry`) sees an Activity failure as unclassified and does not retry it again.

A failed tool Activity is handed to the model as a tool error, as Locus does for any tool that raises.

## Cancellation

Cancelling the Workflow cancels a running model call: the heartbeat delivers the cancel to the Activity, and the model's `complete` or `stream` is cancelled on the Worker. This needs a `heartbeat_timeout`, which `TemporalModel` sets by default.

## MCP

An MCP server's tools run as Activities. Register a factory for each server on the Worker, and refer to it by name in Workflow code with `TemporalMCPClient`. This needs the `mcp` extra, `temporalio-locus[mcp]`, which brings [`temporalio-mcp`](https://github.com/temporalio/ai-integrations/tree/main/python/mcp) and MCP Python SDK v2.

```python
# worker
from mcp import Client, StdioServerParameters, stdio_client

plugin = LocusPlugin(
    models={"default": lambda: get_model("oci:openai.gpt-5.5")},
    mcp_clients={
        "orders": lambda: Client(
            stdio_client(StdioServerParameters(command="orders-mcp-server"))
        )
    },
)

# workflow
orders = TemporalMCPClient("orders", start_to_close_timeout=timedelta(seconds=30))
agent = Agent(model=TemporalModel("default", ...), mcp_servers=[orders])
```

- The factory returns an `mcp.Client`, not Locus's `MCPClient`: the connection lives on the Worker, so its transport and credentials never enter Event History. Locus's own `MCPClient` connects from where it is built and is not supported in Workflow code.
- Listing the server's tools is the `temporalio.mcp.{name}.list-tools` Activity, and each tool call is a `temporalio.mcp.{name}.call-tool` Activity whose summary is the tool's name. The listing runs once per `TemporalMCPClient`; with `mcp_servers` that is on the agent's first run.
- `await orders.load_tools()` returns the tools as Locus tools, to pass in `Agent(tools=[...])` beside your own.
- `prefix="shop"` names the tools `shop_{name}` for the model; `allowed_tools=[...]` keeps only the named tools.
- A tool the server reports as failed is a tool error for the model, and its Activity succeeds. An Activity that fails, such as one whose server is down, is retried with `retry_policy`; by default without limit.
- Text, `structuredContent`, images and other content blocks reach Locus as they do from its `MCPClient`.
- Per-run headers (`headers_provider`), progress notifications and tool filters that are functions are not carried.

## Multi-agent shapes

Locus's shapes run in Workflow code as they are. Give every agent a `TemporalModel`; each agent's model calls are then Activities, and the shape's own control flow replays with the Workflow.

```python
pipeline = SequentialPipeline(agents=[researcher, writer])
result = await pipeline.run("Summarise the incident.")
```

Tested: `SequentialPipeline`, `ParallelPipeline` (its agents' model Activities run concurrently), `StateGraph` with agent nodes, `Orchestrator` with specialists, `Swarm` and a `Handoff` chain.

## Streaming

The Workflow receives a streamed call's chunks together, when the model Activity returns, so that replay sees the same chunks. To show them to a user while the model is still answering, pass `streaming_topic` to `TemporalModel` and host a [`WorkflowStream`](https://docs.temporal.io/workflow-streams) on the Workflow. The model Activity then also publishes each `ModelChunkEvent` on that topic as the model produces it, batched on `streaming_batch_interval` (default 100 ms), and clients subscribe with `WorkflowStreamClient`.

```python
# workflow
@workflow.defn
class ChatWorkflow:
    def __init__(self) -> None:
        self.stream = WorkflowStream()

    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            model=TemporalModel(
                "default",
                start_to_close_timeout=timedelta(seconds=60),
                streaming_topic="tokens",
            )
        )
        text = ""
        async for event in agent.run(prompt, stream_tokens=True):
            if isinstance(event, ModelChunkEvent) and event.content:
                text += event.content
        return text

# client
stream = WorkflowStreamClient.create(client, workflow_id)
async for item in stream.subscribe(["tokens"], result_type=ModelChunkEvent):
    print(item.data.content or "", end="", flush=True)
```

- Locus streams a model call only in a streamed run, `agent.run(prompt, stream_tokens=True)`. `agent.arun(...)` makes unstreamed calls and publishes nothing.
- Without a `WorkflowStream` on the Workflow, published chunks are dropped and the run is unaffected.
- Publishing is a side effect of an Activity attempt: a failed attempt may leave chunks on the topic before the retry publishes its own.
- Workflow Streams is at an experimental release stage in the Python SDK.

## Waiting on a person

A Workflow can wait on a Signal or Update for as long as it takes, across Worker restarts. Locus's own pauses fit that: the run stops, the Workflow waits, and the answer resumes the run.

`ask_user`, which an agent has with `completion_mode="explicit"`, ends the run with an `InterruptEvent`. Wait for the answer and pass it to `agent.resume`:

```python
events = agent.run(prompt)
while True:
    question = None
    async for event in events:
        if isinstance(event, InterruptEvent):
            question = event.question
    if question is None:
        break
    self.question = question  # expose it with a Query
    await workflow.wait_condition(lambda: self.answer is not None)
    events = agent.resume(self.answer)
```

`interrupt()` in a `StateGraph` node returns a result with `interrupt` set. Resume with the paused state; no checkpointer is needed, because the Workflow holds it:

```python
paused = await graph.execute(state)
if paused.interrupt is not None:
    await workflow.wait_condition(lambda: self.answer is not None)
    result = await graph.execute(Command(resume=self.answer, update=paused.final_state))
```

`admit()` raises `AdmissionError` for an action that needs a person. Wait for the approver and call it again with `approved_by`; the side effect in `perform` can be an Activity. Give an `AuditTrail` the Workflow's clock, so a replayed run rebuilds the same hash chain:

```python
trail = AuditTrail(clock=lambda: workflow.now().isoformat())
try:
    await admit(action, perform, policy=policy, trail=trail)
except AdmissionError:
    await workflow.wait_condition(lambda: self.approver is not None)
    await admit(action, perform, policy=policy, trail=trail, approved_by=self.approver)
```

A tool that runs in the Workflow can also wait directly:

```python
@tool
async def refund_with_approval(order_id: str) -> str:
    """Refund an order once a person approves."""
    await workflow.wait_condition(lambda: self.approved is not None)
    ...
```

## Time, ids and randomness

Inside a Workflow, Locus's time budgets, `TimeLimit`, timestamps, run ids, tool-call ids and retry jitter use the Workflow's clock, `workflow.uuid4()` and `workflow.random()`, so a replayed run takes the same path. Activities and plain Locus code in the same process keep the real clock.

Where Locus has `locus.core.sources`, the plugin installs its sources through it. Older Locus releases have no such seam; there the plugin replaces the `time`, `datetime`, `uuid4` and `random` names in the Locus modules the loop and the shapes run (`locus.core.state`, `locus.core.events`, `locus.core.interrupt`, `locus.core.messages`, `locus.core.termination`, `locus.agent.agent`, `locus.agent.runtime_loop`, `locus.agent.model_retry`, `locus.tools.executor` and the `locus.multiagent` modules). On those releases two reads stay on the system's clock and random generator, because Locus imports them inside the function: a `StateGraph` node cache's `ttl_seconds` check and the jitter of a node's `RetryPolicy`. Avoid a node cache with a ttl in Workflow code there.

## Not supported in Workflow code

- `agent.run_sync(...)`: it starts a thread. Use `await agent.arun(...)` or `agent.run(...)`.
- Checkpointers, memory managers, Locus's `MCPClient`, RAG stores and other Locus features that do their own I/O. Event History is the state store; call such features from an Activity. For MCP use `TemporalMCPClient`.
- Model call options that cannot be sent as JSON, such as a callback. The run fails with `LocusUnserializableModelOption`; set such an option on the model the factory builds.

## Payload size

A model Activity's input holds the whole conversation and its result holds the model's reply, so a long conversation or a large tool result can pass Temporal's payload size limit.

## Develop

```bash
make sync   # install (non-editable) into .venv
make lint
make test
```

Repository conventions: [`AGENTS.md`](https://github.com/temporalio/ai-integrations/blob/main/AGENTS.md).
