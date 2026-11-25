# Locus

> Release stage: Pre-release.

This Temporal [Plugin](https://docs.temporal.io/develop/plugins-guide) runs [Locus](https://locusagents.ai/) agents inside Temporal Workflows. The agent's loop runs in the Workflow; every model call and every tool you register with `activity_as_tool` runs as a Temporal Activity, so it is retried by Temporal, recorded in Event History, and not repeated when a Worker restarts.

Published as [`temporalio-locus`](https://pypi.org/project/temporalio-locus/) and imported as `temporalio.locus`.

## Install

```bash
uv add temporalio-locus
```

Add the Locus extra for your model provider, for example `locus-sdk[oci]` for OCI Generative AI.

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
| Streamed model call | `agent.run(prompt, stream_tokens=True)` | `locus_model_stream` Activity; chunks arrive together when the call ends |
| Tool that does I/O | `activity_as_tool(fn, ...)` | the Activity `fn` |
| `@tool` function, sync or async | passed to `Agent(tools=[...])` | Workflow code; must be deterministic |

## Models

`LocusPlugin(models=...)` maps a name to a factory returning a Locus model. A factory is called on first use and the model is kept for the Worker's lifetime. `TemporalModel(name, ...)` selects the factory and carries the Activity options of that agent's model calls. `TemporalModel` needs `start_to_close_timeout` or `schedule_to_close_timeout`. `heartbeat_timeout` defaults to 10 seconds, and the model Activities heartbeat at half of it.

## Retries

Temporal retries a failed model call with the Activity's `retry_policy`. An error Locus classifies as final (authentication, billing, a malformed request, a context overflow) fails the Activity without a retry, with type `Locus.<reason>`. Locus's own model retry (`model_retry`) sees an Activity failure as unclassified and does not retry it again.

A failed tool Activity is handed to the model as a tool error, as Locus does for any tool that raises.

## Cancellation

Cancelling the Workflow cancels a running model call: the heartbeat delivers the cancel to the Activity, and the model's `complete` or `stream` is cancelled on the Worker. This needs a `heartbeat_timeout`, which `TemporalModel` sets by default.

## Waiting on a person

A tool that runs in the Workflow can wait on a Signal or Update for as long as it takes, across Worker restarts, and then run an Activity:

```python
@tool
async def refund_with_approval(order_id: str) -> str:
    """Refund an order once a person approves."""
    await workflow.wait_condition(lambda: self.approved is not None)
    ...
```

## Time, ids and randomness

Inside a Workflow, Locus's time budgets, `TimeLimit`, timestamps, run ids, tool-call ids and retry jitter use the Workflow's clock, `workflow.uuid4()` and `workflow.random()`, so a replayed run takes the same path. Locus has no clock or id seam, so the plugin replaces the `time`, `datetime`, `uuid4` and `random` names in these Locus modules: `locus.core.state`, `locus.core.events`, `locus.core.messages`, `locus.core.termination`, `locus.agent.agent`, `locus.agent.runtime_loop`, `locus.agent.model_retry` and `locus.tools.executor`. Activities and plain Locus code in the same process keep the real clock.

## Not supported in Workflow code

- `agent.run_sync(...)`: it starts a thread. Use `await agent.arun(...)` or `agent.run(...)`.
- Checkpointers, memory managers, MCP servers, RAG stores and other Locus features that do their own I/O. Event History is the state store; call such features from an Activity.
- Model call options that cannot be sent as JSON, such as a callback. The run fails with `LocusUnserializableModelOption`; set such an option on the model the factory builds.
- Live token streaming to a client. Streamed chunks reach the Workflow when the model call ends.

## Payload size

A model Activity's input holds the whole conversation and its result holds the model's reply, so a long conversation or a large tool result can pass Temporal's payload size limit.

## Develop

```bash
make sync   # install (non-editable) into .venv
make lint
make test
```

Repository conventions: [`AGENTS.md`](https://github.com/temporalio/ai-integrations/blob/main/AGENTS.md).
