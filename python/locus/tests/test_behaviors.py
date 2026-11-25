"""Replay, retries, streaming, sync tools and waiting on a person."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest
from locus.agent import Agent
from locus.core.events import ModelChunkEvent
from locus.core.messages import Message, Role, ToolCall
from locus.models.base import ModelResponse
from locus.tools.decorator import tool

from temporalio import activity, workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.locus import LocusPlugin, TemporalModel, activity_as_tool
from temporalio.worker import Replayer, Worker

MODEL_TIMEOUT = timedelta(seconds=30)


class HTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class ScriptedModel:
    """Calls ``tool_name`` once, then answers with the tool's result.

    ``failures`` are raised first, one per call.
    """

    def __init__(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        failures: list[Exception] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.failures = list(failures or [])
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        del tools, kwargs
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        results = [m for m in messages if m.role == Role.TOOL]
        if not results:
            call = ToolCall(id="call_1", name=self.tool_name, arguments=self.arguments)
            return ModelResponse(
                message=Message.assistant(content=None, tool_calls=[call]),
                stop_reason="tool_calls",
            )
        return ModelResponse(
            message=Message.assistant(content=f"Done: {results[-1].content}"),
            stop_reason="stop",
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        response = await self.complete(messages, tools, **kwargs)
        text = response.message.content or ""
        for word in text.split(" ") if text else []:
            yield ModelChunkEvent(content=word + " ")
        yield ModelChunkEvent(tool_calls=response.message.tool_calls or None, done=True)


@activity.defn
async def refund(order_id: str, amount_usd: float = 10.0) -> str:
    """Refund an order."""
    return f"refunded {order_id} ${amount_usd:.2f}"


def _client(client: Client, model: Any) -> Client:
    config = client.config()
    config["plugins"] = [LocusPlugin(models={"m": lambda: model})]
    return Client(**config)


def _model(**options: Any) -> TemporalModel:
    return TemporalModel("m", start_to_close_timeout=MODEL_TIMEOUT, **options)


@workflow.defn
class RefundWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            model=_model(
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(milliseconds=50), maximum_attempts=3
                )
            ),
            tools=[activity_as_tool(refund, start_to_close_timeout=MODEL_TIMEOUT)],
        )
        return (await agent.arun(prompt)).message or ""


async def _activity_types(handle: Any) -> list[str]:
    return [
        event.activity_task_scheduled_event_attributes.activity_type.name
        async for event in handle.fetch_history_events()
        if event.HasField("activity_task_scheduled_event_attributes")
    ]


async def test_run_replays_on_every_task_and_from_history(client: Client) -> None:
    """With no workflow cache each task replays the loop; the history replays too."""
    model = ScriptedModel("refund", {"order_id": "4821"})
    client = _client(client, model)
    task_queue = f"locus-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[RefundWorkflow],
        activities=[refund],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            RefundWorkflow.run,
            "refund order 4821",
            id=f"refund-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        assert await handle.result() == "Done: refunded 4821 $10.00"
    # The model and the tool each ran once, although the loop ran several times.
    assert model.calls == 2
    assert await _activity_types(handle) == [
        "locus_model_complete",
        "refund",
        "locus_model_complete",
    ]
    replayer = Replayer(
        workflows=[RefundWorkflow], plugins=[LocusPlugin(models={"m": lambda: model})]
    )
    await replayer.replay_workflow(await handle.fetch_history())


async def test_transient_model_error_is_retried_by_temporal(client: Client) -> None:
    model = ScriptedModel("refund", {"order_id": "1"}, failures=[HTTPError(503)])
    client = _client(client, model)
    task_queue = f"locus-{uuid.uuid4()}"
    async with Worker(
        client, task_queue=task_queue, workflows=[RefundWorkflow], activities=[refund]
    ):
        handle = await client.start_workflow(
            RefundWorkflow.run,
            "refund order 1",
            id=f"refund-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        assert await handle.result() == "Done: refunded 1 $10.00"
    # Three model calls on the worker, but two model activities: Temporal
    # retried the failed attempt inside the first one.
    assert model.calls == 3
    assert (await _activity_types(handle)).count("locus_model_complete") == 2


async def test_final_model_error_fails_the_run_without_retry(client: Client) -> None:
    model = ScriptedModel("refund", {"order_id": "1"}, failures=[HTTPError(400)])
    client = _client(client, model)
    task_queue = f"locus-{uuid.uuid4()}"
    async with Worker(
        client, task_queue=task_queue, workflows=[RefundWorkflow], activities=[refund]
    ):
        with pytest.raises(WorkflowFailureError) as raised:
            await client.execute_workflow(
                RefundWorkflow.run,
                "refund order 1",
                id=f"refund-{uuid.uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=30),
            )
    assert model.calls == 1
    cause = raised.value.cause
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.type == "Locus.format_error"


@workflow.defn
class StreamingWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[str]:
        agent = Agent(
            model=_model(),
            tools=[activity_as_tool(refund, start_to_close_timeout=MODEL_TIMEOUT)],
        )
        chunks: list[str] = []
        async for event in agent.run(prompt, stream_tokens=True):
            if isinstance(event, ModelChunkEvent) and event.content:
                chunks.append(event.content)
        return chunks


async def test_streamed_run_yields_the_models_chunks(client: Client) -> None:
    model = ScriptedModel("refund", {"order_id": "7", "amount_usd": 2.5})
    client = _client(client, model)
    task_queue = f"locus-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StreamingWorkflow],
        activities=[refund],
    ):
        handle = await client.start_workflow(
            StreamingWorkflow.run,
            "refund order 7",
            id=f"stream-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        chunks = await handle.result()
    assert "".join(chunks).strip() == "Done: refunded 7 $2.50"
    assert (await _activity_types(handle)).count("locus_model_stream") == 2


@workflow.defn
class ApprovalWorkflow:
    """A tool that waits, for as long as it takes, for a person's signal."""

    def __init__(self) -> None:
        self.approved: bool | None = None
        self.waiting_on: str | None = None

    @workflow.signal
    def decide(self, approved: bool) -> None:
        self.approved = approved

    @workflow.query
    def pending(self) -> str | None:
        return self.waiting_on

    @workflow.run
    async def run(self, prompt: str) -> str:
        @tool
        async def refund_with_approval(order_id: str) -> str:
            """Refund an order once a person approves."""
            self.waiting_on = order_id
            await workflow.wait_condition(lambda: self.approved is not None)
            self.waiting_on = None
            if not self.approved:
                return "refused by reviewer"
            return await workflow.execute_activity(
                refund, args=[order_id, 10.0], start_to_close_timeout=MODEL_TIMEOUT
            )

        agent = Agent(model=_model(), tools=[refund_with_approval])
        return (await agent.arun(prompt)).message or ""


async def test_tool_waits_for_a_signal_across_a_worker_restart(client: Client) -> None:
    model = ScriptedModel("refund_with_approval", {"order_id": "4821"})
    client = _client(client, model)
    task_queue = f"locus-{uuid.uuid4()}"
    workflow_id = f"approval-{uuid.uuid4()}"

    def worker() -> Worker:
        return Worker(
            client,
            task_queue=task_queue,
            workflows=[ApprovalWorkflow],
            activities=[refund],
        )

    async with worker():
        handle = await client.start_workflow(
            ApprovalWorkflow.run,
            "refund order 4821",
            id=workflow_id,
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        for _ in range(100):
            if await handle.query(ApprovalWorkflow.pending) == "4821":
                break
        else:
            raise AssertionError("the tool never started waiting")
    # The first worker is gone. A new one rebuilds the agent's loop from History
    # and finishes the run once the person decides.
    async with worker():
        await handle.signal(ApprovalWorkflow.decide, True)
        assert await handle.result() == "Done: refunded 4821 $10.00"
    # The model call before the wait was not made again.
    assert model.calls == 2


@workflow.defn
class SyncToolWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        @tool
        def lookup(order_id: str) -> str:
            """Look an order up."""
            return f"order {order_id}"

        agent = Agent(model=_model(), tools=[lookup])
        return (await agent.arun(prompt)).message or ""


async def test_sync_tool_runs_inline_in_the_workflow(client: Client) -> None:
    model = ScriptedModel("lookup", {"order_id": "9"})
    client = _client(client, model)
    task_queue = f"locus-{uuid.uuid4()}"
    async with Worker(client, task_queue=task_queue, workflows=[SyncToolWorkflow]):
        result = await client.execute_workflow(
            SyncToolWorkflow.run,
            "look up order 9",
            id=f"sync-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )
    assert result == "Done: order 9"


@activity.defn
async def broken(order_id: str) -> str:
    """Always fails."""
    raise ApplicationError(f"no such order {order_id}", non_retryable=True)


@workflow.defn
class BrokenToolWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            model=_model(),
            tools=[activity_as_tool(broken, start_to_close_timeout=MODEL_TIMEOUT)],
        )
        return (await agent.arun(prompt)).message or ""


async def test_failed_tool_activity_reaches_the_model_as_a_tool_error(
    client: Client,
) -> None:
    model = ScriptedModel("broken", {"order_id": "404"})
    client = _client(client, model)
    task_queue = f"locus-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[BrokenToolWorkflow],
        activities=[broken],
    ):
        result = await client.execute_workflow(
            BrokenToolWorkflow.run,
            "refund order 404",
            id=f"broken-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )
    assert result.startswith("Done: ")
    assert "Error" in result
