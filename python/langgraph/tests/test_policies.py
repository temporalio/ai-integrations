"""Tests for LangGraph timeout and cache policies while a graph runs."""

import asyncio
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
from langgraph.cache.memory import (  # pyright: ignore[reportMissingTypeStubs]
    InMemoryCache,
)
from langgraph.errors import NodeTimeoutError
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    StateGraph,
)
from langgraph.types import CachePolicy, Command, Send, TimeoutPolicy
from pytest import raises
from typing_extensions import TypedDict

from temporalio import workflow
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowFailureError, WorkflowHandle
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
from temporalio.langgraph import LangGraphPlugin, graph
from temporalio.worker import Replayer, Worker

_ACTIVITY_OPTIONS = {
    "start_to_close_timeout": timedelta(seconds=10),
    "retry_policy": RetryPolicy(maximum_attempts=1),
}


class State(TypedDict):
    value: str


async def worker(state: State) -> dict[str, str]:  # pyright: ignore[reportUnusedParameter]
    return {"value": "done"}


async def never_finishes(state: State) -> dict[str, str]:  # pyright: ignore[reportUnusedParameter]
    # Wait (forever) until worker shutdown cancellation
    await asyncio.Event().wait()
    return {"value": "done"}


async def route_with_idle_timeout(state: State) -> list[Send]:
    return [Send("worker", state, timeout=TimeoutPolicy(idle_timeout=30))]


async def dispatch_with_idle_timeout(state: State) -> Command[Any]:
    return Command(goto=Send("worker", state, timeout=TimeoutPolicy(idle_timeout=30)))


async def route_with_run_timeout(state: State) -> list[Send]:
    return [Send("worker", state, timeout=0.5)]


async def _scheduled_activities(handle: WorkflowHandle[Any, Any]) -> list[str]:
    return [
        event.activity_task_scheduled_event_attributes.activity_type.name
        async for event in handle.fetch_history_events()
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ]


@workflow.defn
class PolicyWorkflow:
    @workflow.run
    async def run(self) -> Any:
        return await graph("my-graph").compile().ainvoke({"value": ""})


@pytest.mark.parametrize(
    ("send_from", "worker_runs_in"),
    [
        ("router", "activity"),
        ("router", "workflow"),
        ("command", "activity"),
    ],
)
async def test_send_idle_timeout_fails_workflow(
    client: Client, send_from: str, worker_runs_in: str
):
    """A Send is made while the graph runs, so the plugin refuses its idle timeout there."""
    g = StateGraph(State)
    g.add_node("worker", worker, metadata={"execute_in": worker_runs_in})
    g.add_edge("worker", END)
    if send_from == "router":
        g.add_conditional_edges(START, route_with_idle_timeout, ["worker"])
    else:
        g.add_node(
            "dispatch", dispatch_with_idle_timeout, metadata={"execute_in": "activity"}
        )
        g.add_edge(START, "dispatch")

    task_queue = f"send-idle-timeout-{uuid4()}"
    plugin = LangGraphPlugin(
        graphs={"my-graph": g}, default_activity_options=_ACTIVITY_OPTIONS
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[PolicyWorkflow],
        plugins=[plugin],
    ):
        handle = await client.start_workflow(
            PolicyWorkflow.run,
            id=f"test-send-idle-timeout-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )
        with raises(WorkflowFailureError) as err:
            await handle.result()

    cause = err.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.non_retryable
    assert "Node 'worker' runs under a LangGraph idle_timeout" in cause.message
    assert "my-graph.worker" not in await _scheduled_activities(handle)
    await Replayer(workflows=[PolicyWorkflow], plugins=[plugin]).replay_workflow(
        await handle.fetch_history()
    )


@workflow.defn
class RunTimeoutWorkflow:
    @workflow.run
    async def run(self) -> str:
        try:
            await graph("my-graph").compile().ainvoke({"value": ""})
        except NodeTimeoutError as err:
            return f"timed out ({err.kind})"
        return "finished"


async def test_send_run_timeout_cancels_node_and_replays(client: Client):
    """LangGraph times a run_timeout with a Workflow timer, so it stays usable."""
    g = StateGraph(State)
    g.add_node("worker", never_finishes, metadata={"execute_in": "activity"})
    g.add_conditional_edges(START, route_with_run_timeout, ["worker"])
    g.add_edge("worker", END)

    task_queue = f"send-run-timeout-{uuid4()}"
    plugin = LangGraphPlugin(
        graphs={"my-graph": g}, default_activity_options=_ACTIVITY_OPTIONS
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[RunTimeoutWorkflow],
        plugins=[plugin],
    ):
        handle = await client.start_workflow(
            RunTimeoutWorkflow.run,
            id=f"test-send-run-timeout-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )
        result = await handle.result()

    assert result == "timed out (run)"
    history = await handle.fetch_history()
    assert EventType.EVENT_TYPE_ACTIVITY_TASK_CANCEL_REQUESTED in [
        event.event_type for event in history.events
    ]
    await Replayer(workflows=[RunTimeoutWorkflow], plugins=[plugin]).replay_workflow(
        history
    )


class PriceState(TypedDict):
    sku: str
    request: str
    price: str


async def price(state: PriceState) -> dict[str, str]:
    return {"price": f"{state['sku']} for {state['request']}"}


def by_sku(state: PriceState) -> str:
    return state["sku"]


@workflow.defn
class CachedGraphWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        app = graph("my-graph").compile(cache=InMemoryCache())
        first = await app.ainvoke({"sku": "a", "request": "first", "price": ""})
        second = await app.ainvoke({"sku": "a", "request": "second", "price": ""})
        return [first["price"], second["price"]]


async def test_cache_policy_without_ttl_replays(client: Client):
    """A LangGraph cache made in the Workflow belongs to the run, so it replays."""
    g = StateGraph(PriceState)
    g.add_node(
        "price",
        price,
        metadata={"execute_in": "activity"},
        cache_policy=CachePolicy(key_func=by_sku),
    )
    g.add_edge(START, "price")

    task_queue = f"cache-policy-{uuid4()}"
    plugin = LangGraphPlugin(
        graphs={"my-graph": g}, default_activity_options=_ACTIVITY_OPTIONS
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[CachedGraphWorkflow],
        plugins=[plugin],
    ):
        handle = await client.start_workflow(
            CachedGraphWorkflow.run,
            id=f"test-cache-policy-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )
        result = await handle.result()

    # The second call has the first one's cache key, so LangGraph answers it.
    assert result == ["a for first", "a for first"]
    assert await _scheduled_activities(handle) == ["my-graph.price"]
    await Replayer(workflows=[CachedGraphWorkflow], plugins=[plugin]).replay_workflow(
        await handle.fetch_history()
    )
