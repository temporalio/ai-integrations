"""Sync routing functions run in the Workflow, as async ones do.

LangGraph runs a sync function given to ``add_conditional_edges`` with
LangChain's ``run_in_executor`` helper. The Workflow event loop has no
threads, so before the plugin ran those calls inline, every Workflow Task
failed with ``NotImplementedError`` and the run never finished.
"""

import asyncio
import operator
import warnings
from datetime import timedelta
from typing import Annotated, Any
from uuid import uuid4

import pytest
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    StateGraph,
)
from langgraph.types import Send
from typing_extensions import TypedDict

from temporalio import workflow
from temporalio.client import Client
from temporalio.langgraph import LangGraphPlugin, graph
from temporalio.worker import Replayer, Worker


class State(TypedDict):
    count: int


def increment(state: State) -> dict[str, int]:
    return {"count": state["count"] + 1}


def route(state: State) -> str:
    # A plain def, not async def.
    return "increment" if state["count"] < 3 else END


class FanOutState(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]


class ItemState(TypedDict):
    item: str


def upper(state: ItemState) -> dict[str, list[str]]:
    return {"results": [state["item"].upper()]}


def fan_out(state: FanOutState) -> list[Send]:
    return [Send("upper", {"item": item}) for item in state["items"]]


@workflow.defn
class SyncRouterWorkflow:
    @workflow.run
    async def run(self, state: dict[str, Any]) -> Any:
        return await graph("sync-router").compile().ainvoke(state)


async def run_and_replay(
    client: Client, g: StateGraph[Any, Any, Any, Any], state: dict[str, Any]
) -> Any:
    task_queue = f"sync-router-{uuid4()}"
    plugin = LangGraphPlugin(
        graphs={"sync-router": g},
        default_activity_options={"start_to_close_timeout": timedelta(seconds=10)},
    )
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[SyncRouterWorkflow],
        plugins=[plugin],
    ):
        handle = await client.start_workflow(
            SyncRouterWorkflow.run,
            state,
            id=f"sync-router-{uuid4()}",
            task_queue=task_queue,
            # A failed Workflow Task retries without end, so bound the wait.
            execution_timeout=timedelta(seconds=10),
        )
        result = await handle.result()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Module .* was imported after initial workflow load"
        )
        await Replayer(
            workflows=[SyncRouterWorkflow], plugins=[plugin]
        ).replay_workflow(await handle.fetch_history())
    return result


@pytest.mark.parametrize("execute_in", ["activity", "workflow"])
async def test_sync_router(client: Client, execute_in: str):
    g = StateGraph(State)
    g.add_node("increment", increment, metadata={"execute_in": execute_in})
    g.add_edge(START, "increment")
    g.add_conditional_edges("increment", route)

    assert await run_and_replay(client, g, {"count": 0}) == {"count": 3}


async def test_sync_router_returning_send(client: Client):
    g = StateGraph(FanOutState)
    g.add_node("upper", upper, metadata={"execute_in": "activity"})
    g.add_conditional_edges(START, fan_out, ["upper"])
    g.add_edge("upper", END)

    result = await run_and_replay(client, g, {"items": ["a", "b"], "results": []})
    assert sorted(result["results"]) == ["A", "B"]


def run_in_executor_from_other_code(loop: asyncio.AbstractEventLoop) -> Any:
    return loop.run_in_executor(None, lambda: "other")


def called_by_other_code(frame: Any) -> bool:
    return frame.f_code is run_in_executor_from_other_code.__code__


def test_only_langchain_calls_run_inline() -> None:
    from langchain_core.runnables.config import run_in_executor

    from temporalio.langgraph._interceptor import (
        _called_by_langchain,
        _run_inline_when_called_by,
    )

    loop = asyncio.new_event_loop()
    try:

        def no_threads(*_args: Any) -> Any:
            # What the Workflow event loop does.
            raise NotImplementedError

        setattr(loop, "run_in_executor", no_threads)
        _run_inline_when_called_by(loop, _called_by_langchain)
        _run_inline_when_called_by(loop, _called_by_langchain)
        assert loop.run_until_complete(run_in_executor(None, lambda: 42)) == 42
        with pytest.raises(NotImplementedError):
            loop.run_in_executor(None, lambda: "unrelated")

        # Another plugin, such as Deep Agents, adds its caller to the same list.
        _run_inline_when_called_by(loop, called_by_other_code)
        assert loop.run_until_complete(run_in_executor_from_other_code(loop)) == "other"
        assert loop.run_until_complete(run_in_executor(None, lambda: 42)) == 42
        with pytest.raises(NotImplementedError):
            loop.run_in_executor(None, lambda: "unrelated")
    finally:
        loop.close()
