"""Test Graph API continue-as-new with task result caching.

Verifies that node results are cached across continue-as-new boundaries,
so nodes don't re-execute when the graph is re-invoked with the same state.
"""

import operator
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated, Any
from uuid import uuid4

import pytest
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    StateGraph,
)
from langgraph.types import Command, Send
from typing_extensions import TypedDict

from temporalio import workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.langgraph import LangGraphPlugin, cache, graph
from temporalio.langgraph._task_cache import cache_key
from temporalio.worker import Worker

# Track execution counts to verify caching
_execution_counts: dict[str, int] = {}


def _reset():
    _execution_counts.clear()


class State(TypedDict):
    value: int


async def multiply_by_3(state: State) -> dict[str, int]:
    _execution_counts["multiply"] = _execution_counts.get("multiply", 0) + 1
    return {"value": state["value"] * 3}


async def add_100(state: State) -> dict[str, int]:
    _execution_counts["add"] = _execution_counts.get("add", 0) + 1
    return {"value": state["value"] + 100}


async def double(state: State) -> dict[str, int]:
    _execution_counts["double"] = _execution_counts.get("double", 0) + 1
    return {"value": state["value"] * 2}


@dataclass
class GraphContinueAsNewInput:
    value: int
    cache: dict[str, Any] | None = None
    phase: int = 1  # 1, 2, 3 — continues-as-new after phases 1 and 2


@workflow.defn
class GraphContinueAsNewWorkflow:
    """Runs a 3-node graph, continuing-as-new after each phase.

    Phase 1: runs graph (all 3 nodes execute), continues-as-new with cache.
    Phase 2: runs graph again with same input (all 3 cached), continues-as-new.
    Phase 3: runs graph again with same input (all 3 cached), returns result.

    Without caching: each node executes 3 times.
    With caching: each node executes once (first run), cached for phases 2 & 3.
    """

    @workflow.run
    async def run(self, input_data: GraphContinueAsNewInput) -> dict[str, int]:
        app = graph("cached-graph", cache=input_data.cache).compile()
        result = await app.ainvoke({"value": input_data.value})

        if input_data.phase < 3:
            workflow.continue_as_new(
                GraphContinueAsNewInput(
                    value=input_data.value,
                    cache=cache(),
                    phase=input_data.phase + 1,
                )
            )

        return result


async def test_graph_continue_as_new_cached(client: Client):
    """Each node executes once despite 3 continue-as-new cycles.

    Graph: multiply_by_3 -> add_100 -> double
    Input 10: 10 * 3 = 30 -> 30 + 100 = 130 -> 130 * 2 = 260
    """
    _reset()

    metadata = {
        "execute_in": "activity",
        "start_to_close_timeout": timedelta(seconds=10),
    }
    g = StateGraph(State)
    g.add_node("multiply_by_3", multiply_by_3, metadata=metadata)
    g.add_node("add_100", add_100, metadata=metadata)
    g.add_node("double", double, metadata=metadata)
    g.add_edge(START, "multiply_by_3")
    g.add_edge("multiply_by_3", "add_100")
    g.add_edge("add_100", "double")

    task_queue = f"graph-cached-{uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[GraphContinueAsNewWorkflow],
        plugins=[LangGraphPlugin(graphs={"cached-graph": g})],
    ):
        result = await client.execute_workflow(
            GraphContinueAsNewWorkflow.run,
            GraphContinueAsNewInput(value=10),
            id=f"graph-cached-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )

    # 10 * 3 = 30 -> + 100 = 130 -> * 2 = 260
    assert result == {"value": 260}

    # Each node should execute exactly once — phases 2 and 3 use cached results.
    assert _execution_counts.get("multiply", 0) == 1, (
        f"multiply executed {_execution_counts.get('multiply', 0)} times, expected 1"
    )
    assert _execution_counts.get("add", 0) == 1, (
        f"add executed {_execution_counts.get('add', 0)} times, expected 1"
    )
    assert _execution_counts.get("double", 0) == 1, (
        f"double executed {_execution_counts.get('double', 0)} times, expected 1"
    )


class FanOutState(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]


class ItemState(TypedDict):
    item: str


async def dispatch(state: FanOutState) -> Command:
    _execution_counts["dispatch"] = _execution_counts.get("dispatch", 0) + 1
    return Command(goto=[Send("upper", {"item": item}) for item in state["items"]])


async def upper(state: ItemState) -> dict[str, list[str]]:
    key = f"upper:{state['item']}"
    _execution_counts[key] = _execution_counts.get(key, 0) + 1
    return {"results": [state["item"].upper()]}


@dataclass
class FanOutContinueAsNewInput:
    items: list[str]
    cache: dict[str, Any] | None = None
    phase: int = 1


@workflow.defn
class FanOutContinueAsNewWorkflow:
    """Runs a Command(goto=Send) fan-out, continuing-as-new once with the cache."""

    @workflow.run
    async def run(self, input_data: FanOutContinueAsNewInput) -> dict[str, Any]:
        app = graph("cached-fan-out", cache=input_data.cache).compile()
        result = await app.ainvoke({"items": input_data.items, "results": []})

        if input_data.phase < 2:
            workflow.continue_as_new(
                FanOutContinueAsNewInput(items=input_data.items, cache=cache(), phase=2)
            )

        return result


async def test_graph_continue_as_new_cached_command_send(client: Client):
    """A cached Command(goto=Send) survives continue-as-new and still routes."""
    _reset()

    metadata = {
        "execute_in": "activity",
        "start_to_close_timeout": timedelta(seconds=10),
        "retry_policy": RetryPolicy(maximum_attempts=1),
    }
    g = StateGraph(FanOutState)
    g.add_node("dispatch", dispatch, metadata=metadata)
    g.add_node("upper", upper, metadata=metadata)
    g.add_edge(START, "dispatch")
    g.add_edge("upper", END)

    task_queue = f"graph-cached-fan-out-{uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[FanOutContinueAsNewWorkflow],
        plugins=[LangGraphPlugin(graphs={"cached-fan-out": g})],
    ):
        result = await client.execute_workflow(
            FanOutContinueAsNewWorkflow.run,
            FanOutContinueAsNewInput(items=["a", "b"]),
            id=f"graph-cached-fan-out-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )

    assert sorted(result["results"]) == ["A", "B"]
    assert _execution_counts == {"dispatch": 1, "upper:a": 1, "upper:b": 1}


def goto_a(state: State) -> Command:
    _execution_counts["goto_a"] = _execution_counts.get("goto_a", 0) + 1
    return Command(update={"value": state["value"] * 10}, goto="goto_b")


def goto_b(state: State) -> Command:
    _execution_counts["goto_b"] = _execution_counts.get("goto_b", 0) + 1
    return Command(update={"value": state["value"] + 1}, goto=END)


@dataclass
class CommandCacheInput:
    cache: dict[str, Any] | None = None
    phase: int = 1
    phases: int = 2


@workflow.defn
class CommandCacheWorkflow:
    """Runs a string-goto Command graph, continuing-as-new until ``phases``."""

    @workflow.run
    async def run(self, input_data: CommandCacheInput) -> dict[str, Any]:
        app = graph("cached-command", cache=input_data.cache).compile()
        state = await app.ainvoke({"value": 1})

        if input_data.phase < input_data.phases:
            workflow.continue_as_new(
                CommandCacheInput(
                    cache=cache(),
                    phase=input_data.phase + 1,
                    phases=input_data.phases,
                )
            )

        return {"state": state, "cache": cache()}


def _command_cache_worker(client: Client, task_queue: str) -> Worker:
    metadata = {
        "execute_in": "activity",
        "start_to_close_timeout": timedelta(seconds=10),
        "retry_policy": RetryPolicy(maximum_attempts=1),
    }
    g = StateGraph(State)
    g.add_node("goto_a", goto_a, metadata=metadata)
    g.add_node("goto_b", goto_b, metadata=metadata)
    g.add_edge(START, "goto_a")
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[CommandCacheWorkflow],
        plugins=[LangGraphPlugin(graphs={"cached-command": g})],
    )


async def test_graph_continue_as_new_cached_command_goto(client: Client):
    """A cached string-goto Command keeps its update and routing after continue-as-new."""
    _reset()
    task_queue = f"graph-cached-command-{uuid4()}"

    async with _command_cache_worker(client, task_queue):
        result = await client.execute_workflow(
            CommandCacheWorkflow.run,
            CommandCacheInput(),
            id=f"graph-cached-command-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )

    assert result["state"] == {"value": 11}
    assert _execution_counts == {"goto_a": 1, "goto_b": 1}


async def test_graph_cached_legacy_command_dict_unchanged(client: Client):
    """A Command cached in the pre-Send format still comes back as a plain dict.

    Before Send support the cache held the live Command, which continue-as-new
    encoded as an untagged dict under the plain cache key. Such a dict can't be
    told apart from an ordinary result, so it is returned unchanged.
    """
    _reset()
    task_queue = f"graph-cached-legacy-{uuid4()}"

    async with _command_cache_worker(client, task_queue):
        first = await client.execute_workflow(
            CommandCacheWorkflow.run,
            CommandCacheInput(phases=1),
            id=f"graph-cached-legacy-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )
        legacy_cache = {
            key.removesuffix(":command"): value for key, value in first["cache"].items()
        }
        assert legacy_cache.keys() != first["cache"].keys()
        second = await client.execute_workflow(
            CommandCacheWorkflow.run,
            CommandCacheInput(cache=legacy_cache, phases=1),
            id=f"graph-cached-legacy-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )

    assert first["state"] == {"value": 11}
    # Each untagged dict is applied as a state update with no "value" key, so
    # the Command's update and routing are skipped, as they were before.
    assert second["state"] == {"value": 1}
    assert _execution_counts == {"goto_a": 1, "goto_b": 1}


# Results are cached under the plain key and Commands under "<key>:command",
# so a result dict comes back as is, even when it looks like a cache marker
# wrapping a Command payload.
_MARKER = "__temporal_langgraph_command__"
ResultState = TypedDict("ResultState", {"__temporal_langgraph_command__": Any})


def ordinary_result(state: ResultState) -> dict[str, Any]:
    _execution_counts["ordinary"] = _execution_counts.get("ordinary", 0) + 1
    return {_MARKER: state["__temporal_langgraph_command__"]}


@dataclass
class OrdinaryResultInput:
    marker: Any
    cache: dict[str, Any] | None = None
    phase: int = 1


@workflow.defn
class OrdinaryResultWorkflow:
    @workflow.run
    async def run(self, input_data: OrdinaryResultInput) -> dict[str, Any]:
        app = graph("cached-ordinary", cache=input_data.cache).compile()
        state = await app.ainvoke({_MARKER: input_data.marker})

        if input_data.phase < 2:
            workflow.continue_as_new(
                OrdinaryResultInput(marker=input_data.marker, cache=cache(), phase=2)
            )

        return {"state": state, "cache": cache()}


@pytest.mark.parametrize(
    "marker",
    [
        "ordinary",
        {"graph": None, "update": {"value": 1}, "resume": None, "goto": "elsewhere"},
    ],
)
async def test_graph_continue_as_new_cached_ordinary_dict(client: Client, marker: Any):
    """A cached result dict is never mistaken for a cached Command."""
    _reset()
    metadata = {
        "execute_in": "activity",
        "start_to_close_timeout": timedelta(seconds=10),
        "retry_policy": RetryPolicy(maximum_attempts=1),
    }
    g = StateGraph(ResultState)
    g.add_node("ordinary_result", ordinary_result, metadata=metadata)
    g.add_edge(START, "ordinary_result")

    task_queue = f"graph-cached-ordinary-{uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[OrdinaryResultWorkflow],
        plugins=[LangGraphPlugin(graphs={"cached-ordinary": g})],
    ):
        result = await client.execute_workflow(
            OrdinaryResultWorkflow.run,
            OrdinaryResultInput(marker=marker),
            id=f"graph-cached-ordinary-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )

    assert result["state"] == {_MARKER: marker}
    assert list(result["cache"].values()) == [{_MARKER: marker}]
    assert _execution_counts == {"ordinary": 1}


@dataclass(repr=False)
class Tags:
    names: set[str]


def test_cache_key_reads_values():
    # str() of these depends on object identity and set order, which differ
    # between processes.
    assert cache_key("task", (Tags({"a", "b"}),), {}) == cache_key(
        "task", (Tags({"b", "a"}),), {}
    )
    # A plain value with the same contents is a different argument.
    assert cache_key("task", (Tags({"a"}),), {}) != cache_key(
        "task", ({"names": ["a"]},), {}
    )
    assert cache_key("task", ({"a"},), {}) != cache_key("task", (["a"],), {})
