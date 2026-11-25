"""Test Graph API continue-as-new with task result caching.

Verifies that node results are cached across continue-as-new boundaries,
so nodes don't re-execute when the graph is re-invoked with the same state.
"""

import asyncio
import operator
import os
import subprocess
import sys
from contextvars import copy_context
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from numbers import Number
from typing import Annotated, Any
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph._internal._runnable import (  # pyright: ignore[reportMissingTypeStubs]
    RunnableCallable,
)
from langgraph.errors import GraphInterrupt
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
from temporalio.langgraph._task_cache import (
    cache_key,
    cache_lookup,
    cache_put,
    call_scope,
    get_task_cache,
    occurrence_key,
    release,
    set_task_cache,
)
from temporalio.langgraph._workflow import scope_router, wrap_workflow
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


@Number.register
class Meters:
    """A number type whose ``str()`` shows its address."""

    def __init__(self, value: float) -> None:
        self.value = value


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


def _occurrence_keys() -> None:
    key = cache_key("task", ("job-42",), {})
    other = cache_key("task", ("job-43",), {})
    set_task_cache({})
    # The first call keeps the key earlier versions wrote, so caches they
    # carried through continue-as-new still hit; each repeat gets its own.
    assert [occurrence_key(key) for _ in range(3)] == [key, f"{key}#1", f"{key}#2"]
    assert occurrence_key(other) == other
    # Each call of a workflow-side node or task counts its own calls under
    # its own keys.
    with call_scope("node"):
        first = [occurrence_key(key) for _ in range(2)]
    with call_scope("node"):
        second = occurrence_key(key)
    assert first[0].startswith(f"{key}@") and first[1] == f"{first[0]}#1"
    assert second not in (key, first[0])
    # graph() and entrypoint() start counting again.
    set_task_cache({})
    assert occurrence_key(key) == key
    # A call that failed or was interrupted stored no result, so the next
    # identical call takes its place. Places go back lowest first.
    set_task_cache({})
    calls = [occurrence_key(key) for _ in range(3)]
    release(calls[1])
    release(calls[0])
    assert [occurrence_key(key) for _ in range(3)] == [key, f"{key}#1", f"{key}#3"]
    # A node or task call that failed gives its place back too, and the
    # places of its calls, so its retry makes them under the same keys.
    # Their results stay in the cache but are stale: the retry's call runs.
    inner: list[str] = []
    with pytest.raises(ValueError), call_scope("node"):
        inner.append(occurrence_key(key))
        cache_put(inner[0], "failed attempt")
        raise ValueError
    assert get_task_cache() == {inner[0]: "failed attempt"}
    with call_scope("node"):
        assert occurrence_key(key) == inner[0]
        assert cache_lookup(inner[0]) == (False, None)
    assert get_task_cache() == {}
    # An interrupted one keeps them: the resumed call counts on.
    with pytest.raises(GraphInterrupt), call_scope("node"):
        inner.append(occurrence_key(key))
        raise GraphInterrupt
    with call_scope("node"):
        assert occurrence_key(key) == f"{inner[1]}#1"
    # A call that the failed call's retry doesn't make, because LangGraph
    # got its result from a checkpoint, keeps the result for the next run.
    set_task_cache({})
    kept: list[str] = []
    with pytest.raises(ValueError), call_scope("node"):
        kept.append(occurrence_key(other))
        cache_put(kept[0], "finished")
        raise ValueError
    with call_scope("node"):
        pass
    set_task_cache(get_task_cache())
    with call_scope("node"):
        assert occurrence_key(other) == kept[0]
    assert cache_lookup(kept[0]) == (True, "finished")
    # When LangGraph runs a failed call again from its checkpoint, as the
    # same task, the calls made in it keep their places: one it restores
    # from the checkpoint is not made again, and the one that failed takes
    # the place it gave back.
    set_task_cache({})
    token = var_child_runnable_config.set(
        RunnableConfig(metadata={"langgraph_checkpoint_ns": "node:1"})
    )
    done: list[str] = []
    with pytest.raises(ValueError), call_scope("node"):
        done.append(occurrence_key(key))
        cache_put(done[0], "finished")
        release(occurrence_key(key))
        raise ValueError
    with call_scope("node"):
        assert occurrence_key(key) == f"{done[0]}#1"
    var_child_runnable_config.reset(token)
    set_task_cache(get_task_cache())
    assert cache_lookup(done[0]) == (True, "finished")
    # Calls with other input count apart, so a failed call never shares a
    # scope with a call that has other input.
    set_task_cache({})
    with call_scope("node", ("a",)):
        first = [occurrence_key(key)]
    with call_scope("node", ("b",)):
        assert occurrence_key(key) != first[0]
    # A scope's place is not a place of an Activity call with the same name
    # and input.
    set_task_cache({})
    with call_scope("task", ("job-42",)):
        pass
    assert occurrence_key(key) == key


def test_repeated_calls_get_their_own_keys():
    copy_context().run(_occurrence_keys)


async def _wrapped_scope(injected: frozenset[str], **kwargs: Any) -> str:
    """Return the key of a call made in a workflow-side function given ``kwargs``."""
    keys: list[str] = []

    async def body(*args: Any, **kwargs: Any) -> None:  # pyright: ignore[reportUnusedParameter]
        keys.append(occurrence_key("key"))

    set_task_cache({})
    await wrap_workflow("node", body, injected=injected)("a", **kwargs)
    return keys[0]


def test_injected_keywords_are_not_input():
    """The config LangGraph passes a node is not input; a config given to a task is."""
    node = frozenset({"config", "runtime"})
    assert asyncio.run(_wrapped_scope(node, config={"run_id": 1})) == asyncio.run(
        _wrapped_scope(node, config={"run_id": 2}, runtime=object())
    )
    assert asyncio.run(_wrapped_scope(frozenset(), config="bad")) != asyncio.run(
        _wrapped_scope(frozenset(), config="good")
    )


def _scope(*args: Any) -> str:
    """Return the key of a call made in a ``call_scope()`` with ``args``."""
    set_task_cache({})
    with call_scope("node", args):
        return occurrence_key("key")


def _scopes_read_values() -> None:
    # str() of these shows an address or an order, so a scope reads their
    # values or, for an object without one, its class. Both helpers stay
    # alive, so their addresses differ, as on two Workers.
    helpers = [object(), object()]
    assert _scope(helpers[0], {"a", "b"}) == _scope(helpers[1], {"b", "a"})
    meters = [Meters(1), Meters(1)]
    assert _scope(meters[0]) == _scope(meters[1])
    assert _scope(Decimal("1.5")) != _scope(Decimal("2.5"))
    assert _scope({(1, 2): Tags({"a", "b"}), "n": 1}) == _scope(
        {"n": 1, (1, 2): Tags({"b", "a"})}
    )
    assert _scope(HumanMessage("hi")) == _scope(HumanMessage("hi"))
    assert _scope(HumanMessage("hi")) != _scope(HumanMessage("bye"))
    assert _scope(Tags({"a"})) != _scope(Tags({"b"}))
    assert _scope(Tags({"a"})) != _scope({"names": ["a"]})
    assert _scope("a") != _scope("b")


def test_scopes_read_values():
    copy_context().run(_scopes_read_values)


_SCOPE_IN_NEW_PROCESS = """
import json
from contextvars import copy_context
from temporalio.langgraph._task_cache import call_scope, occurrence_key, set_task_cache

def scope():
    set_task_cache({})
    state = {"tags": {"a", "b", "c"}, frozenset({"x", "y"}): json.dumps, "client": object()}
    with call_scope("node", (state,)):
        return occurrence_key("key")

print(copy_context().run(scope))
"""


def test_scopes_match_across_processes():
    """A scope is the same on every Worker, though str() of its input is not.

    Each process has its own hash seed, so its own set order, and its own
    object addresses.
    """
    keys = {
        subprocess.run(
            [sys.executable, "-c", _SCOPE_IN_NEW_PROCESS],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("0", "1", "2")
    }
    assert len(keys) == 1


class FactoryState(TypedDict):
    value: int
    low: int
    high: int


def make_scaled(field: str, factor: int):
    """Return a closure node setting ``field`` to ``value * factor``.

    Every closure built here shares the qualname
    ``make_scaled.<locals>.scale``, so their cache entries must be keyed
    by their distinct ``<graph>.<node>`` activity names, not the qualname.
    """

    async def scale(state: FactoryState) -> dict[str, int]:
        _execution_counts[field] = _execution_counts.get(field, 0) + 1
        return {field: state["value"] * factor}

    return scale


@dataclass
class FactoryCacheInput:
    value: int
    cache: dict[str, Any] | None = None
    phase: int = 1


@workflow.defn
class FactoryCacheWorkflow:
    """Runs two same-qualname closure nodes in parallel, continuing-as-new once.

    The parallel nodes receive identical state, so one cache entry shared
    between them would make the second node return the first's result.
    """

    @workflow.run
    async def run(self, input_data: FactoryCacheInput) -> dict[str, int]:
        app = graph("cached-factory", cache=input_data.cache).compile()
        result = await app.ainvoke({"value": input_data.value, "low": 0, "high": 0})

        if input_data.phase < 2:
            workflow.continue_as_new(
                FactoryCacheInput(value=input_data.value, cache=cache(), phase=2)
            )

        return result


async def test_graph_continue_as_new_cached_factory_nodes(client: Client):
    """Same-qualname closures cache under distinct ids and stay distinct.

    Both factory closures run with identical args, so an implementation
    keying them by their shared qualname would reuse one entry: one node
    would return the other's result and only one would execute. Each must
    run once, and phase 2 must hit each node's own entry.
    """
    _reset()
    metadata = {
        "execute_in": "activity",
        "start_to_close_timeout": timedelta(seconds=10),
    }
    g = StateGraph(FactoryState)
    g.add_node("low", make_scaled("low", 2), metadata=metadata)
    g.add_node("high", make_scaled("high", 10), metadata=metadata)
    g.add_edge(START, "low")
    g.add_edge(START, "high")

    task_queue = f"graph-cached-factory-{uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[FactoryCacheWorkflow],
        plugins=[LangGraphPlugin(graphs={"cached-factory": g})],
    ):
        result = await client.execute_workflow(
            FactoryCacheWorkflow.run,
            FactoryCacheInput(value=5),
            id=f"graph-cached-factory-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )

    assert result == {"value": 5, "low": 10, "high": 50}
    assert _execution_counts == {"low": 1, "high": 1}


@dataclass
class LegacyKeyInput:
    value: int
    cache: dict[str, Any] | None = None
    phase: int = 1


@workflow.defn
class LegacyKeyWorkflow:
    """Runs the one-node graph, continuing-as-new once with the cache."""

    @workflow.run
    async def run(self, input_data: LegacyKeyInput) -> dict[str, Any]:
        app = graph("cached-legacy-id", cache=input_data.cache).compile()
        state = await app.ainvoke({"value": input_data.value})

        if input_data.phase < 2:
            workflow.continue_as_new(
                LegacyKeyInput(value=input_data.value, cache=cache(), phase=2)
            )

        return {"state": state, "cache": cache()}


async def test_graph_continue_as_new_cached_module_qualname_key(client: Client):
    """A module-level node keeps its pre-change ``module.qualname`` cache id.

    The entry is seeded exactly as an earlier version would have written
    it — ``cache_key`` over the function's ``module.qualname``, its args,
    and its context — so a node whose cache was populated before this
    change is still honoured after continue-as-new. The seeded result (41)
    differs from a real execution (20 * 2) so a cache miss is observable.
    """
    _reset()
    metadata = {
        "execute_in": "activity",
        "start_to_close_timeout": timedelta(seconds=10),
    }
    g = StateGraph(State)
    g.add_node("double", double, metadata=metadata)
    g.add_edge(START, "double")

    initial = {"value": 20}
    legacy_id = f"{double.__module__}.{double.__qualname__}"
    seeded = {cache_key(legacy_id, (initial,), {}, None): {"value": 41}}

    task_queue = f"graph-cached-legacy-id-{uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[LegacyKeyWorkflow],
        plugins=[LangGraphPlugin(graphs={"cached-legacy-id": g})],
    ):
        result = await client.execute_workflow(
            LegacyKeyWorkflow.run,
            LegacyKeyInput(value=20, cache=seeded),
            id=f"graph-cached-legacy-id-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )

    assert result["state"] == {"value": 41}
    assert result["cache"] == seeded
    assert _execution_counts == {}


async def _route_twice(order: list[int]) -> dict[int, str]:
    """Run a router in two ``Send`` tasks of one node, in ``order``.

    Return the key of the call each task's router made.
    """
    keys: dict[int, str] = {}

    async def route(state: dict[str, Any], config: RunnableConfig) -> str:
        index = config.get("metadata", {})["langgraph_path"][1]
        keys[index] = occurrence_key(cache_key("task", (state["job_id"],), {}))
        return END

    router = scope_router("graph.node.route", RunnableCallable(None, route))
    set_task_cache({})
    for index in order:
        config = RunnableConfig(metadata={"langgraph_path": ("__pregel_push", index)})
        await router.ainvoke({"job_id": "job-42"}, config)
    return keys


def test_router_calls_count_per_task():
    """The routers of two tasks with the same input keep their own keys.

    LangGraph runs a router once its node finishes, so the order changes
    after continue-as-new, when the nodes' results come from the cache.
    """
    first = asyncio.run(_route_twice([1, 0]))
    assert asyncio.run(_route_twice([0, 1])) == first
    assert first[0] != first[1]
