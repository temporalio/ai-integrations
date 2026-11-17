"""Tests for LangGraphPlugin validation."""

from __future__ import annotations

import importlib.metadata
from dataclasses import make_dataclass
from types import SimpleNamespace
from typing import Any, Callable
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.cache.memory import (  # pyright: ignore[reportMissingTypeStubs]
    InMemoryCache,
)
from langgraph.func import (  # pyright: ignore[reportMissingTypeStubs]
    entrypoint,
    task,
)
from langgraph.graph import START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.runtime import Runtime
from langgraph.types import (  # pyright: ignore[reportMissingTypeStubs]
    CachePolicy,
    RetryPolicy,
    TimeoutPolicy,
)
from pytest import raises
from typing_extensions import TypedDict

from temporalio.langgraph import LangGraphPlugin
from temporalio.langgraph._activity import (
    ActivityInput,
    wrap_activity,
    wrap_execute_activity,
)
from tests.test_two_nodes import Context, ModelContext


class State(TypedDict):
    value: str


async def async_node(state: State) -> dict[str, str]:  # pyright: ignore[reportUnusedParameter]
    return {"value": "done"}


def sync_node(state: State) -> dict[str, str]:  # pyright: ignore[reportUnusedParameter]
    return {"value": "done"}


def test_non_runnable_callable_node_raises() -> None:
    """Nodes whose runnable isn't a RunnableCallable can't be wrapped as activities."""
    g = StateGraph(State)
    g.add_node("node", RunnableLambda(sync_node))
    g.add_edge(START, "node")

    with raises(ValueError, match="must be a RunnableCallable"):
        LangGraphPlugin(graphs={f"validation-{uuid4()}": g})


def test_invalid_execute_in_raises() -> None:
    g = StateGraph(State)
    g.add_node("node", async_node, metadata={"execute_in": "bogus"})
    g.add_edge(START, "node")

    with raises(ValueError, match="Invalid execute_in value"):
        LangGraphPlugin(graphs={f"validation-{uuid4()}": g})


def test_graph_node_missing_execute_in_raises() -> None:
    g = StateGraph(State)
    g.add_node("node", async_node)
    g.add_edge(START, "node")

    with raises(ValueError, match="missing required 'execute_in'"):
        LangGraphPlugin(graphs={f"validation-{uuid4()}": g})


def test_functional_task_missing_execute_in_raises() -> None:
    @task
    def my_task(x: int) -> int:
        return x + 1

    with raises(ValueError, match="missing required 'execute_in'"):
        LangGraphPlugin(tasks=[my_task])


def test_functional_task_closure_raises() -> None:
    """@task functions must be module-level: their qualname is the activity name.

    Graph API nodes may be closures (see test_sync_node.py and
    test_continue_as_new_cached.py), but a Functional API task has no
    separate activity name to fall back to.
    """

    @task
    def my_task(x: int) -> int:
        return x + 1

    with raises(ValueError, match="closures/local functions are not supported"):
        LangGraphPlugin(
            tasks=[my_task],
            activity_options={"my_task": {"execute_in": "activity"}},
        )


def test_graph_node_main_module_raises() -> None:
    """Only closures and local functions get past task_id's rejections.

    A ``__main__`` node function stays rejected as before.
    """

    def node(state: State) -> dict[str, str]:  # pyright: ignore[reportUnusedParameter]
        return {"value": "done"}

    node.__qualname__ = node.__name__  # not a closure as far as task_id cares
    node.__module__ = "__main__"
    g = StateGraph(State)
    g.add_node("node", node, metadata={"execute_in": "activity"})
    g.add_edge(START, "node")

    with raises(ValueError, match="defined in __main__"):
        LangGraphPlugin(graphs={"main-module-graph": g})


def test_execute_in_in_default_activity_options_raises() -> None:
    with raises(ValueError, match="cannot be set in default_activity_options"):
        LangGraphPlugin(default_activity_options={"execute_in": "activity"})


def test_node_retry_policy_raises() -> None:
    g = StateGraph(State)
    g.add_node("node", async_node, retry_policy=RetryPolicy(max_attempts=3))
    g.add_edge(START, "node")

    with raises(ValueError, match="retry_policy"):
        LangGraphPlugin(graphs={f"validation-{uuid4()}": g})


def test_task_retry_policy_raises() -> None:
    decorator: Any = task(retry_policy=RetryPolicy(max_attempts=3))

    @decorator
    def my_task(x: int) -> int:
        return x + 1

    with raises(ValueError, match="retry_policy"):
        LangGraphPlugin(tasks=[my_task])


def test_node_defaults_retry_policy_raises() -> None:
    """compile() gives nodes the builder's defaults after the plugin is built."""
    g = StateGraph(State)
    g.set_node_defaults(retry_policy=RetryPolicy(max_attempts=3))
    g.add_node("node", async_node, metadata={"execute_in": "activity"})
    g.add_edge(START, "node")

    with raises(ValueError, match="Graph validation has a LangGraph retry_policy"):
        LangGraphPlugin(graphs={"validation": g})


def test_entrypoint_retry_policy_raises() -> None:
    @entrypoint(retry_policy=RetryPolicy(max_attempts=3))
    async def program(x: int) -> int:
        return x

    with raises(ValueError, match="Entrypoint program has a LangGraph retry_policy"):
        LangGraphPlugin(entrypoints={"program": program})


def _node_route(**policy: Any) -> dict[str, Any]:
    g = StateGraph(State)
    g.add_node("node", async_node, metadata={"execute_in": "activity"}, **policy)
    g.add_edge(START, "node")
    return {"graphs": {"validation": g}}


def _node_defaults_route(**policy: Any) -> dict[str, Any]:
    g = StateGraph(State)
    g.set_node_defaults(**policy)
    g.add_node("node", async_node, metadata={"execute_in": "activity"})
    g.add_edge(START, "node")
    return {"graphs": {"validation": g}}


def _task_route(**policy: Any) -> dict[str, Any]:
    decorator: Any = task(**policy)

    @decorator  # pyright: ignore[reportUntypedFunctionDecorator]
    async def my_task(x: int) -> int:
        return x + 1

    return {"tasks": [my_task]}


def _entrypoint_route(**policy: Any) -> dict[str, Any]:
    @entrypoint(**policy)
    async def program(x: int) -> int:
        return x

    return {"entrypoints": {"program": program}}


@pytest.mark.parametrize(
    ("route", "owner"),
    [
        (_node_route, "Node validation.node"),
        (_node_defaults_route, "Graph validation"),
        (_task_route, "Task my_task"),
        (_entrypoint_route, "Entrypoint program"),
    ],
    ids=["node", "node_defaults", "task", "entrypoint"],
)
@pytest.mark.parametrize(
    ("policy", "option"),
    [
        ({"timeout": TimeoutPolicy(idle_timeout=5)}, "idle_timeout"),
        ({"timeout": TimeoutPolicy(run_timeout=30, idle_timeout=5)}, "idle_timeout"),
        ({"cache_policy": CachePolicy(ttl=60)}, "cache_policy ttl"),
    ],
    ids=["idle_timeout", "run_and_idle_timeout", "cache_ttl"],
)
def test_idle_timeout_and_cache_ttl_raise(
    route: Callable[..., dict[str, Any]],
    owner: str,
    policy: dict[str, Any],
    option: str,
) -> None:
    with raises(ValueError, match=f"{owner} has a LangGraph {option}"):
        LangGraphPlugin(**route(**policy))


def test_entrypoint_cache_raises() -> None:
    @entrypoint(cache=InMemoryCache())
    async def program(x: int) -> int:
        return x

    with raises(ValueError, match="Entrypoint program has a LangGraph cache set"):
        LangGraphPlugin(entrypoints={"program": program})


_with_supported_policies: Any = task(
    timeout=TimeoutPolicy(run_timeout=30), cache_policy=CachePolicy()
)


@_with_supported_policies
async def supported_policies_task(x: int) -> int:
    return x + 1


@entrypoint(timeout=30, cache_policy=CachePolicy())
async def supported_policies_entrypoint(x: int) -> int:
    return await supported_policies_task(x)


def test_run_timeout_and_cache_policy_without_ttl_are_accepted() -> None:
    g = StateGraph(State)
    g.set_node_defaults(timeout=30, cache_policy=CachePolicy())
    g.add_node(
        "node",
        async_node,
        metadata={"execute_in": "activity"},
        timeout=TimeoutPolicy(run_timeout=30),
        cache_policy=CachePolicy(),
    )
    g.add_edge(START, "node")

    LangGraphPlugin(
        graphs={"validation": g},
        entrypoints={"supported": supported_policies_entrypoint},
        tasks=[supported_policies_task],
        activity_options={"supported_policies_task": {"execute_in": "activity"}},
    )


def test_context_schema_id_conflict_raises() -> None:
    first = make_dataclass("Collision", [("user_id", str)])
    second = make_dataclass("Collision", [("user_id", str)])
    with raises(ValueError, match="Conflicting context schema ID"):
        LangGraphPlugin(
            graphs={
                "first": StateGraph(State, context_schema=first),
                "second": StateGraph(State, context_schema=second),
            }
        )


@pytest.mark.parametrize("unknown_id", [False, True])
async def test_context_payload_errors(unknown_id: bool) -> None:
    schema_id = f"{ModelContext.__module__}.{ModelContext.__qualname__}"
    wrapped = wrap_activity(async_node, context_schemas={schema_id: ModelContext})
    input = ActivityInput(
        args=({},),
        kwargs={},
        langgraph_config={
            "context_schema_id": "unknown" if unknown_id else schema_id,
            "context": {"userId": 123, "tenant": {"name": "acme"}},
        },
    )
    with raises(
        ValueError, match="Unknown context schema ID" if unknown_id else "userId"
    ):
        await wrapped(input)


async def test_unregistered_context_type_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "temporalio.workflow.info",
        lambda: SimpleNamespace(run_id="unregistered-context"),
    )
    wrapped = wrap_execute_activity(wrap_activity(async_node))
    with raises(ValueError, match="is not registered"):
        await wrapped({}, runtime=Runtime(context=Context(user_id="alice")))


def test_plugin_reports_package_version() -> None:
    assert LangGraphPlugin().version() == importlib.metadata.version(
        "temporalio-langgraph"
    )
