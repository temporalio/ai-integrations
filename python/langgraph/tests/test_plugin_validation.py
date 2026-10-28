"""Tests for LangGraphPlugin validation."""

from __future__ import annotations

from dataclasses import make_dataclass
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.func import task  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph import START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy  # pyright: ignore[reportMissingTypeStubs]
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
