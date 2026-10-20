import sys
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
from langgraph.graph import START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from temporalio import workflow
from temporalio.client import Client
from temporalio.langgraph import LangGraphPlugin, graph
from temporalio.worker import Worker

requires_async_context = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="LangGraph async context propagation requires Python >= 3.11",
)


class State(TypedDict):
    value: str
    context_type_a: str
    context_type_b: str


@dataclass
class Context:
    user_id: str


@dataclass
class ComputedContext(Context):
    label: str = field(init=False)

    def __post_init__(self) -> None:
        self.label = self.user_id.upper()


class DictContext(TypedDict):
    user_id: str


def context_user_id(context: Context | DictContext | None) -> str:
    if context is None:
        return ""
    if isinstance(context, dict):
        return context["user_id"]
    label = getattr(context, "label", None)
    if label is not None:
        return f"{context.user_id}:{label}"
    return context.user_id


async def node_a(
    state: State,
    runtime: Runtime[Any],
) -> dict[str, str]:
    return {
        "value": state["value"] + "a" + context_user_id(runtime.context),
        "context_type_a": type(runtime.context).__name__,
    }


async def node_b(
    state: State,
    runtime: Runtime[Any],
) -> dict[str, str]:
    return {
        "value": state["value"] + "b" + context_user_id(runtime.context),
        "context_type_b": type(runtime.context).__name__,
    }


@workflow.defn
class TwoNodesWorkflow:
    def __init__(self) -> None:
        self.app = graph("my-graph").compile()

    @workflow.run
    async def run(self, input: str, context_kind: str = "none") -> Any:
        context: Context | DictContext | None
        if context_kind == "computed":
            computed_context = ComputedContext(user_id="alice")
            computed_context.label = "saved"
            context = computed_context
        elif context_kind == "dataclass":
            context = Context(user_id="alice")
        elif context_kind in ("dict", "typeddict"):
            context = DictContext(user_id="alice")
        else:
            context = None
        return await self.app.ainvoke({"value": input}, context=context)


@pytest.mark.parametrize("first_node_execution", ["activity", "workflow"])
@pytest.mark.parametrize(
    ("context_kind", "expected_context_type"),
    [
        ("none", "NoneType"),
        pytest.param("dataclass", "Context", marks=requires_async_context),
        pytest.param("computed", "ComputedContext", marks=requires_async_context),
        pytest.param("dict", "Context", marks=requires_async_context),
        pytest.param("typeddict", "dict", marks=requires_async_context),
    ],
)
async def test_two_nodes(
    client: Client,
    first_node_execution: str,
    context_kind: str,
    expected_context_type: str,
):
    schema: type[Any] | None = (
        ComputedContext
        if context_kind == "computed"
        else DictContext
        if context_kind == "typeddict"
        else Context
        if context_kind in ("dataclass", "dict")
        else None
    )
    g = StateGraph(State, context_schema=schema)
    g.add_node("node_a", node_a, metadata={"execute_in": first_node_execution})
    g.add_node("node_b", node_b, metadata={"execute_in": "activity"})
    g.add_edge(START, "node_a")
    g.add_edge("node_a", "node_b")

    task_queue = f"my-graph-{uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[TwoNodesWorkflow],
        plugins=[
            LangGraphPlugin(
                graphs={"my-graph": g},
                default_activity_options={
                    "start_to_close_timeout": timedelta(seconds=10)
                },
            )
        ],
    ):
        result = await client.execute_workflow(
            TwoNodesWorkflow.run,
            args=["", context_kind],
            id=f"test-workflow-{uuid4()}",
            task_queue=task_queue,
        )

    expected_value = "ab" if context_kind == "none" else "aalicebalice"
    if context_kind == "computed":
        expected_value = "aalice:savedbalice:saved"
    assert result == {
        "value": expected_value,
        "context_type_a": expected_context_type,
        "context_type_b": expected_context_type,
    }
