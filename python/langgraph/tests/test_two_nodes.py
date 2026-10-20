from dataclasses import dataclass
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


class State(TypedDict):
    value: str
    context_type_a: str
    context_type_b: str


@dataclass
class Context:
    user_id: str


class DictContext(TypedDict):
    user_id: str


def context_user_id(context: Context | DictContext | None) -> str:
    if context is None:
        return ""
    if isinstance(context, dict):
        return context["user_id"]
    return context.user_id


async def node_a(
    state: State, runtime: Runtime[Context | DictContext]
) -> dict[str, str]:
    return {
        "value": state["value"] + "a" + context_user_id(runtime.context),
        "context_type_a": type(runtime.context).__name__,
    }


async def node_b(
    state: State, runtime: Runtime[Context | DictContext]
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
        context = (
            Context(user_id="alice")
            if context_kind == "dataclass"
            else {"user_id": "alice"}
            if context_kind in ("dict", "typeddict")
            else None
        )
        return await self.app.ainvoke({"value": input}, context=context)


@pytest.mark.parametrize("first_node_execution", ["activity", "workflow"])
@pytest.mark.parametrize(
    ("context_kind", "expected_context_type"),
    [
        ("none", "NoneType"),
        ("dataclass", "Context"),
        ("dict", "Context"),
        ("typeddict", "dict"),
    ],
)
async def test_two_nodes(
    client: Client, first_node_execution: str, context_kind: str, expected_context_type: str
):
    schema = (
        DictContext
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

    assert result == {
        "value": "ab" if context_kind == "none" else "aalicebalice",
        "context_type_a": expected_context_type,
        "context_type_b": expected_context_type,
    }
