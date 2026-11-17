"""Tests for a graph's default error handler (set_node_defaults)."""

from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from langgraph.errors import NodeError
from langgraph.graph import START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.types import Command
from typing_extensions import TypedDict

from temporalio import workflow
from temporalio.client import Client, WorkflowHandle, WorkflowHistory
from temporalio.exceptions import ApplicationError
from temporalio.langgraph import LangGraphPlugin, graph
from temporalio.worker import Replayer, Worker

# Replay is promised only within a major (see AGENTS.md), so only the histories
# in the current major's folder replay.
REPLAY_MAJOR = 1
HISTORIES = Path(__file__).parent / "histories" / f"{REPLAY_MAJOR}.x"

CHARGED = {"value": "charged, notified"}
RECOVERED = {"value": "charge: ActivityError: card declined, notified"}
# A failed Workflow Task retries without end, so bound the wait.
EXECUTION_TIMEOUT = timedelta(seconds=20)


class State(TypedDict):
    value: str


async def charge(state: State) -> dict[str, str]:
    if state["value"] == "declined":
        raise ApplicationError("card declined", non_retryable=True)
    return {"value": "charged"}


async def notify(state: State) -> dict[str, str]:
    return {"value": state["value"] + ", notified"}


async def recover(
    state: State,  # pyright: ignore[reportUnusedParameter]
    error: NodeError,
) -> Command[str]:
    cause = error.error.__cause__ or error.error
    return Command(
        update={"value": f"{error.node}: {type(error.error).__name__}: {cause}"},
        goto="notify",
    )


@workflow.defn
class DefaultErrorHandlerWorkflow:
    def __init__(self) -> None:
        self.app = graph("my-graph").compile()

    @workflow.run
    async def run(self, input: str) -> Any:
        return await self.app.ainvoke({"value": input})


@workflow.defn
class CompilePerInputWorkflow:
    @workflow.run
    async def run(self, inputs: list[str]) -> list[Any]:
        return [
            await graph("my-graph").compile().ainvoke({"value": input})
            for input in inputs
        ]


def _plugin(charge_in: str = "activity") -> LangGraphPlugin:
    g = StateGraph(State)
    # LangGraph's type for error_handler leaves out the injected error argument.
    g.set_node_defaults(error_handler=recover)  # type: ignore[arg-type]
    g.add_node("charge", charge, metadata={"execute_in": charge_in})
    g.add_node("notify", notify, metadata={"execute_in": "activity"})
    g.add_edge(START, "charge")
    g.add_edge("charge", "notify")
    return LangGraphPlugin(
        graphs={"my-graph": g},
        default_activity_options={"start_to_close_timeout": timedelta(seconds=10)},
    )


async def _scheduled_activities(handle: WorkflowHandle[Any, Any]) -> list[str]:
    return [
        e.activity_task_scheduled_event_attributes.activity_type.name
        async for e in handle.fetch_history_events()
        if e.HasField("activity_task_scheduled_event_attributes")
    ]


async def test_default_error_handler_two_workflows(client: Client) -> None:
    plugin = _plugin()
    task_queue = f"error-handler-{uuid4()}"
    handles: list[WorkflowHandle[Any, Any]] = []
    results: list[Any] = []

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[DefaultErrorHandlerWorkflow],
        plugins=[plugin],
    ):
        for input in ("ok", "declined"):
            handle = await client.start_workflow(
                DefaultErrorHandlerWorkflow.run,
                input,
                id=f"test-workflow-{uuid4()}",
                task_queue=task_queue,
                execution_timeout=EXECUTION_TIMEOUT,
            )
            results.append(await handle.result())
            handles.append(handle)

    assert results == [CHARGED, RECOVERED]

    # The same plugin replays both runs, as a Replayer in the Worker's process does.
    replayer = Replayer(workflows=[DefaultErrorHandlerWorkflow], plugins=[plugin])
    for handle in handles:
        await replayer.replay_workflow(await handle.fetch_history())


async def test_default_error_handler_evicted_workflow(client: Client) -> None:
    task_queue = f"error-handler-{uuid4()}"

    # Without a cache, every Workflow Task replays the run and compiles again.
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[DefaultErrorHandlerWorkflow],
        plugins=[_plugin()],
        max_cached_workflows=0,
    ):
        result = await client.execute_workflow(
            DefaultErrorHandlerWorkflow.run,
            "declined",
            id=f"test-workflow-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=EXECUTION_TIMEOUT,
        )

    assert result == RECOVERED


async def test_default_error_handler_compiled_twice_in_one_run(client: Client) -> None:
    task_queue = f"error-handler-{uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[CompilePerInputWorkflow],
        plugins=[_plugin()],
    ):
        result = await client.execute_workflow(
            CompilePerInputWorkflow.run,
            ["ok", "declined"],
            id=f"test-workflow-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=EXECUTION_TIMEOUT,
        )

    assert result == [CHARGED, RECOVERED]


@pytest.mark.parametrize(
    ("charge_in", "error_type", "activities"),
    [
        ("activity", "ActivityError", ["my-graph.charge", "my-graph.notify"]),
        ("workflow", "ApplicationError", ["my-graph.notify"]),
    ],
    ids=["activity", "workflow"],
)
async def test_default_error_handler_runs_on_node_failure(
    client: Client, charge_in: str, error_type: str, activities: list[str]
) -> None:
    task_queue = f"error-handler-{uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[DefaultErrorHandlerWorkflow],
        plugins=[_plugin(charge_in)],
    ):
        handle = await client.start_workflow(
            DefaultErrorHandlerWorkflow.run,
            "declined",
            id=f"test-workflow-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=EXECUTION_TIMEOUT,
        )
        result = await handle.result()

    assert result == {"value": f"charge: {error_type}: card declined, notified"}
    # The handler runs in the Workflow, so it schedules no Activity of its own.
    assert await _scheduled_activities(handle) == activities


# Recorded on main at d9687b9, when graph() still returned the registered
# builder, against a local dev server. Only the stack trace of the failed
# Activity was removed.
async def test_default_error_handler_replays_earlier_history() -> None:
    history = WorkflowHistory.from_json(
        "default-error-handler",
        (HISTORIES / "default_error_handler.json").read_text(),
    )

    await Replayer(
        workflows=[DefaultErrorHandlerWorkflow], plugins=[_plugin()]
    ).replay_workflow(history)
