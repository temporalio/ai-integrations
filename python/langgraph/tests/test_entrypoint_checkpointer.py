"""Functional API checkpoints belong to one Workflow execution instance."""

from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.func import (  # pyright: ignore[reportMissingTypeStubs]
    entrypoint as lg_entrypoint,
)
from langgraph.graph import START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from temporalio import workflow
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowHandle, WorkflowHistory
from temporalio.langgraph import LangGraphPlugin, entrypoint, graph
from temporalio.worker import Replayer, Worker


@dataclass
class MemoryInput:
    values: list[str]
    thread_id: str = "conversation"
    wait: bool = False
    runs_left: int = 0
    seen: list[list[str]] = field(default_factory=list)


@workflow.defn
class MemoryWorkflow:
    def __init__(self) -> None:
        self.program = entrypoint("memory")
        self.finished = False

    @workflow.signal
    def finish(self) -> None:
        self.finished = True

    @workflow.run
    async def run(self, request: MemoryInput) -> list[list[str]]:
        config: RunnableConfig = {"configurable": {"thread_id": request.thread_id}}
        result: list[str] = []
        for index, value in enumerate(request.values):
            assert entrypoint("memory") is self.program
            assert entrypoint("alias") is self.program
            result = cast(list[str], await self.program.ainvoke(value, config))
            assert result == request.values[: index + 1]
            if request.wait and index == 0:
                await workflow.wait_condition(lambda: self.finished)
        seen = [*request.seen, result]
        if request.runs_left:
            workflow.continue_as_new(
                replace(request, runs_left=request.runs_left - 1, seen=seen)
            )
        return seen


def memory_plugin(saver: InMemorySaver | None = None) -> LangGraphPlugin:
    @lg_entrypoint(checkpointer=saver if saver is not None else InMemorySaver())
    async def remember(value: str, previous: list[str] | None = None) -> list[str]:
        return [*(previous or []), value]

    return LangGraphPlugin(entrypoints={"memory": remember, "alias": remember})


async def wait_for_task(handle: WorkflowHandle[Any, Any]) -> None:
    """Wait for the first task to finish before delivering its pending signal."""
    import asyncio

    async with asyncio.timeout(10):
        while True:
            history = await handle.fetch_history()
            if any(
                event.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_COMPLETED
                for event in history.events
            ):
                return
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("on_client", [False, True], ids=["worker", "client"])
@pytest.mark.parametrize("evict", [False, True], ids=["cached", "evicted"])
@pytest.mark.parametrize(
    "shared_thread", [False, True], ids=["unique-thread", "shared-thread"]
)
async def test_memory_is_scoped_to_each_workflow(
    client: Client, on_client: bool, evict: bool, shared_thread: bool
) -> None:
    """Two workflows may reuse a thread ID; replay restores each run's own state."""
    plugin = memory_plugin()
    if on_client:
        client_config = client.config()
        client_config["plugins"] = [plugin]
        client = Client(**client_config)
    task_queue = f"checkpoint-{uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[MemoryWorkflow],
        plugins=[] if on_client else [plugin],
        max_cached_workflows=0 if evict else 1000,
    ):
        handles = [
            await client.start_workflow(
                MemoryWorkflow.run,
                MemoryInput(
                    [name, "done"],
                    thread_id="conversation" if shared_thread else name,
                    wait=True,
                ),
                id=f"{name}-{uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=30),
            )
            for name in ("alice", "bob")
        ]
        for handle in handles:
            await wait_for_task(handle)
        for handle in handles:
            await handle.signal(MemoryWorkflow.finish)
        assert [await handle.result() for handle in handles] == [
            [["alice", "done"]],
            [["bob", "done"]],
        ]

    replayer = Replayer(workflows=[MemoryWorkflow], plugins=[plugin])
    for handle in handles:
        await replayer.replay_workflow(await handle.fetch_history())


async def test_continue_as_new_starts_with_a_fresh_saver(client: Client) -> None:
    """A new run starts its native memory again, even when it reuses the thread ID."""
    plugin = memory_plugin()
    task_queue = f"checkpoint-can-{uuid4()}"
    async with Worker(
        client, task_queue=task_queue, workflows=[MemoryWorkflow], plugins=[plugin]
    ):
        handle = await client.start_workflow(
            MemoryWorkflow.run,
            MemoryInput(["alice", "done"], runs_left=1),
            id=f"checkpoint-can-{uuid4()}",
            task_queue=task_queue,
        )
        first_run = client.get_workflow_handle(
            handle.id, run_id=handle.first_execution_run_id
        )
        assert await handle.result() == [["alice", "done"], ["alice", "done"]]
    replayer = Replayer(workflows=[MemoryWorkflow], plugins=[plugin])
    await replayer.replay_workflow(await first_run.fetch_history())
    await replayer.replay_workflow(await handle.fetch_history())


class CountingSerializer(JsonPlusSerializer):
    def __init__(self) -> None:
        super().__init__()
        self.dumps_calls = 0

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        self.dumps_calls += 1
        return super().dumps_typed(obj)


async def test_registered_saver_supplies_its_serializer(client: Client) -> None:
    serializer = CountingSerializer()
    saver = InMemorySaver(serde=serializer)
    plugin = memory_plugin(saver)
    task_queue = f"checkpoint-serde-{uuid4()}"
    async with Worker(
        client, task_queue=task_queue, workflows=[MemoryWorkflow], plugins=[plugin]
    ):
        assert await client.execute_workflow(
            MemoryWorkflow.run,
            MemoryInput(["alice", "done"]),
            id=f"checkpoint-serde-{uuid4()}",
            task_queue=task_queue,
        ) == [["alice", "done"]]
    assert serializer.dumps_calls > 0
    assert not saver.storage


@workflow.defn
class ApprovalWorkflow:
    def __init__(self) -> None:
        self.decision: str | None = None

    @workflow.signal
    def decide(self, decision: str) -> None:
        self.decision = decision

    @workflow.run
    async def run(self, value: str) -> dict[str, str]:
        config: RunnableConfig = {"configurable": {"thread_id": "approval"}}
        asked = await entrypoint("approval").ainvoke(value, config)
        assert asked["__interrupt__"][0].value == f"approve {value}?"
        await workflow.wait_condition(lambda: self.decision is not None)
        return await entrypoint("approval").ainvoke(
            Command(resume=self.decision), config
        )


async def test_interrupt_resumes_after_workflow_eviction(client: Client) -> None:
    @lg_entrypoint(checkpointer=InMemorySaver())
    async def approve(value: str) -> dict[str, str]:
        return {"value": value, "decision": interrupt(f"approve {value}?")}

    plugin = LangGraphPlugin(entrypoints={"approval": approve})
    task_queue = f"checkpoint-approval-{uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ApprovalWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            ApprovalWorkflow.run,
            "alice",
            id=f"checkpoint-approval-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        await wait_for_task(handle)
        await handle.signal(ApprovalWorkflow.decide, "yes")
        assert await handle.result() == {"value": "alice", "decision": "yes"}
    await Replayer(workflows=[ApprovalWorkflow], plugins=[plugin]).replay_workflow(
        await handle.fetch_history()
    )


class GraphState(TypedDict):
    value: str
    remembered: list[str]


async def remember_in_graph(state: GraphState) -> dict[str, list[str]]:
    result = await entrypoint("memory").ainvoke(
        state["value"], {"configurable": {"thread_id": "graph"}}
    )
    return {"remembered": cast(list[str], result)}


@workflow.defn
class GraphWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        app = graph("parent").compile()
        await app.ainvoke({"value": "alice"})
        result = await app.ainvoke({"value": "done"})
        return result["remembered"]


async def test_workflow_graph_nodes_share_the_entrypoint_instance(
    client: Client,
) -> None:
    @lg_entrypoint(checkpointer=InMemorySaver())
    async def remember(value: str, previous: list[str] | None = None) -> list[str]:
        return [*(previous or []), value]

    parent = StateGraph(GraphState)
    parent.add_node("remember", remember_in_graph, metadata={"execute_in": "workflow"})
    parent.add_edge(START, "remember")
    plugin = LangGraphPlugin(
        graphs={"parent": parent}, entrypoints={"memory": remember}
    )
    task_queue = f"checkpoint-graph-{uuid4()}"
    async with Worker(
        client, task_queue=task_queue, workflows=[GraphWorkflow], plugins=[plugin]
    ):
        handle = await client.start_workflow(
            GraphWorkflow.run, id=f"checkpoint-graph-{uuid4()}", task_queue=task_queue
        )
        assert await handle.result() == ["alice", "done"]
    await Replayer(workflows=[GraphWorkflow], plugins=[plugin]).replay_workflow(
        await handle.fetch_history()
    )


REPLAY_MAJOR = 1


async def test_committed_entrypoint_history_replays_twice() -> None:
    """The same plugin may replay a recorded run again without retaining its state."""
    history = WorkflowHistory.from_json(
        "entrypoint-checkpointer",
        (
            Path(__file__).parent
            / "histories"
            / f"{REPLAY_MAJOR}.x"
            / "entrypoint_checkpointer.json"
        ).read_text(),
    )
    replayer = Replayer(workflows=[MemoryWorkflow], plugins=[memory_plugin()])
    await replayer.replay_workflow(history)
    await replayer.replay_workflow(history)


class CustomMemorySaver(InMemorySaver):
    pass


@pytest.mark.parametrize("saver", [BaseCheckpointSaver(), CustomMemorySaver()])
def test_custom_checkpointers_are_refused(saver: BaseCheckpointSaver) -> None:
    @lg_entrypoint(checkpointer=saver)
    async def custom(value: str) -> str:
        return value

    with pytest.raises(
        ValueError, match="Custom checkpointers and InMemorySaver subclasses"
    ):
        LangGraphPlugin(entrypoints={"custom": custom})
