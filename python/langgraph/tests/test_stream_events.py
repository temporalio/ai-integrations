"""Event streams complete in Workflows and replay without repeating Activities."""

import asyncio
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    StateGraph,
)
from typing_extensions import TypedDict

from temporalio import activity, workflow
from temporalio.client import Client, WithStartWorkflowOperation
from temporalio.common import WorkflowIDConflictPolicy
from temporalio.langgraph import LangGraphPlugin, graph
from temporalio.worker import Replayer, Worker


class State(TypedDict):
    text: str


def shout(state: State) -> dict[str, str]:
    return {"text": state["text"].upper()}


@workflow.defn
class StreamEventsWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        app = graph("shout").compile()
        return [
            event["event"]
            async for event in app.astream_events({"text": "hi"}, version="v2")
        ]


async def test_stream_events_completes_and_replays(client: Client) -> None:
    g = StateGraph(State)
    g.add_node("shout", shout, metadata={"execute_in": "activity"})
    g.add_edge(START, "shout")
    g.add_edge("shout", END)
    expected = [
        event["event"]
        async for event in g.compile().astream_events({"text": "hi"}, version="v2")
    ]
    plugin = LangGraphPlugin(
        graphs={"shout": g},
        default_activity_options={"start_to_close_timeout": timedelta(seconds=10)},
    )
    queue = f"stream-events-{uuid4()}"
    async with Worker(
        client, task_queue=queue, workflows=[StreamEventsWorkflow], plugins=[plugin]
    ):
        handle = await client.start_workflow(
            StreamEventsWorkflow.run, id=str(uuid4()), task_queue=queue
        )
        try:
            result = await asyncio.wait_for(handle.result(), 10)
            history = await handle.fetch_history()
        except BaseException:
            await handle.terminate()
            raise
    assert result == expected
    await Replayer(workflows=[StreamEventsWorkflow], plugins=[plugin]).replay_workflow(
        history
    )


@workflow.defn
class StartHandlerStreamWorkflow:
    def __init__(self) -> None:
        self.events: list[str] | None = None

    @workflow.run
    async def run(self) -> list[str]:
        await workflow.wait_condition(lambda: self.events is not None)
        assert self.events is not None
        return self.events

    async def collect(self) -> None:
        app = graph("shout").compile()
        events: list[str] = []
        async for message, _ in app.astream({"text": "hi"}, stream_mode="messages"):
            assert isinstance(message, BaseMessage)
            events.append(message.text)
        self.events = events

    @workflow.signal
    async def signal_stream(self) -> None:
        await self.collect()

    @workflow.update
    async def update_stream(self) -> None:
        await self.collect()


@pytest.mark.parametrize("handler", ["signal", "update"])
async def test_streams_from_start_handlers(client: Client, handler: str) -> None:
    g = StateGraph(State)
    g.add_node("shout", chat, metadata={"execute_in": "workflow"})
    g.add_edge(START, "shout")
    g.add_edge("shout", END)
    plugin = LangGraphPlugin(
        graphs={"shout": g},
        default_activity_options={"start_to_close_timeout": timedelta(seconds=10)},
    )
    queue = f"start-stream-{uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[StartHandlerStreamWorkflow],
        plugins=[plugin],
    ):
        if handler == "signal":
            handle = await client.start_workflow(
                StartHandlerStreamWorkflow.run,
                id=str(uuid4()),
                task_queue=queue,
                start_signal="signal_stream",
            )
        else:
            operation = WithStartWorkflowOperation(
                StartHandlerStreamWorkflow.run,
                id=str(uuid4()),
                task_queue=queue,
                id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
            )
            try:
                await asyncio.wait_for(
                    client.execute_update_with_start_workflow(
                        "update_stream", start_workflow_operation=operation
                    ),
                    5,
                )
            except BaseException:
                await (await operation.workflow_handle()).terminate()
                raise
            handle = await operation.workflow_handle()
        try:
            events = await asyncio.wait_for(handle.result(), 5)
            history = await handle.fetch_history()
        except BaseException:
            await handle.terminate()
            raise
    assert "".join(events) == "HI"
    await Replayer(
        workflows=[StartHandlerStreamWorkflow], plugins=[plugin]
    ).replay_workflow(history)


async def chat(state: State) -> dict[str, str]:
    response = await FakeListChatModel(responses=["HI"]).ainvoke(state["text"])
    return {"text": response.text}


class EventIdentity(TypedDict):
    event: str
    run_id: str
    parent_ids: list[str]


@activity.defn
async def verified_ids() -> str:
    return "verified"


@workflow.defn
class ReplayStreamIdsWorkflow:
    def __init__(self) -> None:
        self.events: list[EventIdentity] = []
        self.expected: list[EventIdentity] | None = None
        self.ready = False

    @workflow.update
    async def identities(self) -> list[EventIdentity]:
        await workflow.wait_condition(lambda: self.ready)
        return self.events

    @workflow.signal
    def confirm(self, expected: list[EventIdentity]) -> None:
        self.expected = expected

    @workflow.run
    async def run(self) -> str:
        app = graph("shout").compile()
        for i in range(2):
            async for event in app.astream_events(
                {"text": "hi"},
                version="v2",
                config={"run_id": UUID(int=42)} if i == 0 else None,
            ):
                self.events.append(
                    EventIdentity(
                        event=event["event"],
                        run_id=event["run_id"],
                        parent_ids=list(event["parent_ids"]),
                    )
                )
        self.ready = True
        await workflow.wait_condition(lambda: self.expected is not None)
        if self.events == self.expected:
            return await workflow.execute_activity(
                verified_ids, start_to_close_timeout=timedelta(seconds=5)
            )
        # A replay with different IDs skips a recorded Activity: Replayer must reject it.
        return "changed IDs"


async def test_event_ids_and_parent_relationships_replay(client: Client) -> None:
    g = StateGraph(State)
    g.add_node("shout", shout, metadata={"execute_in": "activity"})
    g.add_edge(START, "shout")
    g.add_edge("shout", END)
    plugin = LangGraphPlugin(
        graphs={"shout": g},
        default_activity_options={"start_to_close_timeout": timedelta(seconds=10)},
    )
    queue = f"stable-ids-{uuid4()}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[ReplayStreamIdsWorkflow],
        activities=[verified_ids],
        plugins=[plugin],
    ):
        handle = await client.start_workflow(
            ReplayStreamIdsWorkflow.run, id=str(uuid4()), task_queue=queue
        )
        try:
            events = await asyncio.wait_for(
                handle.execute_update(ReplayStreamIdsWorkflow.identities), 10
            )
            parents: dict[str, list[str]] = {}
            for event in events:
                if event["run_id"] in parents:
                    assert parents[event["run_id"]] == event["parent_ids"]
                parents[event["run_id"]] = event["parent_ids"]
                assert event["run_id"] not in event["parent_ids"]
            assert sum(not p for p in parents.values()) == 2
            assert parents[str(UUID(int=42))] == []
            assert any(parents.values())
            assert all(parent in parents for p in parents.values() for parent in p)
            await handle.signal(ReplayStreamIdsWorkflow.confirm, events)
            assert await handle.result() == "verified"
            history = await handle.fetch_history()
        except BaseException:
            await handle.terminate()
            raise
    await Replayer(
        workflows=[ReplayStreamIdsWorkflow], plugins=[plugin]
    ).replay_workflow(history)
