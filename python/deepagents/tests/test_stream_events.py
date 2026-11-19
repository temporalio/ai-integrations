"""Deep Agent event streams finish and replay using a local model."""

import asyncio
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from typing_extensions import TypedDict

from temporalio import activity, workflow
from temporalio.client import WithStartWorkflowOperation
from temporalio.common import WorkflowIDConflictPolicy
from temporalio.deepagents import DeepAgentsPlugin
from temporalio.deepagents.testing import FakeModel, mock_model_provider
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

with workflow.unsafe.imports_passed_through():
    from deepagents import create_deep_agent
    from langchain_core.messages import BaseMessage


@workflow.defn
class StreamEventsWorkflow:
    @workflow.run
    async def run(self, mode: str) -> list[str]:
        agent = create_deep_agent(model="anthropic:claude-sonnet-4-5")
        if mode == "messages":
            messages: list[str] = []
            async for message, _ in agent.astream(
                {"messages": [{"role": "user", "content": "What is 6 * 7?"}]},
                stream_mode="messages",
            ):
                assert isinstance(message, BaseMessage)
                messages.append(message.text)
            return messages
        return [
            event["event"]
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": "What is 6 * 7?"}]},
                version="v2",
            )
        ]


@pytest.mark.parametrize("mode", ["events", "messages"])
async def test_stream_events_completes_and_replays(
    env: WorkflowEnvironment, mode: str
) -> None:
    agent = create_deep_agent(model=FakeModel(["42"]))
    expected = [
        event["event"]
        async for event in agent.astream_events(
            {"messages": [{"role": "user", "content": "What is 6 * 7?"}]},
            version="v2",
        )
    ]
    plugin = DeepAgentsPlugin(model_provider=mock_model_provider(["42"]))
    queue = f"stream-events-{uuid4()}"
    async with Worker(
        env.client, task_queue=queue, workflows=[StreamEventsWorkflow], plugins=[plugin]
    ):
        handle = await env.client.start_workflow(
            StreamEventsWorkflow.run, mode, id=str(uuid4()), task_queue=queue
        )
        try:
            result = await asyncio.wait_for(handle.result(), 10)
            history = await handle.fetch_history()
        except BaseException:
            await handle.terminate()
            raise
    if mode == "messages":
        assert "".join(result) == "42"
    else:
        assert_event_lifecycle(result, expected)
    await Replayer(workflows=[StreamEventsWorkflow], plugins=[plugin]).replay_workflow(
        history
    )


def assert_event_lifecycle(result: list[str], expected: list[str]) -> None:
    # TemporalModel exposes Activity output as chunks, so it adds stream events
    # that FakeModel outside a Workflow does not emit. Lifecycle events agree.
    assert [event for event in result if not event.endswith("_stream")] == [
        event for event in expected if not event.endswith("_stream")
    ]
    assert "on_chat_model_stream" in result


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
        agent = create_deep_agent(model="anthropic:claude-sonnet-4-5")
        events: list[str] = []
        async for message, _ in agent.astream(
            {"messages": [{"role": "user", "content": "What is 6 * 7?"}]},
            stream_mode="messages",
        ):
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
async def test_streams_from_start_handlers(
    env: WorkflowEnvironment, handler: str
) -> None:
    client = env.client
    plugin = DeepAgentsPlugin(model_provider=mock_model_provider(["42"]))
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
    assert "".join(events) == "42"
    await Replayer(
        workflows=[StartHandlerStreamWorkflow], plugins=[plugin]
    ).replay_workflow(history)


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
        agent = create_deep_agent(model="anthropic:claude-sonnet-4-5")
        for i in range(2):
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": "What is 6 * 7?"}]},
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


async def test_event_ids_and_parent_relationships_replay(
    env: WorkflowEnvironment,
) -> None:
    client = env.client
    plugin = DeepAgentsPlugin(model_provider=mock_model_provider(["42", "42"]))
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
