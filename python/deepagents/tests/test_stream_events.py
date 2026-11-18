"""Deep Agent event streams finish and replay using a local model."""

import asyncio
from uuid import uuid4

import pytest

from temporalio import workflow
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
