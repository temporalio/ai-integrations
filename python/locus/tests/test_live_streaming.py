"""Chunks reach outside subscribers while the model is still answering."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

from locus.agent import Agent
from locus.core.events import ModelChunkEvent
from locus.core.messages import Message
from locus.models.base import ModelResponse

from temporalio import workflow
from temporalio.client import Client
from temporalio.contrib.workflow_streams import WorkflowStream, WorkflowStreamClient
from temporalio.locus import LocusPlugin, TemporalModel
from temporalio.worker import Replayer, Worker

WORDS = ["The ", "refund ", "is ", "on ", "its ", "way."]


class GatedModel:
    """Streams two words, then waits for ``gate`` before the rest."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        del messages, tools, kwargs
        return ModelResponse(message=Message.assistant(content="".join(WORDS)))

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        del messages, tools, kwargs
        for index, word in enumerate(WORDS):
            if index == 2:
                await self.gate.wait()
            yield ModelChunkEvent(content=word)
        yield ModelChunkEvent(done=True)


@workflow.defn
class LiveWorkflow:
    def __init__(self) -> None:
        self.stream = WorkflowStream()

    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            model=TemporalModel(
                "m",
                start_to_close_timeout=timedelta(seconds=30),
                streaming_topic="tokens",
                streaming_batch_interval=timedelta(milliseconds=50),
            )
        )
        text = ""
        async for event in agent.run(prompt, stream_tokens=True):
            if isinstance(event, ModelChunkEvent) and event.content:
                text += event.content
        return text


async def test_subscriber_sees_chunks_before_the_model_call_ends(
    client: Client,
) -> None:
    model = GatedModel()
    plugin = LocusPlugin(models={"m": lambda: model})
    config = client.config()
    config["plugins"] = [plugin]
    client = Client(**config)
    task_queue = f"locus-{uuid.uuid4()}"
    workflow_id = f"live-{uuid.uuid4()}"
    received: list[str] = []
    two_arrived = asyncio.Event()

    async def subscribe() -> None:
        stream = WorkflowStreamClient.create(client, workflow_id)
        async for item in stream.subscribe(
            ["tokens"],
            from_offset=0,
            result_type=ModelChunkEvent,
            poll_cooldown=timedelta(milliseconds=20),
        ):
            if item.data.content:
                received.append(item.data.content)
            if len(received) == 2:
                two_arrived.set()
            if item.data.done:
                return

    async with Worker(client, task_queue=task_queue, workflows=[LiveWorkflow]):
        handle = await client.start_workflow(
            LiveWorkflow.run,
            "where is my refund?",
            id=workflow_id,
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        subscriber = asyncio.create_task(subscribe())
        # The first two words arrive while the model call is still open.
        await asyncio.wait_for(two_arrived.wait(), timeout=20)
        assert received == WORDS[:2]
        assert not model.gate.is_set()
        description = await handle.describe()
        assert description.status is not None and description.status.name == "RUNNING"

        model.gate.set()
        assert await handle.result() == "".join(WORDS)
        await asyncio.wait_for(subscriber, timeout=20)
    assert received == WORDS

    # What subscribers saw is not what the workflow replays from.
    await Replayer(workflows=[LiveWorkflow], plugins=[plugin]).replay_workflow(
        await handle.fetch_history()
    )


@workflow.defn
class NoStreamHostedWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            model=TemporalModel(
                "m",
                start_to_close_timeout=timedelta(seconds=30),
                streaming_topic="tokens",
                streaming_batch_interval=timedelta(milliseconds=50),
            )
        )
        text = ""
        async for event in agent.run(prompt, stream_tokens=True):
            if isinstance(event, ModelChunkEvent) and event.content:
                text += event.content
        return text


async def test_run_is_unaffected_when_no_stream_is_hosted(client: Client) -> None:
    model = GatedModel()
    model.gate.set()
    config = client.config()
    config["plugins"] = [LocusPlugin(models={"m": lambda: model})]
    client = Client(**config)
    task_queue = f"locus-{uuid.uuid4()}"
    async with Worker(
        client, task_queue=task_queue, workflows=[NoStreamHostedWorkflow]
    ):
        result = await client.execute_workflow(
            NoStreamHostedWorkflow.run,
            "where is my refund?",
            id=f"no-stream-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
    assert result == "".join(WORDS)
