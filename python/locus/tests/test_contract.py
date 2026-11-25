"""Values crossing the Activity boundary, and cancellation reaching the model call."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest
from locus.agent import Agent
from locus.core.events import ModelChunkEvent
from locus.core.messages import Message, Role, ToolCall
from locus.models.base import ModelResponse

from temporalio import activity, workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.exceptions import ApplicationError, CancelledError
from temporalio.locus import LocusPlugin, TemporalModel, activity_as_tool
from temporalio.worker import Worker

ARGUMENTS: dict[str, Any] = {
    "order": {"id": "ORD-1", "lines": [{"sku": "α-β", "qty": 2, "price": 9.5}]},
    "note": 'café ☕ "quoted"\nnew line',
    "flags": [True, False, None],
}


class RecordingModel:
    """Records what reaches the worker, and returns a reply with every field set."""

    def __init__(self) -> None:
        self.seen_options: list[dict[str, Any]] = []
        self.seen_tools: list[list[dict[str, Any]] | None] = []
        self.seen_messages: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.seen_options.append(kwargs)
        self.seen_tools.append(tools)
        self.seen_messages.append(messages)
        if not [m for m in messages if m.role == Role.TOOL]:
            return ModelResponse(
                message=Message.assistant(
                    content=None,
                    tool_calls=[
                        ToolCall(id="call_1", name="record", arguments=ARGUMENTS)
                    ],
                ),
                usage={"prompt_tokens": 11, "completion_tokens": 7},
                stop_reason="tool_calls",
                reasoning="think ∴ act",
                provider_state={"previous_response_id": "resp_1"},
            )
        return ModelResponse(
            message=Message.assistant(content="ok"), stop_reason="stop"
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        response = await self.complete(messages, tools, **kwargs)
        yield ModelChunkEvent(content=response.message.content, done=True)


RECORDED: list[dict[str, Any]] = []


@activity.defn
async def record(order: dict[str, Any], note: str, flags: list[bool | None]) -> str:
    """Record an order."""
    RECORDED.append({"order": order, "note": note, "flags": flags})
    return "recorded"


def _client(client: Client, model: Any) -> Client:
    config = client.config()
    config["plugins"] = [LocusPlugin(models={"m": lambda: model})]
    return Client(**config)


@workflow.defn
class RecordWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            model=TemporalModel("m", start_to_close_timeout=timedelta(seconds=30)),
            tools=[
                activity_as_tool(record, start_to_close_timeout=timedelta(seconds=30))
            ],
            system_prompt="système ✓",
        )
        result = await agent.arun(
            prompt,
            model_kwargs={"top_p": 0.5, "stop": ["END"], "extra_body": {"seed": 7}},
        )
        return result.message or ""


async def test_values_survive_the_payload_converter(client: Client) -> None:
    model = RecordingModel()
    RECORDED.clear()
    client = _client(client, model)
    task_queue = f"locus-{uuid.uuid4()}"
    async with Worker(
        client, task_queue=task_queue, workflows=[RecordWorkflow], activities=[record]
    ):
        result = await client.execute_workflow(
            RecordWorkflow.run,
            "enregistre ✓",
            id=f"record-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
    assert result == "ok"
    # The model's tool-call arguments reach the tool Activity unchanged.
    assert RECORDED == [ARGUMENTS]
    # The agent's model options reach the worker's model unchanged.
    for options in model.seen_options:
        assert options["top_p"] == 0.5
        assert options["stop"] == ["END"]
        assert options["extra_body"] == {"seed": 7}
    # The tool's schema reaches the model, named after the Activity.
    first_tools = model.seen_tools[0]
    assert first_tools is not None
    assert [t["function"]["name"] for t in first_tools] == ["record"]
    assert set(first_tools[0]["function"]["parameters"]["properties"]) == {
        "order",
        "note",
        "flags",
    }
    # The second call's messages are typed Locus messages carrying the first reply.
    second = model.seen_messages[1]
    assert all(isinstance(m, Message) for m in second)
    assert second[0].content == "système ✓"
    assert second[1].content == "enregistre ✓"
    assistant = next(m for m in second if m.role == Role.ASSISTANT)
    assert assistant.tool_calls[0].arguments == ARGUMENTS
    assert next(m for m in second if m.role == Role.TOOL).content == "recorded"


@workflow.defn
class BadOptionWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            model=TemporalModel("m", start_to_close_timeout=timedelta(seconds=30)),
        )
        result = await agent.arun(prompt, model_kwargs={"on_token": lambda token: None})
        return result.message or ""


async def test_unserializable_model_option_fails_the_run_clearly(
    client: Client,
) -> None:
    model = RecordingModel()
    client = _client(client, model)
    task_queue = f"locus-{uuid.uuid4()}"
    async with Worker(client, task_queue=task_queue, workflows=[BadOptionWorkflow]):
        with pytest.raises(WorkflowFailureError) as raised:
            await client.execute_workflow(
                BadOptionWorkflow.run,
                "hello",
                id=f"bad-option-{uuid.uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=20),
            )
    cause = raised.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "LocusUnserializableModelOption"
    assert model.seen_options == []


class HangingModel:
    """Never answers; records that its call was cancelled."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        del messages, tools, kwargs
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        await self.complete(messages, tools, **kwargs)
        yield ModelChunkEvent(done=True)


@workflow.defn
class HangingWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            model=TemporalModel(
                "m",
                start_to_close_timeout=timedelta(seconds=60),
                heartbeat_timeout=timedelta(seconds=1),
            )
        )
        return (await agent.arun(prompt)).message or ""


async def test_cancelling_the_workflow_cancels_the_running_model_call(
    client: Client,
) -> None:
    model = HangingModel()
    client = _client(client, model)
    task_queue = f"locus-{uuid.uuid4()}"
    async with Worker(client, task_queue=task_queue, workflows=[HangingWorkflow]):
        handle = await client.start_workflow(
            HangingWorkflow.run,
            "hello",
            id=f"hang-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        await asyncio.wait_for(model.started.wait(), timeout=20)
        # The call outlives several heartbeat timeouts: the timer heartbeats.
        await asyncio.sleep(3)
        assert not model.cancelled.is_set()
        await handle.cancel()
        with pytest.raises(WorkflowFailureError) as raised:
            await handle.result()
        assert isinstance(raised.value.cause, CancelledError)
        # The heartbeat delivers the cancel to the model call on the worker.
        await asyncio.wait_for(model.cancelled.wait(), timeout=20)
