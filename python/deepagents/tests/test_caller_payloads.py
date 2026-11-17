"""Caller values keep every field through the plugin's payload converter.

With the plugin on the Client, as in the README, its payload converter writes
every payload of that Client: Workflow inputs and results, Signals, Updates,
Queries and Activity arguments, for Deep Agents Workflows and for any other.
A LangChain message's ``type`` and a ``default_factory`` value are fields that
Pydantic fills in, not fields the caller sets, so these tests send both across
those boundaries.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable
from datetime import timedelta
from typing import Any, TypeVar

import pytest

from temporalio import activity, workflow
from temporalio.client import Client, WorkflowHandle
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.converter import DataConverter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

# Resolve framework symbols through the imported module to avoid confusing
# the top-level package with this same-named test directory.
create_deep_agent = pytest.importorskip("deepagents").create_deep_agent

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
    convert_to_messages,
)
from pydantic import BaseModel, Field  # noqa: E402

with workflow.unsafe.imports_passed_through():
    from temporalio.deepagents import DeepAgentsPlugin, run_deep_agent
    from temporalio.deepagents.testing import mock_model_provider

T = TypeVar("T")


class Transcript(BaseModel):
    """A caller's model that holds LangChain messages."""

    messages: list[AnyMessage]


class Order(BaseModel):
    """A caller's model with an id that Pydantic generates."""

    item: str
    order_id: str = Field(default_factory=lambda: uuid.uuid4().hex)


class Notes(BaseModel):
    """A caller's model whose default list is filled after construction."""

    items: list[str] = Field(default_factory=list)


# One message of every class ``AnyMessage`` accepts.
MESSAGES: list[AnyMessage] = [
    AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "c1"}]),
    HumanMessage(content="find the docs"),
    ChatMessage(role="critic", content="be brief"),
    SystemMessage(content="be terse"),
    FunctionMessage(name="lookup", content="sunny"),
    ToolMessage(content="42", tool_call_id="c1"),
    AIMessageChunk(content="a"),
    HumanMessageChunk(content="h"),
    ChatMessageChunk(role="critic", content="c"),
    SystemMessageChunk(content="s"),
    FunctionMessageChunk(name="lookup", content="f"),
    ToolMessageChunk(content="t", tool_call_id="c1"),
]


def _client_with(client: Client, plugin: DeepAgentsPlugin) -> Client:
    config = client.config()
    config["plugins"] = [plugin]
    return Client(**config)


@pytest.fixture
def converter(client: Client) -> DataConverter:
    """The data converter of a Client that has the plugin."""
    return _client_with(client, DeepAgentsPlugin()).data_converter


async def _wait(handle: WorkflowHandle[Any, Any], call: Awaitable[T]) -> T:
    """Await a call on a Workflow, or fail with the task failure that blocks it."""
    try:
        return await asyncio.wait_for(call, timeout=30)
    except asyncio.TimeoutError:
        async for e in handle.fetch_history_events():
            if e.HasField("workflow_task_failed_event_attributes"):
                failure = e.workflow_task_failed_event_attributes.failure
                pytest.fail(f"workflow task is failing: {failure.message}")
        raise


@pytest.mark.asyncio
async def test_message_types_survive_roundtrip(converter: DataConverter) -> None:
    transcript = Transcript(messages=MESSAGES)

    payloads = await converter.encode([transcript])
    [restored] = await converter.decode(payloads, [Transcript])

    written = json.loads(payloads[0].data)["messages"]
    assert [m.get("type") for m in written] == [m.type for m in MESSAGES]
    assert restored == transcript


@pytest.mark.asyncio
async def test_generated_default_survives_roundtrip(converter: DataConverter) -> None:
    order = Order(item="book")
    assert "order_id" not in order.model_fields_set

    payloads = await converter.encode([order])
    [restored] = await converter.decode(payloads, [Order])

    assert json.loads(payloads[0].data)["order_id"] == order.order_id
    assert restored == order


@pytest.mark.asyncio
async def test_default_filled_in_place_survives_roundtrip(
    converter: DataConverter,
) -> None:
    notes = Notes()
    notes.items.append("first")
    assert "items" not in notes.model_fields_set

    payloads = await converter.encode([notes])
    [restored] = await converter.decode(payloads, [Notes])

    assert restored == notes


@pytest.mark.asyncio
async def test_writes_what_the_sdk_pydantic_converter_writes(
    converter: DataConverter,
) -> None:
    values = [Transcript(messages=MESSAGES), Order(item="book")]

    assert await converter.encode(values) == await pydantic_data_converter.encode(
        values
    )


@workflow.defn
class TranscriptWorkflow:
    """Takes and returns messages in every handler. No agent is involved."""

    def __init__(self) -> None:
        self._signaled: list[AnyMessage] = []
        self._updated: list[AnyMessage] | None = None

    @workflow.run
    async def run(self, transcript: Transcript) -> Transcript:
        await workflow.wait_condition(lambda: self._updated is not None)
        return Transcript(
            messages=[*transcript.messages, *self._signaled, *(self._updated or [])]
        )

    @workflow.signal
    def add(self, transcript: Transcript) -> None:
        self._signaled.extend(transcript.messages)

    @workflow.query
    def signaled(self) -> Transcript:
        return Transcript(messages=self._signaled)

    @workflow.update
    def finish(self, transcript: Transcript) -> Transcript:
        self._updated = transcript.messages
        return transcript


@pytest.mark.asyncio
async def test_messages_keep_their_type_in_every_handler(
    env: WorkflowEnvironment,
) -> None:
    """Workflow input and result, Signal, Query and Update carry typed messages."""
    client = _client_with(env.client, DeepAgentsPlugin())
    by_input, by_signal, by_update = MESSAGES[:4], MESSAGES[4:8], MESSAGES[8:]
    async with Worker(
        client,
        task_queue="da-payloads-handlers",
        workflows=[TranscriptWorkflow],
    ):
        handle = await client.start_workflow(
            TranscriptWorkflow.run,
            Transcript(messages=by_input),
            id=f"da-payloads-handlers-{uuid.uuid4()}",
            task_queue="da-payloads-handlers",
        )
        await handle.signal(TranscriptWorkflow.add, Transcript(messages=by_signal))
        updated = await _wait(
            handle,
            handle.execute_update(
                TranscriptWorkflow.finish, Transcript(messages=by_update)
            ),
        )
        queried = await handle.query(TranscriptWorkflow.signaled)
        result = await _wait(handle, handle.result())

    assert queried.messages == by_signal
    assert updated.messages == by_update
    assert result.messages == MESSAGES


@workflow.defn
class LongResearchAgent:
    """The README's continue-as-new example, which returns the agent's state."""

    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
        agent = create_deep_agent(name="research", model="anthropic:claude-sonnet-5-5")
        return await run_deep_agent(
            agent,
            input,
            state_snapshot=state_snapshot,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [{"role": "user", "content": "Research."}, HumanMessage(content="Research.")],
    ids=["dict", "HumanMessage"],
)
async def test_readme_agent_keeps_message_types(
    env: WorkflowEnvironment, question: Any
) -> None:
    plugin = DeepAgentsPlugin(model_provider=mock_model_provider(["done"]))
    client = _client_with(env.client, plugin)
    async with Worker(
        client,
        task_queue="da-payloads-readme",
        workflows=[LongResearchAgent],
    ):
        handle = await client.start_workflow(
            LongResearchAgent.run,
            {"messages": [question]},
            id=f"da-payloads-readme-{uuid.uuid4()}",
            task_queue="da-payloads-readme",
        )
        result = await _wait(handle, handle.result())

    # The result type is ``dict``, so the messages arrive as dicts. Each one
    # names its type, which LangChain needs to rebuild the message.
    messages = convert_to_messages(result["messages"])
    assert [type(m) for m in messages] == [HumanMessage, AIMessage]
    assert [m.content for m in messages] == ["Research.", "done"]


@workflow.defn
class ChatAgent:
    """Continues as new between turns and passes the transcript as a snapshot."""

    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> list[str]:
        agent = create_deep_agent(name="chat", model="fake:model")
        result = await run_deep_agent(agent, input, state_snapshot=state_snapshot)
        if state_snapshot is None:
            workflow.continue_as_new(
                args=[
                    {"messages": [{"role": "user", "content": "second question"}]},
                    {"messages": result["messages"]},
                ]
            )
        return [f"{m.type}:{m.content}" for m in result["messages"]]


@pytest.mark.asyncio
async def test_caller_built_snapshot_keeps_message_types(
    env: WorkflowEnvironment,
) -> None:
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(["first answer", "second answer"])
    )
    client = _client_with(env.client, plugin)
    async with Worker(
        client,
        task_queue="da-payloads-snapshot",
        workflows=[ChatAgent],
    ):
        handle = await client.start_workflow(
            ChatAgent.run,
            {"messages": [{"role": "user", "content": "first question"}]},
            id=f"da-payloads-snapshot-{uuid.uuid4()}",
            task_queue="da-payloads-snapshot",
        )
        result = await _wait(handle, handle.result())

    assert result == [
        "human:first question",
        "ai:first answer",
        "human:second question",
        "ai:second answer",
    ]


@activity.defn
async def charge(order: Order) -> str:
    return order.order_id


@workflow.defn
class CheckoutWorkflow:
    """Uses the order's generated id as the Activity id."""

    @workflow.run
    async def run(self, order: Order) -> list[str]:
        charged = await workflow.execute_activity(
            charge,
            order,
            activity_id=f"charge-{order.order_id}",
            start_to_close_timeout=timedelta(seconds=10),
        )
        return [order.order_id, charged]


@pytest.mark.asyncio
async def test_generated_default_is_the_same_on_every_side(
    env: WorkflowEnvironment,
) -> None:
    """Client, Workflow and Activity see one id, and the History replays."""
    client = _client_with(env.client, DeepAgentsPlugin())
    order = Order(item="book")
    async with Worker(
        client,
        task_queue="da-payloads-order",
        workflows=[CheckoutWorkflow],
        activities=[charge],
    ):
        handle = await client.start_workflow(
            CheckoutWorkflow.run,
            order,
            id=f"da-payloads-order-{uuid.uuid4()}",
            task_queue="da-payloads-order",
        )
        result = await _wait(handle, handle.result())
        history = await handle.fetch_history()

    assert result == [order.order_id, order.order_id]
    # Replay decodes the input again, so the id has to come from the payload.
    replayer = Replayer(workflows=[CheckoutWorkflow], plugins=[DeepAgentsPlugin()])
    await replayer.replay_workflow(history)
