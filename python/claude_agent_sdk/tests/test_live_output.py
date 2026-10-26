"""Live output: the agent's events through Workflow Streams, in order, across Continue-As-New."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from temporalio.claude_agent_sdk import (
    AgentState,
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    DurableClaudeAgent,
    FileSessionStore,
    SegmentRunner,
    follow_agent,
)
from temporalio.claude_agent_sdk._events import FIELD_LIMIT, TOPIC, cap_event
from temporalio.claude_agent_sdk._workflow import _HANDOVER_BYTES
from temporalio.claude_agent_sdk.testing import ScriptedClaude
from temporalio.client import Client
from temporalio.contrib.workflow_streams import PublisherState, WorkflowStreamState
from temporalio.contrib.workflow_streams._types import _WorkflowStreamWireItem
from temporalio.converter import default
from temporalio.worker import Worker
from tests.endless.activities import ALL
from tests.endless.policy import count_policy
from tests.endless.workflows import LongTaskWorkflow, TaskOptions
from tests.helpers.fake_messages_api import engine_env, start_with_policy
from tests.helpers.outside_workflow import CarriedStream
from tests.helpers.workers import FAIL_FAST

pytestmark = pytest.mark.timeout(180)


async def read_until_done(
    client: Client, workflow_id: str, from_offset: int = 0
) -> list[dict[str, Any]]:
    """Every event from ``from_offset`` through ``done`` (or the end of the stream)."""
    events: list[dict[str, Any]] = []
    async for event in follow_agent(client, workflow_id, from_offset=from_offset):
        events.append(event)
        if event["type"] in ("done", "error"):
            break
    return events


def worker(client: Client, queue: str, runner: SegmentRunner) -> Worker:
    """A Worker for the live agents."""
    return Worker(
        client,
        task_queue=queue,
        workflows=[LongTaskWorkflow],
        activities=ALL,
        plugins=[ClaudeAgentPlugin(runner, heartbeat_every=1.0)],
        **FAIL_FAST,
    )


def check_sequence(events: list[dict[str, Any]], steps: int) -> None:
    """The events tell the whole story, once each, in order."""
    offsets = [e["offset"] for e in events]
    assert offsets == list(
        range(offsets[0], offsets[0] + len(events))
    )  # no gap, no duplicate
    kinds = [e["type"] for e in events]
    assert kinds[0] == "prompt" and kinds[-1] == "done"
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert [c["input"]["n"] for c in calls] == list(range(1, steps + 1))
    assert [r["status"] for r in results] == ["done"] * steps
    for call in calls:  # each call's result comes after it
        assert kinds.index("tool_result", events.index(call)) > events.index(call)
    assert {"type": "text", "text": f"counted to {steps}"}.items() <= next(
        e for e in events if e["type"] == "text"
    ).items()
    assert events[-1]["result"] == f"counted to {steps}"


@pytest.mark.usefixtures("shop_dir")
async def test_live_events_arrive_in_order_across_continue_as_new(
    client: Client,
) -> None:
    queue = f"live-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(count_policy)):
        handle = await client.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 12", TaskOptions(after_events=60, live=True), None],
            id=queue,
            task_queue=queue,
        )
        events = await asyncio.wait_for(read_until_done(client, queue), 120)
        assert await handle.result() == "counted to 12"
    check_sequence(events, 12)
    assert any(
        e["type"] == "continued_as_new" for e in events
    )  # it followed the new runs


@pytest.mark.usefixtures("shop_dir")
async def test_a_subscriber_that_reconnects_misses_nothing(client: Client) -> None:
    queue = f"reconnect-{uuid.uuid4().hex[:8]}"
    async with worker(
        client,
        queue,
        ScriptedClaude(count_policy, think_seconds=0.05),
    ):
        await client.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 10", TaskOptions(after_events=60, live=True), None],
            id=queue,
            task_queue=queue,
        )
        first: list[dict[str, Any]] = []
        async for event in follow_agent(client, queue):  # disconnect after 7 events
            first.append(event)
            if len(first) == 7:
                break
        rest = await asyncio.wait_for(
            read_until_done(client, queue, first[-1]["offset"] + 1), 120
        )
    check_sequence(first + rest, 10)


@pytest.mark.timeout(240)
@pytest.mark.usefixtures("shop_dir")
async def test_real_engine_text_streams_live(client: Client, tmp_path: Path) -> None:
    api = start_with_policy(count_policy)
    (tmp_path / "work").mkdir()
    runner = ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "sessions"),
        cwd=str(tmp_path / "work"),
        env=engine_env(api, str(tmp_path / "cfg")),
    )
    queue = f"live-real-{uuid.uuid4().hex[:8]}"
    try:
        async with worker(client, queue, runner):
            await client.start_workflow(
                LongTaskWorkflow.run,
                args=["count to 3", TaskOptions(after_events=40, live=True), None],
                id=queue,
                task_queue=queue,
            )
            events = await asyncio.wait_for(read_until_done(client, queue), 200)
    finally:
        api.stop()
    check_sequence(
        events, 3
    )  # the text event came from the engine, through the Activity
    assert api.errors == [] and runner.stub_calls == 0


# ---- what an event and the carried stream hold ----


@pytest.mark.parametrize(
    "event",
    [
        {"type": "text", "text": "\u00e9" * (FIELD_LIMIT + 5)},
        {"type": "tool_call", "input": {"command": "x" * FIELD_LIMIT}},
        {"type": "tool_call", "input": {"q": '"\\' * FIELD_LIMIT}},
        {"type": "tool_call", "input": "y" * (FIELD_LIMIT + 1)},
        {"type": "tool_result", "result": "short", "error": None},
    ],
    ids=["long-text", "long-input", "escaped-input", "text-input", "short"],
)
def test_capping_a_capped_event_changes_nothing(event: dict[str, Any]) -> None:
    once = cap_event(event)
    assert cap_event(once) == once


def test_an_input_that_is_text_and_fits_stays_as_it_is() -> None:
    """Its JSON is two quotes longer: before, that cut it, with a quote in front."""
    event = {"type": "tool_call", "input": "z" * FIELD_LIMIT}
    assert cap_event(event) == event


def input_bytes(args: list[Any]) -> int:
    """A run's input size, measured as the Workflow measures it (default converter)."""
    return sum(p.ByteSize() for p in default().payload_converter.to_payloads(args))


def agent_carrying(
    monkeypatch: pytest.MonkeyPatch,
    conversation: list[str],
    keep_bytes: int,
    publishers: int,
) -> tuple[DurableClaudeAgent, list[Any], list[int]]:
    """An agent with 5,000 small events in its stream, the way a Workflow keeps them.

    Returns the agent, the events, and a list that counts the input's measurements.
    """
    converter = default().payload_converter
    log = [
        _WorkflowStreamWireItem(
            topic=TOPIC,
            data=base64.b64encode(
                converter.to_payload(
                    {"type": "tool_result", "id": f"toolu_{i:06d}", "status": "done"}
                ).SerializeToString()
            ).decode("ascii"),
        )
        for i in range(5000)
    ]
    seen = {
        f"publisher-{i:02d}": PublisherState(
            sequence=i, last_seen=datetime(2026, 10, 2, tzinfo=timezone.utc)
        )
        for i in range(publishers)
    }
    agent = DurableClaudeAgent(
        state=AgentState(conversation=conversation),
        live_output_keep=len(log),
        live_output_keep_bytes=keep_bytes,
    )
    agent._stream = cast(  # type: ignore[reportPrivateUsage]
        Any,
        CarriedStream(WorkflowStreamState(log=log, base_offset=1000, publishers=seen)),
    )
    measured: list[int] = []

    def measure(state: AgentState) -> int:
        measured.append(1)
        return input_bytes(agent._new_run_args(state))  # type: ignore[reportPrivateUsage]

    monkeypatch.setattr(agent, "_input_bytes", measure)
    return agent, log, measured


def test_continue_as_new_carries_the_newest_events_that_fit_the_new_runs_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured as the new run's input carries the stream (each event's JSON, the
    stream's publishers): the input stays within its share, and one more event would
    not have fit. Before, each event counted as its data only."""
    conversation = ["x" * 1024] * 1200  # most of the new run's input
    agent, log, measured = agent_carrying(monkeypatch, conversation, 1024**2, 40)
    state = agent.state()
    total = agent._carry_stream(state, agent._input_bytes(state))  # type: ignore[reportPrivateUsage]
    assert state.stream is not None
    kept = len(state.stream.log)
    assert 0 < kept < len(log) and state.stream.log == log[-kept:]
    assert total == input_bytes([state.task_prompt, state]) <= _HANDOVER_BYTES
    one_more = dataclasses.replace(state.stream, log=log[-kept - 1 :])
    with_one_more = dataclasses.replace(state, stream=one_more)
    assert input_bytes([state.task_prompt, with_one_more]) > _HANDOVER_BYTES
    assert len(measured) > 2  # the publishers were not in the first estimate


def test_the_carried_stream_keeps_to_live_output_keep_bytes_as_carried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = 64 * 1024
    agent, log, _ = agent_carrying(monkeypatch, [], budget, 0)
    state = agent.state()
    agent._carry_stream(state, agent._input_bytes(state))  # type: ignore[reportPrivateUsage]
    assert state.stream is not None and 0 < len(state.stream.log) < len(log)
    empty = dataclasses.replace(state, stream=dataclasses.replace(state.stream, log=[]))
    carried = input_bytes([state.task_prompt, state]) - input_bytes(
        [state.task_prompt, empty]
    )
    assert budget - 300 < carried <= budget
