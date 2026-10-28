"""The workflow side of the tool bridge: rejected submits and attempt rotation."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from temporalio import activity
from temporalio.claude_agent_sdk._models import ClaudeToolRequest, ClaudeToolResult
from temporalio.claude_agent_sdk._session import ClaudeSession, _BridgeHub, _ToolBridge
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

from .concurrent_tool_cancellation import ConcurrentToolCancellation
from .helpers.histories import record
from .helpers.workflows import get_weather


def _cleanup_activity(number: int) -> Any:
    @activity.defn(name=f"cleanup_{number}")
    async def cleanup(index: int) -> int:
        return index

    return cleanup


async def test_concurrent_tool_cancellation_replays(env: WorkflowEnvironment) -> None:
    """Concurrent tools schedule their cleanup commands in submission order on replay too."""
    queue = f"cancel-tools-{uuid.uuid4()}"
    async with Worker(
        env.client,
        task_queue=queue,
        workflows=[ConcurrentToolCancellation],
        activities=[_cleanup_activity(i) for i in range(6)],
    ):
        handle = await env.client.start_workflow(
            ConcurrentToolCancellation.run, id=queue, task_queue=queue
        )
        await asyncio.wait_for(handle.result(), 15)
        history = await handle.fetch_history()
    scheduled = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField("activity_task_scheduled_event_attributes")
    ]
    assert scheduled == [f"cleanup_{i}" for i in range(6)]
    await record("concurrent_tool_cancellation", history)
    await Replayer(workflows=[ConcurrentToolCancellation]).replay_workflow(history)


async def _wait_until(fn: Callable[[], bool], **_: object) -> None:
    """Stand in for ``workflow.wait_condition`` outside a workflow runtime."""
    loop = asyncio.get_running_loop()
    ready: asyncio.Future[None] = loop.create_future()

    def poll() -> None:
        if ready.done():
            return
        if fn():
            ready.set_result(None)
        else:
            loop.call_soon(poll)

    poll()
    await ready


async def test_rotated_attempt_attaches_to_the_in_flight_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def lookup(q: str) -> str:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return f"picked {q}"

    session = ClaudeSession(session_id="sess")
    bridge = _ToolBridge("token-a", {"lookup": lookup}, 4, session, "msg")
    hub = _BridgeHub()
    hub.bridges[bridge.token] = bridge

    monkeypatch.setattr(_BridgeHub, "current", lambda: hub)
    monkeypatch.setattr(
        "temporalio.claude_agent_sdk._session.workflow.uuid4", uuid.uuid4
    )
    monkeypatch.setattr(
        "temporalio.claude_agent_sdk._session.workflow.wait_condition", _wait_until
    )

    def request(tool_use_id: str) -> ClaudeToolRequest:
        return ClaudeToolRequest(
            token=bridge.token,
            tool_use_id=tool_use_id,
            name="lookup",
            arguments={"q": "paris"},
            ordinal=0,
        )

    bridge.submit(request("A"))
    await started.wait()
    lost = bridge.token
    bridge.rotate()
    assert bridge.token != lost
    assert hub.bridges[bridge.token] is bridge
    assert lost not in hub.bridges

    bridge.submit(request("B"))
    assert calls == 1

    release.set()
    finished, pending = await asyncio.wait(tuple(bridge._tasks))
    assert not pending
    for task in finished:
        task.result()

    assert calls == 1
    by_id = {item.tool_use_id: item for item in bridge.results}
    assert set(by_id) == {"A", "B"}
    assert by_id["A"].content == by_id["B"].content == "picked paris"
    assert by_id["A"].is_error is False
    assert by_id["B"].is_error is False
    assert len(session.state.tool_record) == 1
    done = session.state.tool_record[0]
    assert done.tool_use_id == "A"
    assert done.aliases == ["B"]
    assert done.result is not None
    assert done.result.content == "picked paris"
    assert done.result.is_error is False


_TOKEN = "bridge-1"


def _bridge(tool: Callable[..., Awaitable[Any]], *, max_in_flight: int) -> _ToolBridge:
    return _ToolBridge(
        token=_TOKEN,
        tools={"get_weather": tool},
        max_in_flight=max_in_flight,
        session=ClaudeSession("sess-1"),
        message_id="msg-1",
    )


def _request(tool_use_id: str, city: object) -> ClaudeToolRequest:
    return ClaudeToolRequest(
        token=_TOKEN,
        tool_use_id=tool_use_id,
        name="get_weather",
        arguments={"city": city},
    )


async def test_a_duplicate_tool_use_id_is_an_mcp_error_and_does_not_run() -> None:
    executed: list[str] = []

    async def get_weather_for_turn(city: str) -> str:
        executed.append(city)
        return await get_weather(city)

    bridge = _bridge(get_weather_for_turn, max_in_flight=2)
    bridge.submit(_request("toolu_dup", "Paris"))
    bridge.submit(_request("toolu_dup", "Lyon"))
    await asyncio.gather(*tuple(bridge._tasks))

    assert executed == ["Paris"]
    assert [result for result in bridge.results if result.is_error] == [
        ClaudeToolResult(
            tool_use_id="toolu_dup",
            content="duplicate tool call 'toolu_dup'",
            is_error=True,
        )
    ]


async def test_the_same_call_under_its_recorded_id_is_answered_from_the_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost attempt's dangling call, asked about under its own id with an ordinal that may
    not match its slot, attaches to the record instead of being rejected or run again."""
    monkeypatch.setattr(
        "temporalio.claude_agent_sdk._session.workflow.wait_condition", _wait_until
    )
    executed: list[str] = []

    async def get_weather_for_turn(city: str) -> str:
        executed.append(city)
        return await get_weather(city)

    bridge = _bridge(get_weather_for_turn, max_in_flight=2)
    bridge.submit(_request("toolu_a", "Paris"))
    await asyncio.gather(*tuple(bridge._tasks))
    bridge.submit(_request("toolu_a", "Paris").model_copy(update={"ordinal": 3}))
    await asyncio.gather(*tuple(bridge._tasks))

    assert executed == ["Paris"]
    assert [r.is_error for r in bridge.results] == [False, False]


async def test_a_call_past_max_concurrent_tool_calls_is_an_mcp_error_and_does_not_run() -> (
    None
):
    executed: list[str] = []

    async def get_weather_for_turn(city: str) -> str:
        executed.append(city)
        return await get_weather(city)

    bridge = _bridge(get_weather_for_turn, max_in_flight=1)
    bridge.submit(_request("toolu_ok", "Paris"))
    bridge.submit(_request("toolu_over", "Lyon"))
    await asyncio.gather(*tuple(bridge._tasks))

    assert executed == ["Paris"]
    assert [result for result in bridge.results if result.is_error] == [
        ClaudeToolResult(
            tool_use_id="toolu_over",
            content="too many concurrent tool calls (limit 1); retry later",
            is_error=True,
        )
    ]


async def test_invalid_tool_arguments_are_an_mcp_error_and_do_not_run() -> None:
    executed: list[str] = []

    async def get_weather_for_turn(city: str) -> str:
        executed.append(city)
        return await get_weather(city)

    bridge = _bridge(get_weather_for_turn, max_in_flight=1)
    bridge.submit(_request("toolu_bad", 12))
    await asyncio.gather(*tuple(bridge._tasks))

    assert executed == []
    assert bridge.results == [
        ClaudeToolResult(
            tool_use_id="toolu_bad",
            content=(
                "invalid arguments for 'get_weather': 1 validation error for str\n"
                "  Input should be a valid string [type=string_type, "
                "input_value=12, input_type=int]\n"
                "    For further information visit "
                "https://errors.pydantic.dev/2.13/v/string_type"
            ),
            is_error=True,
        )
    ]
