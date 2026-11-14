"""Workflow-side cancellation at connection startup boundaries."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from openai_codex import TransportClosedError
from openai_codex.models import UnknownNotification

from temporalio import workflow
from temporalio.openai_codex._client import TemporalCodexClient
from temporalio.openai_codex._constants import CLOSE, CONNECT
from temporalio.openai_codex._types import Connection, Message, ReadResult


@pytest.mark.parametrize("body_fails", [False, True])
async def test_cleanup_failure_preserves_context_result(
    monkeypatch: pytest.MonkeyPatch, body_fails: bool
) -> None:
    client = TemporalCodexClient()
    client._connection = Connection("connection", "owner")

    async def start() -> None:
        pass

    async def execute(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TransportClosedError("owner unavailable")

    monkeypatch.setattr(client, "start", start)
    monkeypatch.setattr(client, "_execute", execute)

    async def run() -> str:
        async with client:
            if body_fails:
                raise ValueError("original failure")
            return "success"

    if body_fails:
        with pytest.raises(ValueError, match="original failure"):
            await run()
    else:
        assert await run() == "success"
    assert client._connection is None
    assert client._reader is None


@pytest.mark.parametrize("method", [None, 42, [], {}, "future/event", "turn/completed"])
async def test_unexpected_notification_keeps_reader_running(
    monkeypatch: pytest.MonkeyPatch, method: Any
) -> None:
    client = TemporalCodexClient()
    client._connection = Connection("connection", "owner")
    waiter = asyncio.get_running_loop().create_future()
    client._pending["request"] = waiter
    release = asyncio.Event()
    messages: list[Message] = [
        {"method": method, "params": ["changed format"]},
        {"method": "future/event", "params": {"value": "retained"}},
        {"id": "request", "result": {"value": "success"}},
    ]

    async def execute(*args: Any, **kwargs: Any) -> ReadResult:
        del args, kwargs
        if client._received:
            await release.wait()
        return ReadResult(messages)

    monkeypatch.setattr(client, "_execute", execute)
    reader = asyncio.create_task(client._read_loop())
    try:
        assert await waiter == {"value": "success"}
        assert client._router.error is None
        if isinstance(method, str):
            event = await client.next_notification()
            assert event.method == method
            assert isinstance(event.payload, UnknownNotification)
            assert event.payload.params == {}
        event = await client.next_notification()
        assert isinstance(event.payload, UnknownNotification)
        assert event.payload.params == {"value": "retained"}
        assert client._received == len(messages)
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


async def test_cancelled_start_stops_pending_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow, "in_workflow", lambda: True)
    client = TemporalCodexClient()
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def execute(
        name: str, arg: Any, config: workflow.ActivityConfig, result_type: Any = None
    ) -> Any:
        del arg, config, result_type
        assert name == CONNECT
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(client, "_execute", execute)
    starting = asyncio.create_task(client.start())
    try:
        await entered.wait()
        starting.cancel()
        done, _ = await asyncio.wait({starting}, timeout=1)
        assert starting in done, "startup kept waiting after cancellation"
        with pytest.raises(asyncio.CancelledError):
            await starting
        assert cancelled.is_set()
        assert client._connection is None
        assert client._reader is None
    finally:
        starting.cancel()
        await asyncio.gather(starting, return_exceptions=True)


async def test_cancelled_callback_can_close_during_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TemporalCodexClient()
    client._connection = Connection("reserved", "owner")
    entered = asyncio.Event()
    closed = asyncio.Event()

    async def execute(
        name: str, arg: Any, config: workflow.ActivityConfig, result_type: Any = None
    ) -> None:
        del arg, config, result_type
        assert name == CLOSE
        closed.set()

    async def callback() -> None:
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            await client.close()

    monkeypatch.setattr(client, "_execute", execute)
    task = asyncio.create_task(callback())
    client._callbacks[task] = None
    await entered.wait()
    cleanup = asyncio.create_task(client.close())
    done, pending = await asyncio.wait({task, cleanup}, timeout=1)
    assert not pending, "cleanup joined a callback that was waiting for cleanup"
    assert cleanup in done and closed.is_set()
    await cleanup
    assert task.cancelled()
