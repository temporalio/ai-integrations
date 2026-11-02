"""Workflow-side cancellation at connection startup boundaries."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from temporalio import workflow
from temporalio.openai_codex import TemporalCodexClient
from temporalio.openai_codex._constants import CLOSE, CONNECT, RESERVE
from temporalio.openai_codex._types import CloseActivityInput, Connection


@pytest.mark.parametrize("stage", [RESERVE, CONNECT])
@pytest.mark.parametrize("late_result", [False, True])
async def test_cancelled_start_stops_pending_operation(
    monkeypatch: pytest.MonkeyPatch, stage: str, late_result: bool
) -> None:
    monkeypatch.setattr(workflow, "in_workflow", lambda: True)
    client = TemporalCodexClient()
    entered = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()
    closed: list[str] = []
    connection = Connection("reserved", "owner")

    async def execute(
        name: str, arg: Any, config: workflow.ActivityConfig, result_type: Any = None
    ) -> Any:
        del config, result_type
        if name == CLOSE:
            assert isinstance(arg, CloseActivityInput)
            closed.append(arg.handle)
            return None
        if name == stage:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                if not late_result:
                    raise
        return connection

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
        assert closed == ([connection.handle] if stage == CONNECT else [])
        assert client._connection is None
        assert client._reader is None
    finally:
        release.set()
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
