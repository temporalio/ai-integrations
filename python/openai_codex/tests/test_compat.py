"""Workflow-aware initialization, turn failures, and ordinary SDK behavior."""

from __future__ import annotations

import asyncio

import pytest
from openai_codex import AsyncCodex, AsyncTurnHandle, CodexError, TransportClosedError
from openai_codex.generated.v2_all import TurnCompletedNotification
from openai_codex.models import Notification, UnknownNotification

from temporalio import workflow
from temporalio.openai_codex import OpenAICodexPlugin
from temporalio.openai_codex._routing import Router


async def test_unknown_turn_completion_raises_sdk_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    OpenAICodexPlugin()
    monkeypatch.setattr(workflow, "in_workflow", lambda: True)
    codex = AsyncCodex()
    codex._initialized = True
    router = Router()
    subscription = router.subscribe("turn")
    router.route(
        Notification(
            method="turn/completed",
            payload=UnknownNotification(
                params={"threadId": "thread", "turn": {"id": "turn"}}
            ),
        )
    )
    turn = AsyncTurnHandle(codex, "thread", "turn", _subscription=subscription)

    with pytest.raises(TransportClosedError, match="Turn is no longer streaming"):
        await turn.run()
    assert subscription.closed


@pytest.mark.parametrize("in_workflow", [False, True])
@pytest.mark.parametrize("has_message", [False, True])
async def test_failed_turn_collection_uses_workflow_failure_type(
    monkeypatch: pytest.MonkeyPatch, in_workflow: bool, has_message: bool
) -> None:
    OpenAICodexPlugin()
    monkeypatch.setattr(workflow, "in_workflow", lambda: in_workflow)
    codex = AsyncCodex()
    codex._initialized = True
    completed = Notification(
        method="turn/completed",
        payload=TurnCompletedNotification.model_validate(
            {
                "threadId": "thread",
                "turn": {
                    "id": "turn",
                    "items": [],
                    "status": "failed",
                    "error": (
                        {"message": "Provider authentication failed"}
                        if has_message
                        else None
                    ),
                },
            }
        ),
    )

    class Subscription:
        closed = False

        def sync_next(self) -> Notification:
            return completed

        async def async_next(self) -> Notification:
            return completed

        next = async_next if in_workflow else sync_next

        def close(self) -> None:
            self.closed = True

    subscription = Subscription()
    turn = AsyncTurnHandle(codex, "thread", "turn", _subscription=subscription)
    message = (
        "Provider authentication failed"
        if has_message
        else "turn failed with status failed"
    )
    error_type = CodexError if in_workflow else RuntimeError
    with pytest.raises(error_type, match=message) as raised:
        await turn.run()
    if in_workflow:
        assert isinstance(raised.value.__cause__, RuntimeError)
    assert subscription.closed


async def test_application_runtime_errors_are_not_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    OpenAICodexPlugin()
    monkeypatch.setattr(workflow, "in_workflow", lambda: True)
    codex = AsyncCodex()
    codex._initialized = True
    error = RuntimeError("Application callback failed")

    class Subscription:
        closed = False

        async def next(self) -> Notification:
            raise error

        def close(self) -> None:
            self.closed = True

    subscription = Subscription()
    turn = AsyncTurnHandle(codex, "thread", "turn", _subscription=subscription)
    with pytest.raises(RuntimeError) as raised:
        await turn.run()
    assert raised.value is error
    assert subscription.closed


async def test_cancelled_initialization_waiter_does_not_close_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    OpenAICodexPlugin()
    monkeypatch.setattr(workflow, "in_workflow", lambda: True)
    codex = AsyncCodex()
    client = getattr(codex, "_client")
    started = asyncio.Event()
    closed = asyncio.Event()

    async def start() -> None:
        started.set()
        await asyncio.Event().wait()

    async def close() -> None:
        closed.set()

    monkeypatch.setattr(client, "start", start)
    monkeypatch.setattr(client, "close", close)
    owner = asyncio.create_task(codex._ensure_initialized())
    waiter: asyncio.Task[None] | None = None
    try:
        await started.wait()
        waiter = asyncio.create_task(codex._ensure_initialized())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not closed.is_set()

        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert closed.is_set()
    finally:
        tasks = [owner] if waiter is None else [owner, waiter]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
