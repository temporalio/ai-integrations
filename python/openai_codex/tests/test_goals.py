"""Goal-start failures preserve SDK errors and release operation state."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from openai_codex import CodexError, TransportClosedError
from openai_codex.generated.v2_all import IdleThreadStatus, ThreadGoalStatus

from temporalio import workflow
from temporalio.openai_codex import TemporalCodexClient
from temporalio.openai_codex._routing import GoalOperation


async def test_goal_registered_after_connection_failure_inherits_error() -> None:
    client = TemporalCodexClient()
    error = TransportClosedError("connection lost before goal registration")
    client._router.fail(error)
    state = client.register_goal_operation("thread")
    assert state.inbox.error is error
    with pytest.raises(TransportClosedError) as raised:
        await client.next_goal_notification(state)
    assert raised.value is error
    client.unregister_goal_operation(state)
    assert not client._router.goals


async def test_goal_start_timeout_is_a_codex_error_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TemporalCodexClient()
    states: list[GoalOperation] = []
    statuses: list[ThreadGoalStatus] = []
    cleared: list[str] = []

    async def thread_read(thread_id: str) -> Any:
        assert thread_id == "thread"
        return SimpleNamespace(
            thread=SimpleNamespace(
                status=SimpleNamespace(root=IdleThreadStatus(type="idle")),
                ephemeral=False,
                path="/thread",
            )
        )

    async def goal_clear(thread_id: str) -> None:
        cleared.append(thread_id)

    async def goal_set(thread_id: str, **params: Any) -> None:
        states.append(client._router.goals[thread_id])
        statuses.append(params["status"])

    async def wait_condition(condition: Callable[[], bool], *, timeout: float) -> None:
        assert timeout == 30
        assert not condition()
        raise asyncio.TimeoutError()

    monkeypatch.setattr(client, "thread_read", thread_read)
    monkeypatch.setattr(client, "thread_goal_clear", goal_clear)
    monkeypatch.setattr(client, "thread_goal_set", goal_set)
    monkeypatch.setattr(workflow, "wait_condition", wait_condition)
    with pytest.raises(
        CodexError, match="timed out waiting for goal turn to start after 30 seconds"
    ) as raised:
        await client.start_goal_operation("thread", "Finish the work")
    assert isinstance(raised.value.__cause__, asyncio.TimeoutError)
    assert cleared == ["thread"]
    assert statuses == [ThreadGoalStatus.active, ThreadGoalStatus.paused]
    assert states[0].finished
    assert not client._router.goals
