"""Notification and subscription behavior at race boundaries."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from openai_codex import AsyncChatgptLoginHandle, AsyncCodex, TransportClosedError
from openai_codex.generated.v2_all import (
    AccountLoginCompletedNotification,
    LoginAccountResponse,
)
from openai_codex.models import Notification, UnknownNotification

from temporalio import workflow
from temporalio.openai_codex import OpenAICodexPlugin
from temporalio.openai_codex._client import TemporalCodexClient
from temporalio.openai_codex._routing import Router
from temporalio.openai_codex._types import Message


def event(turn: str, value: str = "hello") -> Notification:
    return Notification(
        "custom/event",
        UnknownNotification({"turnId": turn, "threadId": "thread", "text": value}),
    )


async def test_independent_streams_and_early_events() -> None:
    router = Router()
    router.pending += 1
    router.route(event("first"))
    router.route(event("second", "world"))
    first = router.subscribe("first", cursor=0)
    also_first = router.subscribe("first", cursor=0)
    second = router.subscribe("second", cursor=0)
    router.pending -= 1
    router.prune()
    assert await first.next() == await also_first.next() == event("first")
    assert await second.next() == event("second", "world")
    first.close()
    also_first.close()
    second.close()
    router.prune()
    assert not router.turns


@pytest.mark.parametrize("close_first", [False, True])
async def test_pending_join_retains_events_consumed_by_existing_subscriber(
    close_first: bool,
) -> None:
    router = Router()
    first = router.subscribe("turn")
    router.route(event("turn", "before join"))
    assert await first.next() == event("turn", "before join")

    cursor = first.state.end
    router.pending += 1
    router.route(event("turn", "during join"))
    assert await first.next() == event("turn", "during join")
    if close_first:
        first.close()

    joined = router.subscribe("turn", cursor=cursor)
    router.pending -= 1
    router.prune()
    assert await joined.next() == event("turn", "during join")
    assert not joined.state.events
    joined.close()
    first.close()
    router.prune()
    assert not router.turns


async def test_failed_pending_join_releases_consumed_events() -> None:
    router = Router()
    first = router.subscribe("turn")
    router.pending += 1
    router.route(event("turn"))
    assert await first.next() == event("turn")
    router.pending -= 1
    router.prune()
    assert not first.state.events
    first.close()
    router.prune()
    assert not router.turns


async def test_repeated_close_preserves_a_reused_subscription_token() -> None:
    router = Router()
    closed = router.subscribe("turn")
    closed.close()
    current = router.subscribe("turn")
    closed.close()
    assert current.token in current.state.cursors
    router.route(event("turn"))
    assert await current.next() == event("turn")
    current.close()


async def test_login_created_after_failure_inherits_terminal_error() -> None:
    router = Router()
    failure = TransportClosedError("reader stopped")
    router.fail(failure)
    inbox = router.login("late")
    assert inbox.error is failure
    with pytest.raises(TransportClosedError, match="reader stopped"):
        await inbox.next()


@pytest.mark.parametrize("high_level", [False, True])
async def test_login_response_racing_reader_failure_cannot_leave_waiters_blocked(
    monkeypatch: pytest.MonkeyPatch,
    high_level: bool,
) -> None:
    client = TemporalCodexClient()
    failure = TransportClosedError("reader stopped")

    async def login_response(*args: Any, **kwargs: Any) -> LoginAccountResponse:
        del args, kwargs
        client._router.fail(failure)
        return LoginAccountResponse.model_validate(
            {
                "type": "chatgpt",
                "loginId": "login",
                "authUrl": "https://example.invalid",
            }
        )

    monkeypatch.setattr(client, "request", login_response)
    handle: AsyncChatgptLoginHandle | None = None
    if high_level:
        OpenAICodexPlugin()
        monkeypatch.setattr(workflow, "in_workflow", lambda: True)
        codex = AsyncCodex()
        setattr(codex, "_client", client)
        setattr(codex, "_initialized", True)
        handle = await codex.login_chatgpt()
    else:
        await client.account_login_start({"type": "chatgpt"})

    assert client._router.logins["login"].error is failure
    with pytest.raises(TransportClosedError, match="reader stopped"):
        if handle is not None:
            await handle.wait()
        else:
            await client.wait_for_login_completed("login")


async def test_early_login_completion_is_scoped_and_retained() -> None:
    router = Router()
    notification = Notification(
        "account/login/completed",
        AccountLoginCompletedNotification(login_id="login", success=True, error=None),
    )
    router.route(notification)
    assert await router.login("login").next() == notification
    assert not router.global_inbox.items
    router.route(Notification("unknown/global", UnknownNotification({})))
    assert (await router.global_inbox.next()).method == "unknown/global"


async def test_failure_wakes_all_consumers_and_future_turns() -> None:
    router = Router()
    subscription = router.subscribe("turn")
    login = router.login("login")
    readers = [
        asyncio.create_task(subscription.next()),
        asyncio.create_task(login.next()),
        asyncio.create_task(router.global_inbox.next()),
    ]
    await asyncio.sleep(0)
    router.fail(TransportClosedError("lost"))
    results = await asyncio.gather(*readers, return_exceptions=True)
    assert all(isinstance(result, TransportClosedError) for result in results)
    with pytest.raises(TransportClosedError):
        await router.subscribe("later").next()


async def test_upstream_approval_defaults() -> None:
    client = TemporalCodexClient()
    writes: list[Message] = []

    async def write(message: Message) -> None:
        writes.append(message)

    setattr(client, "_write", write)
    for method in (
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
        "unknown/serverRequest",
    ):
        await client._approve({"id": 42, "method": method, "params": {}})
    assert [message["result"] for message in writes] == [
        {"decision": "accept"},
        {"decision": "accept"},
        {},
    ]
