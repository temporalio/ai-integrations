"""Workflow-local notification routing using cooperative asyncio waiters."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from openai_codex import TransportClosedError
from openai_codex.generated.notification_registry import notification_turn_id
from openai_codex.models import Notification, UnknownNotification


class Inbox:
    """An asynchronous inbox that wakes every reader on terminal failure."""

    def __init__(self) -> None:
        """Create an empty inbox."""
        self.items: deque[Notification] = deque()
        self.changed = asyncio.Event()
        self.error: BaseException | None = None

    def put(self, event: Notification) -> None:
        """Append an event and wake waiting readers."""
        self.items.append(event)
        self.changed.set()

    def fail(self, error: BaseException) -> None:
        """Fail current and future reads after buffered events are consumed."""
        self.error = error
        self.changed.set()

    async def next(self) -> Notification:
        """Read one event without blocking the Workflow event loop."""
        while not self.items:
            if self.error is not None:
                raise self.error
            self.changed.clear()
            await self.changed.wait()
        return self.items.popleft()


@dataclass
class TurnEvents:
    """Shared unread events and independent subscription cursors."""

    events: dict[int, Notification] = field(default_factory=dict)
    end: int = 0
    first: int = 0
    cursors: dict[int, int] = field(default_factory=dict)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    complete: bool = False
    error: BaseException | None = None

    def prune(self) -> None:
        """Discard events consumed by every attached subscription."""
        consumed = min(self.cursors.values(), default=self.end)
        while self.first < consumed:
            self.events.pop(self.first, None)
            self.first += 1


class Subscription:
    """One consumer of a shared turn stream."""

    def __init__(self, router: Router, state: TurnEvents, cursor: int) -> None:
        """Subscribe at a known event offset."""
        self._router = router
        self.state = state
        self.token = max(state.cursors, default=-1) + 1
        state.cursors[self.token] = cursor
        self.closed = False

    async def next(self) -> Notification:
        """Wait for this consumer's next event."""
        state = self.state
        while not self.closed:
            cursor = state.cursors[self.token]
            if cursor < state.end:
                event = state.events[cursor]
                state.cursors[self.token] = cursor + 1
                self._prune()
                return event
            if state.error is not None:
                raise state.error
            if state.complete:
                raise TransportClosedError("Turn is no longer streaming")
            state.changed.clear()
            await state.changed.wait()
        raise TransportClosedError("Turn subscription closed")

    def close(self) -> None:
        """Release this cursor and wake any outstanding read."""
        if self.closed:
            return
        self.closed = True
        self.state.cursors.pop(self.token, None)
        self._prune()
        self.state.changed.set()

    def _prune(self) -> None:
        # Pending starts can attach at a cursor captured before their RPC reply.
        # Existing consumers must retain those events even after reading/closing.
        if not self._router.pending:
            self.state.prune()


class Router:
    """Route notifications, buffering races with turn/start and login replies."""

    def __init__(self) -> None:
        """Create empty notification routes."""
        self.global_inbox = Inbox()
        self.logins: dict[str, Inbox] = {}
        self.turns: dict[str, TurnEvents] = {}
        self.defaults: dict[str, Subscription] = {}
        self.pending = 0
        self.error: BaseException | None = None

    def subscribe(self, turn_id: str, *, cursor: int | None = None) -> Subscription:
        """Subscribe independently to a turn's events."""
        state = self.turns.setdefault(turn_id, TurnEvents(error=self.error))
        return Subscription(self, state, state.end if cursor is None else cursor)

    def register_turn(self, turn_id: str) -> None:
        """Register the low-level API's default consumer."""
        if turn_id not in self.defaults:
            self.defaults[turn_id] = self.subscribe(turn_id)

    def unregister_turn(self, turn_id: str) -> None:
        """Release the default consumer."""
        if subscription := self.defaults.pop(turn_id, None):
            subscription.close()
        self.prune()

    def login(self, login_id: str) -> Inbox:
        """Get or create a login inbox, including notifications received early."""
        if login_id not in self.logins:
            inbox = Inbox()
            if self.error is not None:
                inbox.fail(self.error)
            self.logins[login_id] = inbox
        return self.logins[login_id]

    def route(self, event: Notification) -> None:
        """Send an event to exactly one kind of route."""
        payload = event.payload
        raw: Any = payload.params if isinstance(payload, UnknownNotification) else None
        if event.method == "account/login/completed":
            login_id = (
                raw.get("loginId")
                if raw is not None
                else getattr(payload, "login_id", None)
            )
            if isinstance(login_id, str):
                self.login(login_id).put(event)
                return
        if isinstance(payload, UnknownNotification):
            raw = payload.params
            nested = raw.get("turn")
            turn_id = raw.get("turnId") or (
                nested.get("id") if isinstance(nested, dict) else None
            )
        else:
            turn_id = notification_turn_id(payload)
        if isinstance(turn_id, str):
            state = self.turns.setdefault(turn_id, TurnEvents())
            state.events[state.end] = event
            state.end += 1
            state.complete = state.complete or event.method == "turn/completed"
            state.changed.set()
            self.prune()
        else:
            self.global_inbox.put(event)

    def prune(self) -> None:
        """Discard unclaimed turn events only after start requests settle."""
        if not self.pending:
            for state in self.turns.values():
                state.prune()
            self.turns = {
                key: state for key, state in self.turns.items() if state.cursors
            }

    def fail(self, error: BaseException) -> None:
        """Wake all routes after the connection closes."""
        self.error = error
        self.global_inbox.fail(error)
        for inbox in self.logins.values():
            inbox.fail(error)
        for state in self.turns.values():
            state.error = error
            state.changed.set()
