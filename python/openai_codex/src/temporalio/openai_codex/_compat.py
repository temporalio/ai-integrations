"""Narrow, idempotent adapters for upstream AsyncCodex in Workflows."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from functools import wraps
from typing import Any

from openai_codex import CodexError, _run, api
from openai_codex._initialize_metadata import validate_initialize_metadata
from openai_codex.generated.v2_all import Turn, TurnCompletedNotification
from openai_codex.models import Notification
from temporalio import workflow

from ._client import TemporalCodexClient

_installed = False


def install_adapters() -> None:
    """Adapt construction, initialization, streams, and failed-turn errors.

    Workflow detection includes replay. The upstream high-level thread, turn,
    authentication handles, and result collection remain in use. Ordinary SDK
    use outside Workflows delegates to the original implementations.
    """
    global _installed
    if _installed:
        return
    original_init = api.AsyncCodex.__init__
    original_ensure_initialized = api.AsyncCodex._ensure_initialized
    original_stream = api.AsyncTurnHandle.stream
    original_raise_for_failed_turn = _run._raise_for_failed_turn

    @wraps(original_raise_for_failed_turn)
    def raise_for_failed_turn(turn: Turn) -> None:
        try:
            original_raise_for_failed_turn(turn)
        except RuntimeError as error:
            if not workflow.in_workflow():
                raise
            # The SDK reports failed turns with RuntimeError, which would keep
            # failing Workflow Tasks instead of terminating the Workflow.
            raise CodexError(str(error)) from error

    @wraps(original_init)
    def initialize(self: Any, config: Any = None) -> None:
        if not workflow.in_workflow():
            original_init(self, config)
            return
        self._client = TemporalCodexClient(config)
        self._init = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @wraps(original_ensure_initialized)
    async def ensure_initialized(self: Any) -> None:
        if not isinstance(self._client, TemporalCodexClient):
            await original_ensure_initialized(self)
            return
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            try:
                await self._client.start()
                payload = await self._client.initialize()
                self._init = validate_initialize_metadata(payload)
                self._initialized = True
            except (Exception, asyncio.CancelledError):
                # Context entry and lazy initialization can be cancelled before
                # __aexit__ runs. Only the initialization owner closes the client;
                # cancellation while waiting for the lock must leave it intact.
                await self.close()
                raise

    @wraps(original_stream)
    async def stream(self: Any) -> AsyncIterator[Notification]:
        if not isinstance(self._codex._client, TemporalCodexClient):
            async for event in original_stream(self):
                yield event
            return
        await self._codex._ensure_initialized()
        try:
            while True:
                event = await self._subscription.next()
                yield event
                if (
                    event.method == "turn/completed"
                    and isinstance(event.payload, TurnCompletedNotification)
                    and event.payload.turn.id == self.id
                ):
                    return
        finally:
            self._subscription.close()

    setattr(api.AsyncCodex, "__init__", initialize)
    setattr(api.AsyncCodex, "_ensure_initialized", ensure_initialized)
    setattr(api.AsyncTurnHandle, "stream", stream)
    setattr(_run, "_raise_for_failed_turn", raise_for_failed_turn)
    _installed = True
