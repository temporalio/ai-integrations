"""Heartbeat on a timer while a model call runs."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator

from temporalio import activity


@contextlib.asynccontextmanager
async def heartbeat_every() -> AsyncGenerator[None]:
    """Heartbeat at half the activity's heartbeat timeout, when one is set.

    Heartbeats let a cancel reach a running model call (it arrives as
    ``asyncio.CancelledError`` in the call) and let the server notice a lost
    worker.
    """
    timeout = activity.info().heartbeat_timeout
    if not timeout:
        yield
        return

    async def beat() -> None:
        while True:
            await asyncio.sleep(timeout.total_seconds() / 2)
            activity.heartbeat()

    task = asyncio.create_task(beat())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
