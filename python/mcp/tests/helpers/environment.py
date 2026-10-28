"""Bounded retries for local Temporal server startup."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from temporalio.testing import WorkflowEnvironment

logger = logging.getLogger(__name__)


async def start_local_with_retry(
    start: Callable[[], Awaitable[WorkflowEnvironment]],
) -> WorkflowEnvironment:
    """Invoke a configured startup callback with up to three attempts.

    The caller binds all server options, keeping this policy independent of
    plugin configuration. Any Exception from the callback is retried after one
    second. Cancellation and other BaseException subclasses propagate immediately.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await start()
        except Exception as exc:
            if attempt >= 3:
                raise
            logger.warning(
                "Temporal dev server setup attempt %s failed: %s", attempt, exc
            )
            await asyncio.sleep(1)
