"""Bounded retries for local Temporal server startup."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable

from temporalio.testing import WorkflowEnvironment

logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT = re.compile(
    r"^Failed starting Temporal dev server: "
    r"ephemeral server at \S+ did not start within \d+s\."
)


async def start_local_with_retry(
    start: Callable[[], Awaitable[WorkflowEnvironment]],
) -> WorkflowEnvironment:
    """Invoke a configured startup callback, retrying only SDK startup timeouts.

    The caller binds all server options, keeping this policy independent of
    plugin configuration. The SDK terminates each timed-out server process
    before raising. Other errors and cancellation propagate immediately.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await start()
        except RuntimeError as exc:
            if attempt >= 3 or not _STARTUP_TIMEOUT.match(str(exc)):
                raise
            logger.warning(
                "Temporal dev server startup attempt %s failed: %s", attempt, exc
            )
            await asyncio.sleep(1)
