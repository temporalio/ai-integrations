"""Bounded retries for local Temporal server startup."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Awaitable, Callable

from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment

logger = logging.getLogger(__name__)

# SDK startup timeouts include their Rust-formatted duration and last connection error.
_RETRYABLE_STARTUP_ERROR = re.compile(
    r"^Failed starting Temporal dev server: (?:"
    r"ephemeral server at \S+ did not start within "
    r"\d+(?:\.\d+)?(?:s|ms|µs|ns); last connection error:"
    r"|failed to download ephemeral server executable:"
    r"|invalid ephemeral server download: could not (?:read|extract) "
    r"(?:tar|ZIP) archive(?: entry)?(?:$|:)"
    r")"
)


async def create_test_environment(
    start: Callable[[], Awaitable[WorkflowEnvironment]],
) -> WorkflowEnvironment:
    """Use the launcher-owned server, or provision one for direct pytest runs."""
    if address := os.environ.get("TEMPORAL_TEST_ADDRESS"):
        return WorkflowEnvironment.from_client(await Client.connect(address))
    return await start_local_with_retry(start)


async def start_local_with_retry(
    start: Callable[[], Awaitable[WorkflowEnvironment]],
) -> WorkflowEnvironment:
    """Invoke a configured startup callback with up to three attempts.

    The caller binds all server options, keeping this policy independent of
    plugin configuration. Only the SDK's startup timeout, download, and archive
    read/extraction errors are retried after one second. Other errors and
    cancellation propagate immediately.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await start()
        except RuntimeError as exc:
            if attempt >= 3 or not _RETRYABLE_STARTUP_ERROR.match(str(exc)):
                raise
            logger.warning(
                "Temporal dev server startup attempt %s failed: %s", attempt, exc
            )
            await asyncio.sleep(1)
