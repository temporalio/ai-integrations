"""Client, process and polling helpers for the tests that run, and SIGKILL, workers."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def group_alive(pgid: int) -> bool:
    """Signal 0 checks that ``pgid`` still names a process group without killing it."""
    if sys.platform == "win32":
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def killpg(pid: int) -> None:
    if sys.platform == "win32":
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


async def until(
    ready: Callable[[], bool | Awaitable[bool]],
    timeout: float = 15,
    interval: float = 0.05,
) -> None:
    async with asyncio.timeout(timeout):
        while True:
            done = ready()
            if await done if inspect.isawaitable(done) else done:
                return
            await asyncio.sleep(interval)


def client(env: WorkflowEnvironment) -> Client:
    config: dict[str, Any] = {
        **env.client.config(),
        "data_converter": pydantic_data_converter,
    }
    return Client(**config)
