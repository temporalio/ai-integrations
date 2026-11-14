"""Activity-only worker for the cross-process backend factory regression."""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
from typing import Any

from temporalio import activity
from temporalio.client import Client
from temporalio.deepagents import DeepAgentsPlugin
from temporalio.deepagents._tools import registered_backends
from temporalio.exceptions import ApplicationError
from temporalio.worker import Worker


def filesystem_factory(config: dict[str, Any]) -> Any:
    """Construct a filesystem backend using only the activity's configuration."""
    activity.info()  # Construction must happen inside an activity.
    assert not registered_backends()
    backends = importlib.import_module("deepagents.backends")
    return backends.FilesystemBackend(**config)


async def async_filesystem_factory(config: dict[str, Any]) -> Any:
    """Also exercise asynchronous sandbox connection factories."""
    await asyncio.sleep(0)
    return filesystem_factory(config)


def blocking_filesystem_factory(config: dict[str, Any]) -> Any:
    """Simulate a synchronous reconnect slower than the heartbeat timeout."""
    time.sleep(2)
    return filesystem_factory(config)


def retry_filesystem_factory(config: dict[str, Any]) -> Any:
    """Simulate a transient sandbox connection failure on each first attempt."""
    if activity.info().attempt == 1:
        raise ApplicationError(
            "Sandbox connection unavailable", type="SandboxUnavailable"
        )
    return filesystem_factory(config)


def shell_factory(root_dir: str) -> Any:
    """Construct a shell backend in the activity process."""
    activity.info()
    assert not registered_backends()
    backends = importlib.import_module("deepagents.backends")
    return backends.LocalShellBackend(root_dir=root_dir)


async def main() -> None:
    """Run until the parent test terminates the process."""
    client = await Client.connect(sys.argv[1])
    async with Worker(
        client,
        task_queue=sys.argv[2],
        plugins=[
            DeepAgentsPlugin(
                backends={
                    "filesystem": filesystem_factory,
                    "worker-filesystem": lambda: filesystem_factory(
                        {"root_dir": sys.argv[3], "virtual_mode": True}
                    ),
                    "async-filesystem": async_filesystem_factory,
                    "blocking-filesystem": blocking_filesystem_factory,
                    "retry-filesystem": retry_filesystem_factory,
                    "shell": shell_factory,
                }
            )
        ],
    ):
        print("READY", flush=True)
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
