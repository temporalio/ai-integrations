"""Test helpers for this integration."""

import uuid
from collections.abc import Callable, Sequence

from temporalio.client import Client
from temporalio.worker import Plugin, Worker


def new_worker(
    client: Client,
    *workflows: type,
    activities: Sequence[Callable] = [],
    plugins: Sequence[Plugin] = [],
    task_queue: str | None = None,
) -> Worker:
    """Create a Worker on a fresh task queue (sandboxed workflow runner, the default)."""
    return Worker(
        client,
        task_queue=task_queue or str(uuid.uuid4()),
        workflows=workflows,
        activities=activities,
        plugins=plugins,
    )
