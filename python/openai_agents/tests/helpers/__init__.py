"""Test helpers vendored from temporalio/sdk-python ``tests/helpers/__init__.py`` (origin/main).

Trimmed to what this plugin's tests import. Re-sync by hand (scripts/migrate/README.md).
"""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import timedelta
from typing import TypeVar

from temporalio.client import Client
from temporalio.service import RPCError, RPCStatusCode
from temporalio.worker import Worker, WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner


def new_worker(
    client: Client,
    *workflows: type,
    activities: Sequence[Callable] = [],
    task_queue: str | None = None,
    workflow_runner: WorkflowRunner = SandboxedWorkflowRunner(),
    max_cached_workflows: int = 1000,
    workflow_failure_exception_types: Sequence[type[BaseException]] = [],
    **kwargs,  # type:ignore[reportMissingParameterType]
) -> Worker:
    return Worker(
        client,
        task_queue=task_queue or str(uuid.uuid4()),
        workflows=workflows,
        activities=activities,
        workflow_runner=workflow_runner,
        max_cached_workflows=max_cached_workflows,
        workflow_failure_exception_types=workflow_failure_exception_types,
        **kwargs,
    )


T = TypeVar("T")


async def assert_eventually(
    fn: Callable[[], Awaitable[T]],
    *,
    timeout: timedelta = timedelta(seconds=10),
    interval: timedelta = timedelta(milliseconds=200),
    retry_on_rpc_cancelled: bool = True,
) -> T:
    start_sec = time.monotonic()
    while True:
        try:
            res = await fn()
            return res
        except AssertionError:
            if timedelta(seconds=time.monotonic() - start_sec) >= timeout:
                raise
        except RPCError as e:
            if retry_on_rpc_cancelled and e.status == RPCStatusCode.CANCELLED:
                continue
            else:
                raise
        await asyncio.sleep(interval.total_seconds())


async def assert_eq_eventually(
    expected: T,
    fn: Callable[[], Awaitable[T]],
    *,
    timeout: timedelta = timedelta(seconds=10),
    interval: timedelta = timedelta(milliseconds=200),
) -> None:
    async def check() -> None:
        assert expected == await fn()

    await assert_eventually(check, timeout=timeout, interval=interval)
