"""Overlap, lost results, subprocess failures, and cleanup at Activity boundaries."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from openai_codex import CodexConfig, CodexError, TransportClosedError

from temporalio.exceptions import ApplicationError
from temporalio.openai_codex import ConnectActivityInput, SubprocessConnector
from temporalio.openai_codex._activities import ConnectionActivities, _fit_payload
from temporalio.openai_codex._subprocess import CodexProcessConnection
from temporalio.openai_codex._types import (
    CloseActivityInput,
    ReadActivityInput,
    WriteActivityInput,
)
from temporalio.testing import ActivityEnvironment
from tests.helpers.protocol import ProtocolConnector


async def test_overlapping_connect_and_write_retries() -> None:
    connector = ProtocolConnector()
    activities = ConnectionActivities(connector, "private")
    run = ActivityEnvironment().run
    first, second = await asyncio.gather(
        run(activities.connect, ConnectActivityInput()),
        run(activities.connect, ConnectActivityInput()),
    )
    assert first == second
    active = [process for process in connector.processes if not process.closed]
    assert len(active) == 1
    process = active[0]
    process.gate = asyncio.Event()
    request = WriteActivityInput(first.handle, 1, {"method": "ignored"})
    attempt = asyncio.create_task(run(activities.write, request))
    while not process.writes:
        await asyncio.sleep(0)
    attempt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    retry = asyncio.create_task(run(activities.write, request))
    later = asyncio.create_task(
        run(activities.write, WriteActivityInput(first.handle, 2, {"method": "later"}))
    )
    await asyncio.sleep(0.01)
    assert len(process.writes) == 1
    process.gate.set()
    await asyncio.gather(retry, later)
    await run(activities.write, request)  # lost completion
    assert process.writes == [{"method": "ignored"}, {"method": "later"}]
    await run(activities.close, CloseActivityInput(first.handle))
    await run(activities.close, CloseActivityInput(first.handle))
    assert process.closed


async def test_cancelled_read_and_lost_result_are_retained() -> None:
    connector = ProtocolConnector()
    activities = ConnectionActivities(connector, "private")
    run = ActivityEnvironment().run
    connection = await run(activities.connect, ConnectActivityInput())
    request = ReadActivityInput(connection.handle, 0)
    attempt = asyncio.create_task(run(activities.read, request))
    process = connector.processes[0]
    while not process.reads:
        await asyncio.sleep(0)
    attempt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    process.output.put_nowait([{"n": 1}, {"n": 2}])
    result, duplicate = await asyncio.gather(
        run(activities.read, request), run(activities.read, request)
    )
    assert result == duplicate and result.output == [{"n": 1}, {"n": 2}]
    assert process.reads == 1
    process.output.put_nowait([{"n": 3}])
    assert (
        await run(activities.read, ReadActivityInput(connection.handle, 2))
    ).output == [{"n": 3}]
    await activities.shut_down()
    assert process.closed
    with pytest.raises(ApplicationError, match="Lost the connection"):
        await run(activities.read, request)


@pytest.mark.parametrize("worker_shutdown", [False, True])
async def test_close_unblocks_retained_io_without_cancelling(
    worker_shutdown: bool,
) -> None:
    connector = ProtocolConnector()
    activities = ConnectionActivities(connector, "private")
    run = ActivityEnvironment().run
    connection = await run(activities.connect, ConnectActivityInput())
    process = connector.processes[0]
    process.gate = asyncio.Event()
    reading = asyncio.create_task(
        run(activities.read, ReadActivityInput(connection.handle, 0))
    )
    writing = asyncio.create_task(
        run(activities.write, WriteActivityInput(connection.handle, 1, {}))
    )
    while not process.reads or not process.writes:
        await asyncio.sleep(0)
    reading.cancel()
    writing.cancel()
    await asyncio.gather(reading, writing, return_exceptions=True)
    retained = activities._connections[connection.handle]
    assert retained.reading is not None
    pending: list[asyncio.Future[Any]] = [retained.reading, *retained.writes.values()]

    if worker_shutdown:
        await activities.shut_down()
    else:
        await run(activities.close, CloseActivityInput(connection.handle))

    assert process.closed
    assert not activities._connections
    assert all(task.done() and not task.cancelled() for task in pending)
    assert [task.result() for task in pending] == [[], None]


@pytest.mark.parametrize(
    "script, error",
    [
        (
            "import sys; print('broken', file=sys.stderr); sys.exit(7)",
            TransportClosedError,
        ),
        ("print('not json', flush=True)", CodexError),
        ("print('[]', flush=True)", CodexError),
    ],
)
async def test_subprocess_errors(script: str, error: type[Exception]) -> None:
    process = await SubprocessConnector().connect(
        ConnectActivityInput(
            CodexConfig(launch_args_override=(sys.executable, "-c", script))
        )
    )
    try:
        with pytest.raises(error):
            await process.read()
    finally:
        await asyncio.gather(process.close(), process.close())


async def test_stdout_eof_waits_for_final_stderr() -> None:
    process = await SubprocessConnector().connect(
        ConnectActivityInput(
            CodexConfig(
                launch_args_override=(
                    sys.executable,
                    "-c",
                    "import os, sys, time; os.close(1); time.sleep(0.03); "
                    "print('final diagnostic', file=sys.stderr, flush=True)",
                )
            )
        )
    )
    try:
        with pytest.raises(TransportClosedError, match="final diagnostic"):
            await process.read()
    finally:
        await process.close()


async def test_cancelled_subprocess_close_can_be_retried() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class Process:
        stdin = None
        stdout = asyncio.StreamReader()
        stderr = asyncio.StreamReader()
        returncode: int | None = None
        reaped = False

        def terminate(self) -> None:
            pass

        async def wait(self) -> int:
            entered.set()
            await release.wait()
            self.reaped = True
            self.returncode = 0
            return 0

    process = Process()
    connection = CodexProcessConnection(
        cast(asyncio.subprocess.Process, cast(object, process))
    )
    closing = asyncio.create_task(connection.close())
    try:
        await entered.wait()
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        release.set()
        await asyncio.gather(connection.close(), connection.close())
        assert process.reaped
        assert connection._stdout_task.done() and connection._stderr_task.done()
    finally:
        release.set()
        await connection.close()


async def test_stdout_line_limit_is_a_permanent_read_error() -> None:
    activities = ConnectionActivities(SubprocessConnector(), "private")
    run = ActivityEnvironment().run
    connection = await run(
        activities.connect,
        ConnectActivityInput(
            CodexConfig(
                launch_args_override=(
                    sys.executable,
                    "-c",
                    "print('x' * (16 * 1024 * 1024 + 1), flush=True)",
                )
            )
        ),
    )
    try:
        for _ in range(2):
            with pytest.raises(ApplicationError, match="limit") as raised:
                await run(activities.read, ReadActivityInput(connection.handle, 0))
            assert raised.value.type == "TransportClosedError"
            assert raised.value.non_retryable
            terminal = raised.value.__cause__
            assert isinstance(terminal, TransportClosedError)
            assert isinstance(terminal.__cause__, ValueError)
    finally:
        await run(activities.close, CloseActivityInput(connection.handle))
    assert not activities._connections


def test_payload_batching() -> None:
    output: list[dict[str, Any]] = [{"data": "x" * 500_000}] * 5
    batch = _fit_payload(output)
    assert 0 < len(batch) < len(output)
    assert batch == output[: len(batch)]


async def test_missing_runtime_is_a_permanent_launch_error(tmp_path: Path) -> None:
    activities = ConnectionActivities(SubprocessConnector(), "private")
    with pytest.raises(ApplicationError) as failure:
        await ActivityEnvironment().run(
            activities.connect,
            ConnectActivityInput(CodexConfig(codex_bin=str(tmp_path / "missing"))),
        )
    assert failure.value.type == "FileNotFoundError"
    assert failure.value.non_retryable


async def test_connect_cancellation_cleans_unreturned_process() -> None:
    from tests.helpers.protocol import ProtocolProcess

    entered = asyncio.Event()
    release = asyncio.Event()
    process = ProtocolProcess()

    class DelayedConnector:
        async def connect(self, request: ConnectActivityInput) -> ProtocolProcess:
            del request
            entered.set()
            await release.wait()
            return process

    activities = ConnectionActivities(DelayedConnector(), "private")
    attempt = asyncio.create_task(
        ActivityEnvironment().run(activities.connect, ConnectActivityInput())
    )
    await entered.wait()
    attempt.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    assert process.closed
    assert not activities._connections


async def test_lost_connect_result_is_closed_at_worker_shutdown() -> None:
    from tests.helpers.protocol import ProtocolProcess

    entered = asyncio.Event()
    release = asyncio.Event()
    process = ProtocolProcess()

    class DelayedConnector:
        async def connect(self, request: ConnectActivityInput) -> ProtocolProcess:
            del request
            entered.set()
            await release.wait()
            return process

    activities = ConnectionActivities(DelayedConnector(), "private")
    run = ActivityEnvironment().run
    request = ConnectActivityInput()
    attempt = asyncio.create_task(run(activities.connect, request))
    try:
        await entered.wait()
        release.set()
        await attempt  # Its result never reaches the Workflow.
        assert not process.closed
        assert len(activities._connections) == 1
    finally:
        release.set()
        await asyncio.gather(attempt, return_exceptions=True)
        await activities.shut_down()
    assert process.closed
    assert not activities._connections


async def test_connect_heartbeats_without_owner_details() -> None:
    import dataclasses
    from datetime import timedelta

    from tests.helpers.protocol import ProtocolProcess

    class DelayedConnector:
        async def connect(self, request: ConnectActivityInput) -> ProtocolProcess:
            del request
            await asyncio.sleep(0.03)
            return ProtocolProcess()

    environment = ActivityEnvironment()
    environment.info = dataclasses.replace(
        environment.info, heartbeat_timeout=timedelta(milliseconds=10)
    )
    beats: list[tuple[Any, ...]] = []
    environment.on_heartbeat = lambda *args: beats.append(args)
    activities = ConnectionActivities(DelayedConnector(), "private")
    await environment.run(activities.connect, ConnectActivityInput())
    assert len(beats) >= 3 and all(beat == () for beat in beats)
    await activities.shut_down()
    environment.info = dataclasses.replace(
        environment.info, heartbeat_details=("another-worker",)
    )
    connection = await environment.run(activities.connect, ConnectActivityInput())
    assert connection.task_queue == "private"
    await activities.shut_down()
