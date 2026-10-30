"""Local server setup retries only known SDK provisioning failures."""

import asyncio
import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from unittest.mock import AsyncMock, call, patch

import pytest

from temporalio.testing import WorkflowEnvironment
from tests.helpers.environment import (
    _capture_server_output,
    _ServerDiagnostics,
    diagnosed_local_server,
    start_local_with_retry,
)

STARTUP_PREFIX = "Failed starting Temporal dev server: "
STARTUP_TIMEOUT = (
    STARTUP_PREFIX + "ephemeral server at 127.0.0.1:60249 did not start within 5s. "
    "Make sure another download isn't stuck and delete the temp file."
)
DOWNLOAD_FAILURE = (
    STARTUP_PREFIX
    + "failed to download ephemeral server executable: error sending request"
)
RETRYABLE_ERRORS = [
    STARTUP_TIMEOUT,
    DOWNLOAD_FAILURE,
    *[
        STARTUP_PREFIX + "invalid ephemeral server download: " + message
        for message in [
            "could not read tar archive",
            "could not read tar archive entry",
            "could not extract tar archive entry",
            "could not read ZIP archive",
            "could not extract ZIP archive entry",
        ]
    ],
]


@pytest.mark.parametrize("message", RETRYABLE_ERRORS)
@pytest.mark.parametrize("failures", [0, 1, 2])
async def test_server_startup_recovers(message: str, failures: int) -> None:
    environment = AsyncMock(spec=WorkflowEnvironment)
    start = AsyncMock(
        side_effect=[RuntimeError(message) for _ in range(failures)] + [environment]
    )
    # The retry policy treats all SDK configuration as opaque caller input.
    options = {
        "dev_server_download_version": "test-version",
        "namespace": "test-namespace",
        "dev_server_extra_args": ["--dynamic-config-value", "test.option=50"],
    }
    with patch(
        "tests.helpers.environment.asyncio.sleep", new_callable=AsyncMock
    ) as sleep:
        assert await start_local_with_retry(partial(start, **options)) is environment

    assert start.await_args_list == [call(**options)] * (failures + 1)
    assert sleep.await_args_list == [call(1)] * failures


@pytest.mark.parametrize("message", RETRYABLE_ERRORS)
async def test_server_startup_stops_after_three_attempts(message: str) -> None:
    error = RuntimeError(message)
    start = AsyncMock(side_effect=error)
    with (
        patch(
            "tests.helpers.environment.asyncio.sleep", new_callable=AsyncMock
        ) as sleep,
        pytest.raises(RuntimeError) as raised,
    ):
        await start_local_with_retry(start)

    assert raised.value is error
    assert start.await_count == 3
    assert sleep.await_args_list == [call(1), call(1)]


async def test_different_startup_errors_share_retry_limit() -> None:
    errors = [RuntimeError(message) for message in RETRYABLE_ERRORS[:3]]
    start = AsyncMock(side_effect=errors)
    with (
        patch(
            "tests.helpers.environment.asyncio.sleep", new_callable=AsyncMock
        ) as sleep,
        pytest.raises(RuntimeError) as raised,
    ):
        await start_local_with_retry(start)

    assert raised.value is errors[-1]
    assert start.await_count == 3
    assert sleep.await_args_list == [call(1), call(1)]


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError(
            STARTUP_PREFIX + "ephemeral server executable does not exist: /bad"
        ),
        RuntimeError(
            STARTUP_PREFIX + "failed to start ephemeral server: permission denied"
        ),
        RuntimeError(
            STARTUP_PREFIX
            + "invalid ephemeral server download: unsupported architecture: other"
        ),
        RuntimeError(
            STARTUP_PREFIX
            + "invalid ephemeral server download: archive URL has unsupported format"
        ),
        RuntimeError(STARTUP_PREFIX + "invalid configuration"),
        RuntimeError("Failed client connect: connection refused"),
        RuntimeError("request did not start within 5s."),
        TypeError("start_local() got an unexpected keyword argument 'invalid'"),
        ValueError(STARTUP_TIMEOUT),
        OSError(DOWNLOAD_FAILURE),
        asyncio.CancelledError(),
        KeyboardInterrupt(),
        SystemExit(),
    ],
)
async def test_other_errors_are_not_retried(error: BaseException) -> None:
    start = AsyncMock(side_effect=error)
    with (
        patch(
            "tests.helpers.environment.asyncio.sleep", new_callable=AsyncMock
        ) as sleep,
        pytest.raises(type(error)) as raised,
    ):
        await start_local_with_retry(start)

    assert raised.value is error
    start.assert_awaited_once()
    sleep.assert_not_awaited()


async def test_unexpected_error_after_retry_propagates_immediately() -> None:
    error = TypeError("invalid startup configuration")
    start = AsyncMock(side_effect=[RuntimeError(DOWNLOAD_FAILURE), error])
    with (
        patch(
            "tests.helpers.environment.asyncio.sleep", new_callable=AsyncMock
        ) as sleep,
        pytest.raises(TypeError) as raised,
    ):
        await start_local_with_retry(start)

    assert raised.value is error
    assert start.await_count == 2
    sleep.assert_awaited_once_with(1)


def _events(directory: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (directory / "lifecycle.jsonl").read_text().splitlines()
    ]


async def test_diagnostics_keep_startup_error_and_restore_streams(
    tmp_path: Path,
) -> None:
    error = RuntimeError("Failed client connect: connection refused")
    start = AsyncMock(side_effect=error)
    with pytest.raises(RuntimeError) as raised:
        async with diagnosed_local_server(
            start, server_version="pinned", log_root=tmp_path
        ):
            pytest.fail("A failed startup must not yield an environment")
    assert raised.value is error
    start.assert_awaited_once()
    events = _events(next(tmp_path.iterdir()))
    assert [event["event"] for event in events] == [
        "session",
        "startup_requested",
        "startup_failed",
    ]
    assert events[0]["server_version"] == "pinned"
    # A failed startup must not leave pytest writing to the server's files.
    os.write(1, b"stdout restored\n")
    os.write(2, b"stderr restored\n")
    assert b"restored" not in (next(tmp_path.iterdir()) / "stdout.log").read_bytes()
    assert b"restored" not in (next(tmp_path.iterdir()) / "stderr.log").read_bytes()


async def test_diagnostics_always_shutdown_after_body_failure(tmp_path: Path) -> None:
    environment = AsyncMock(spec=WorkflowEnvironment)
    environment.client.service_client.config.target_host = "127.0.0.1:7233"
    error = ValueError("test body failed")
    with pytest.raises(ValueError) as raised:
        async with diagnosed_local_server(
            AsyncMock(return_value=environment),
            server_version="pinned",
            log_root=tmp_path,
        ):
            raise error
    assert raised.value is error
    environment.shutdown.assert_awaited_once()
    events = _events(next(tmp_path.iterdir()))
    assert (
        next(event for event in events if event["event"] == "ready")["address"]
        == "127.0.0.1:7233"
    )
    assert [event["event"] for event in events][-2:] == [
        "shutdown_requested",
        "shutdown_completed",
    ]


def test_native_child_output_and_exit_are_retained(tmp_path: Path) -> None:
    """Native children keep their log handles after the parent's are restored."""
    diagnostics = _ServerDiagnostics(tmp_path, "pinned")
    diagnostics.start()
    try:
        with _capture_server_output(tmp_path):
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import os, time; time.sleep(0.5); os.write(1, b'child stdout\\n'); "
                    "os.write(2, b'child stderr\\n'); time.sleep(0.5); raise SystemExit(7)",
                    "server",
                    "start-dev",
                ]
            )
        try:
            assert child.wait(timeout=10) == 7
            deadline = time.monotonic() + 5
            while not any(
                event["event"] == "process_exited" for event in _events(tmp_path)
            ):
                assert time.monotonic() < deadline, "Child exit was not observed"
                time.sleep(0.05)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    finally:
        diagnostics.close()
    assert (tmp_path / "stdout.log").read_bytes() == b"child stdout\n"
    assert (tmp_path / "stderr.log").read_bytes() == b"child stderr\n"
    events = _events(tmp_path)
    assert (
        next(event for event in events if event["event"] == "process_started")["pid"]
        == child.pid
    )
    exited = next(event for event in events if event["event"] == "process_exited")
    assert exited["pid"] == child.pid
    if sys.platform == "win32":
        assert exited["exit_code"] == 7
    else:
        assert exited["exit_code"] is None


async def test_cancellation_during_retry_delay_is_not_retried() -> None:
    start = AsyncMock(side_effect=RuntimeError(DOWNLOAD_FAILURE))
    with (
        patch(
            "tests.helpers.environment.asyncio.sleep",
            side_effect=asyncio.CancelledError,
        ) as sleep,
        pytest.raises(asyncio.CancelledError),
    ):
        await start_local_with_retry(start)

    start.assert_awaited_once()
    sleep.assert_awaited_once_with(1)
