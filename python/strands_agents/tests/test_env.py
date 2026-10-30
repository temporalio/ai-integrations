"""Local server setup retries only known SDK provisioning failures."""

import asyncio
from functools import partial
from unittest.mock import AsyncMock, call, patch

import pytest

from temporalio.testing import WorkflowEnvironment
from tests.helpers.environment import start_local_with_retry

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
