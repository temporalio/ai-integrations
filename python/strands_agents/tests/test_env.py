"""Local server setup tolerates only bounded startup timeouts."""

import asyncio
from functools import partial
from unittest.mock import AsyncMock, call, patch

import pytest

from temporalio.testing import WorkflowEnvironment
from tests.helpers.environment import start_local_with_retry

STARTUP_TIMEOUT = (
    "Failed starting Temporal dev server: ephemeral server at 127.0.0.1:60249 "
    "did not start within 5s. Make sure another download isn't stuck and delete the temp file."
)


@pytest.mark.parametrize("timeouts", [0, 1, 2])
async def test_server_startup_recovers(timeouts: int) -> None:
    environment = AsyncMock(spec=WorkflowEnvironment)
    start = AsyncMock(
        side_effect=[RuntimeError(STARTUP_TIMEOUT) for _ in range(timeouts)]
        + [environment]
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

    assert start.await_args_list == [call(**options)] * (timeouts + 1)
    assert sleep.await_args_list == [call(1)] * timeouts


async def test_server_startup_stops_after_three_attempts() -> None:
    error = RuntimeError(STARTUP_TIMEOUT)
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


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Failed starting Temporal dev server: invalid configuration"),
        RuntimeError("Failed starting Temporal dev server: download failed"),
        RuntimeError("Failed starting Temporal dev server: ConnectionRefused"),
        RuntimeError(
            "Failed starting Temporal dev server: Failed connecting to test server "
            "after 5 seconds, last error: None"
        ),
        ValueError(STARTUP_TIMEOUT),
        asyncio.CancelledError(),
    ],
)
async def test_other_startup_errors_are_not_retried(error: BaseException) -> None:
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
