"""Local server setup retries exceptions with a bounded attempt count."""

import asyncio
from functools import partial
from unittest.mock import AsyncMock, call, patch

import pytest

from temporalio.testing import WorkflowEnvironment
from tests.helpers.environment import start_local_with_retry


@pytest.mark.parametrize("error_type", [RuntimeError, OSError, ValueError])
@pytest.mark.parametrize("failures", [0, 1, 2])
async def test_server_startup_recovers(
    error_type: type[Exception], failures: int
) -> None:
    environment = AsyncMock(spec=WorkflowEnvironment)
    start = AsyncMock(
        side_effect=[error_type("server setup failed") for _ in range(failures)]
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

    assert start.await_args_list == [call(**options)] * (failures + 1)
    assert sleep.await_args_list == [call(1)] * failures


@pytest.mark.parametrize("error_type", [RuntimeError, OSError, ValueError])
async def test_server_startup_stops_after_three_attempts(
    error_type: type[Exception],
) -> None:
    error = error_type("server setup failed")
    start = AsyncMock(side_effect=error)
    with (
        patch(
            "tests.helpers.environment.asyncio.sleep", new_callable=AsyncMock
        ) as sleep,
        pytest.raises(error_type) as raised,
    ):
        await start_local_with_retry(start)

    assert raised.value is error
    assert start.await_count == 3
    assert sleep.await_args_list == [call(1), call(1)]


async def test_different_startup_errors_share_retry_limit() -> None:
    error = RuntimeError("server startup failed")
    start = AsyncMock(
        side_effect=[
            OSError("download failed"),
            ValueError("invalid archive"),
            error,
        ]
    )
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
    [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit()],
)
async def test_cancellation_and_process_exit_are_not_retried(
    error: BaseException,
) -> None:
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


async def test_cancellation_during_retry_delay_is_not_retried() -> None:
    start = AsyncMock(side_effect=OSError("download failed"))
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
