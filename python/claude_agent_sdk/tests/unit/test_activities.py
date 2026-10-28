"""The plugin's Activities, run as ActivityEnvironment runs them, with a shell script as the CLI."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from claude_agent_sdk import (
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
)

from temporalio.activity import ActivityCancellationDetails
from temporalio.claude_agent_sdk import (
    ClaudeCodeProcess,
    ConnectActivityInput,
    SubprocessConnector,
)
from temporalio.claude_agent_sdk._activities import ConnectionActivities
from temporalio.claude_agent_sdk._constants import CONNECTION_LOST
from temporalio.claude_agent_sdk._errors import activity_error, sdk_error
from temporalio.claude_agent_sdk._types import (
    CloseActivityInput,
    EndInputActivityInput,
    Output,
    ReadActivityInput,
    ReadResult,
    WriteActivityInput,
)
from temporalio.exceptions import ActivityError, ApplicationError, RetryState
from temporalio.service import PayloadLimitsConfig
from temporalio.testing import ActivityEnvironment
from tests.helpers.harness import CLAUDE, LocalCli

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not CLAUDE.is_file(),
    reason="needs sh and the Claude Agent SDK's bundled Claude Code",
)

# The plugin's own task queue for this process.
OWN_QUEUE = "claude-agent-sdk-own"


@pytest.fixture
def cli(tmp_path: Path) -> LocalCli:
    return LocalCli(tmp_path)


@pytest.fixture
def activities() -> ConnectionActivities:
    return ConnectionActivities(SubprocessConnector(), OWN_QUEUE)


def request(cli: LocalCli, script: str = "exec cat") -> ConnectActivityInput:
    """The input to start a CLI that runs ``script``, by default echoing its stdin."""
    cli.set_cli(script)
    return ConnectActivityInput([], {}, cli_path=str(cli.cli_path))


async def test_a_read_returns_the_next_output_until_claude_code_exits(
    cli: LocalCli, activities: ConnectionActivities
) -> None:
    run = ActivityEnvironment().run
    launch = request(cli)

    launched = await run(activities.connect, launch)
    # A retried launch starts nothing more.
    assert await run(activities.connect, launch) == launched
    # Only this Worker can reach the subprocess, so the rest goes to its own queue.
    assert launched.task_queue == OWN_QUEUE
    handle = launched.handle
    await run(activities.write, WriteActivityInput(handle, '{"n": 1}\n'))
    assert await run(activities.read, ReadActivityInput(handle, 0)) == ReadResult(
        [{"n": 1}]
    )
    await run(activities.write, WriteActivityInput(handle, '{"n": 2}\n'))
    assert await run(activities.read, ReadActivityInput(handle, 1)) == ReadResult(
        [{"n": 2}]
    )

    await run(activities.end_input, EndInputActivityInput(handle))

    assert await run(activities.read, ReadActivityInput(handle, 2)) == ReadResult()
    await run(activities.close, CloseActivityInput(handle))
    # A retried close, such as after its result was lost, does nothing.
    await run(activities.close, CloseActivityInput(handle))
    assert len(cli.cli_pids()) == 1
    assert cli.running() == []


# Stands in for Claude Code: records whether it got SIGTERM or saw its stdin close.
SIGTERM_OR_EOF = """
import pathlib, signal, sys
seen = pathlib.Path(sys.argv[1])
def on_sigterm(signum, frame):
    seen.write_text("SIGTERM")
    sys.exit(0)
signal.signal(signal.SIGTERM, on_sigterm)
seen.with_suffix(".ready").touch()
sys.stdin.read()
seen.write_text("EOF")
"""


async def test_shutting_down_stops_claude_code_with_sigterm_before_its_stdin_closes(
    cli: LocalCli, activities: ConnectionActivities, tmp_path: Path
) -> None:
    """On SIGTERM, Claude Code leaves a pending tool permission for the resumed session to ask."""
    stand_in = tmp_path / "claude.py"
    stand_in.write_text(SIGTERM_OR_EOF)
    seen = tmp_path / "seen"
    launched = await ActivityEnvironment().run(
        activities.connect,
        request(cli, f'exec "{sys.executable}" "{stand_in}" "{seen}"'),
    )
    while not seen.with_suffix(".ready").exists():
        await asyncio.sleep(0.05)
    reading = asyncio.create_task(
        ActivityEnvironment().run(
            activities.read, ReadActivityInput(launched.handle, 0)
        )
    )
    await asyncio.sleep(0.1)

    await activities.shut_down()

    assert seen.read_text() == "SIGTERM"
    assert cli.running() == []
    # The read fails for good, so the Workflow resumes the session now, not
    # after its retry times out on this Worker's task queue.
    with pytest.raises(ApplicationError) as failed:
        await reading
    assert failed.value.type == "CLIConnectionError"
    assert failed.value.non_retryable
    assert failed.value.message == CONNECTION_LOST


async def test_a_read_the_worker_shutdown_cancelled_fails_for_good(
    cli: LocalCli, activities: ConnectionActivities
) -> None:
    """Such as when Claude Code takes a while to exit on SIGTERM."""
    handle = (await ActivityEnvironment().run(activities.connect, request(cli))).handle
    env = ActivityEnvironment()
    reading = asyncio.create_task(
        env.run(activities.read, ReadActivityInput(handle, 0))
    )
    await asyncio.sleep(0.1)

    env.cancel(ActivityCancellationDetails(worker_shutdown=True))

    # Not cancelled, which Temporal would retry on this Worker's task queue.
    with pytest.raises(ApplicationError) as failed:
        await reading
    assert failed.value.non_retryable
    assert failed.value.message == CONNECTION_LOST
    await activities.shut_down()


class WritesOnSigterm:
    """Stands in for Claude Code that writes once it gets SIGTERM."""

    def __init__(self) -> None:
        self.stopped = asyncio.Event()

    async def read(self) -> list[Output]:
        await self.stopped.wait()
        return [{"n": 0}]

    async def terminate(self) -> None:
        self.stopped.set()

    async def write(self, data: str) -> None:
        del data

    async def end_input(self) -> None: ...
    async def close(self) -> None: ...


async def test_a_read_that_got_output_during_a_shutdown_returns_it() -> None:
    """Only the next read fails, so no output is dropped."""
    process = WritesOnSigterm()

    class Connector:
        async def connect(self, request: ConnectActivityInput) -> WritesOnSigterm:
            del request
            return process

    activities = ConnectionActivities(Connector(), OWN_QUEUE)
    run = ActivityEnvironment().run
    handle = (await run(activities.connect, ConnectActivityInput([], {}))).handle
    reading = asyncio.create_task(run(activities.read, ReadActivityInput(handle, 0)))
    await asyncio.sleep(0.1)

    await activities.shut_down()

    assert await reading == ReadResult([{"n": 0}])
    with pytest.raises(ApplicationError) as failed:
        await run(activities.read, ReadActivityInput(handle, 1))
    assert failed.value.message == CONNECTION_LOST


async def test_a_read_returns_at_most_about_a_mebibyte(
    cli: LocalCli, activities: ConnectionActivities
) -> None:
    """To stay under Temporal's payload limit: the next read returns the rest."""
    line = """printf '{"x": "'; head -c 600000 /dev/zero | tr '\\0' x; echo '"}'"""
    launched = await ActivityEnvironment().run(
        activities.connect, request(cli, f"{line}; {line}; {line}")
    )

    counts: list[int] = []
    while True:
        read = ReadActivityInput(launched.handle, sum(counts))
        result = await ActivityEnvironment().run(activities.read, read)
        counts.append(len(result.output))
        if not result.output:
            break

    assert counts == [1, 1, 1, 0]
    await ActivityEnvironment().run(
        activities.close, CloseActivityInput(launched.handle)
    )


async def test_a_read_includes_stderr_in_the_payload_limit(cli: LocalCli) -> None:
    line = "head -c 600000 /dev/zero | tr '\\0' x >&2; echo >&2"
    process = await SubprocessConnector().connect(
        request(cli, f"{line}; {line}; echo '{{\"n\": 1}}'")
    )

    batches: list[list[Output]] = []
    while output := await process.read():
        batches.append(output)

    limit = PayloadLimitsConfig().payloads_warn_size
    assert all(
        len(json.dumps(batch).encode()) <= limit or len(batch) == 1 for batch in batches
    )
    assert [item for batch in batches for item in batch][-1] == {"n": 1}
    await process.close()


async def test_a_read_returns_all_the_output_already_written(cli: LocalCli) -> None:
    process = await SubprocessConnector().connect(
        request(cli, """printf '{"n": 1}\\n{"n": 2}\\n'; sleep 1""")
    )

    assert await process.read() == [{"n": 1}, {"n": 2}]
    await process.close()


async def test_a_cancelled_read_leaves_the_next_output_to_its_retry(
    cli: LocalCli, activities: ConnectionActivities
) -> None:
    """Such as an attempt that timed out, cancelled while its retry runs."""
    run = ActivityEnvironment().run
    handle = (await run(activities.connect, request(cli))).handle

    attempt_env = ActivityEnvironment()
    attempt = asyncio.create_task(
        attempt_env.run(activities.read, ReadActivityInput(handle, 0))
    )
    await asyncio.sleep(0.1)
    attempt_env.cancel()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    await run(activities.write, WriteActivityInput(handle, '{"n": 1}\n'))

    assert await run(activities.read, ReadActivityInput(handle, 0)) == ReadResult(
        [{"n": 1}]
    )
    await run(activities.close, CloseActivityInput(handle))


async def test_a_read_whose_result_was_lost_returns_the_same_output_again(
    cli: LocalCli, activities: ConnectionActivities
) -> None:
    """Such as an attempt the server timed out while this Worker still ran it. The server drops its result."""
    run = ActivityEnvironment().run
    handle = (await run(activities.connect, request(cli))).handle
    await run(activities.write, WriteActivityInput(handle, '{"n": 1}\n'))
    read = ReadActivityInput(handle, 0)
    assert await run(activities.read, read) == ReadResult([{"n": 1}])

    # The retry still says the Workflow received nothing.
    assert await run(activities.read, read) == ReadResult([{"n": 1}])
    await run(activities.close, CloseActivityInput(handle))


async def test_a_read_retries_an_error_the_connector_says_a_retry_can_fix() -> None:
    """Such as a dropped network call to a remote Claude Code."""
    reads: list[str] = []

    class FlakyProcess:
        async def write(self, data: str) -> None:
            del data

        async def end_input(self) -> None: ...
        async def close(self) -> None: ...
        async def terminate(self) -> None: ...

        async def read(self) -> list[Output]:
            reads.append("read")
            if len(reads) == 1:
                raise ConnectionResetError("dropped")
            return [{"n": 1}]

    class FlakyConnector:
        async def connect(self, request: ConnectActivityInput) -> ClaudeCodeProcess:
            del request
            return FlakyProcess()

    run = ActivityEnvironment().run
    activities = ConnectionActivities(FlakyConnector(), OWN_QUEUE)
    handle = (await run(activities.connect, ConnectActivityInput([], {}))).handle
    read = ReadActivityInput(handle, 0)

    with pytest.raises(ConnectionResetError):
        await run(activities.read, read)

    assert await run(activities.read, read) == ReadResult([{"n": 1}])


async def test_a_read_returns_stderr_before_the_output(cli: LocalCli) -> None:
    process = await SubprocessConnector().connect(
        request(cli, """echo warning >&2; sleep 0.5; echo '{"n": 1}'""")
    )

    assert await process.read() == ["warning", {"n": 1}]
    assert await process.read() == []
    await process.close()


async def test_a_nonzero_exit_fails_the_read_for_good(
    cli: LocalCli, activities: ConnectionActivities
) -> None:
    run = ActivityEnvironment().run
    launched = await run(activities.connect, request(cli, "exit 3"))

    with pytest.raises(ApplicationError) as raised:
        await run(activities.read, ReadActivityInput(launched.handle, 0))

    assert raised.value.type == "ProcessError"
    assert raised.value.non_retryable
    assert raised.value.details[0] == 3
    await run(activities.close, CloseActivityInput(launched.handle))


@pytest.mark.parametrize(
    ("connect", "error", "reason"),
    [
        (
            ConnectActivityInput([], {}, cwd="/missing"),
            "CLIConnectionError",
            "Working directory",
        ),
        (
            ConnectActivityInput([], {}, cli_path="/missing/claude"),
            "CLINotFoundError",
            "/missing/claude",
        ),
    ],
)
async def test_what_no_retry_can_fix_fails_for_good(
    activities: ConnectionActivities,
    connect: ConnectActivityInput,
    error: str,
    reason: str,
) -> None:
    with pytest.raises(ApplicationError) as raised:
        await ActivityEnvironment().run(activities.connect, connect)

    assert raised.value.type == error
    assert raised.value.non_retryable
    assert reason in raised.value.message


async def test_a_launch_cancelled_while_starting_closes_what_it_started(
    cli: LocalCli,
) -> None:
    """Such as a Workflow cancelled during the launch, which never gets the handle to close it."""
    starting = asyncio.Event()
    started: list[ClaudeCodeProcess] = []

    class SlowConnector:
        async def connect(self, request: ConnectActivityInput) -> ClaudeCodeProcess:
            starting.set()
            await asyncio.sleep(0.5)
            process = await SubprocessConnector().connect(request)
            started.append(process)
            return process

    env = ActivityEnvironment()
    activities = ConnectionActivities(SlowConnector(), OWN_QUEUE)
    launch = asyncio.create_task(env.run(activities.connect, request(cli)))
    await starting.wait()
    env.cancel()

    with pytest.raises(asyncio.CancelledError):
        await launch

    [process] = started
    with pytest.raises(CLIConnectionError):
        await process.write("{}\n")


@pytest.mark.parametrize(
    "error",
    [
        ProcessError("Command failed", exit_code=3, stderr="bad flag"),
        CLINotFoundError("Claude Code not found"),
        CLIJSONDecodeError('{"x', ValueError("Unterminated string")),
        CLIConnectionError("Claude Code is not running"),
    ],
)
def test_the_transport_raises_the_error_the_activity_failed_with(
    error: ClaudeSDKError,
) -> None:
    failed = ActivityError(
        "Activity failed",
        scheduled_event_id=1,
        started_event_id=2,
        identity="worker",
        activity_type="claude-agent-sdk-read",
        activity_id="1",
        retry_state=RetryState.NON_RETRYABLE_FAILURE,
    )
    failed.__cause__ = activity_error(error)

    raised = sdk_error(failed)

    assert type(raised) is type(error)
    assert str(raised) == str(error)
    # Such as ProcessError's exit_code, and CLIJSONDecodeError's original_error.
    assert repr(vars(raised)) == repr(vars(error))
