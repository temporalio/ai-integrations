"""The plugin's Activities, run as ActivityEnvironment runs them, with a shell script as the CLI."""

from __future__ import annotations

import asyncio
import dataclasses
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
from temporalio.testing import ActivityEnvironment
from tests.helpers.harness import CLAUDE, LocalCli

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not CLAUDE.is_file(),
    reason="needs sh and the Claude Agent SDK's bundled Claude Code",
)

# The task queue only this process polls.
CONNECTOR_QUEUE = "claude-agent-sdk-connector"


@pytest.fixture
def cli(tmp_path: Path) -> LocalCli:
    return LocalCli(tmp_path)


@pytest.fixture
def activities() -> ConnectionActivities:
    return ConnectionActivities(SubprocessConnector(), CONNECTOR_QUEUE)


def request(cli: LocalCli, script: str = "exec cat") -> ConnectActivityInput:
    """The input to start a CLI that runs ``script``, by default echoing its stdin."""
    cli.set_cli(script)
    return ConnectActivityInput([], {}, cli_path=str(cli.cli_path))


class ScriptedProcess:
    """A Claude Code whose reads return ``reads`` in turn, raising each that is an error."""

    def __init__(self, *reads: list[Output] | Exception) -> None:
        self.reads = list(reads)

    async def read(self) -> list[Output]:
        result = self.reads.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def write(self, data: str) -> None:
        del data

    async def end_input(self) -> None: ...
    async def close(self) -> None: ...
    async def terminate(self) -> None: ...


def scripted(*reads: list[Output] | Exception) -> ConnectionActivities:
    """The Activities, with a :class:`ScriptedProcess` of ``reads`` as Claude Code."""

    class ScriptedConnector:
        async def connect(self, request: ConnectActivityInput) -> ClaudeCodeProcess:
            del request
            return ScriptedProcess(*reads)

    return ConnectionActivities(ScriptedConnector(), CONNECTOR_QUEUE)


async def test_a_read_returns_the_next_output_until_claude_code_exits(
    cli: LocalCli, activities: ConnectionActivities
) -> None:
    run = ActivityEnvironment().run
    launch = request(cli)

    launched = await run(activities.connect, launch)
    # A retried launch starts nothing more.
    assert await run(activities.connect, launch) == launched
    # Only this Worker can reach the subprocess, so the rest goes to its connector queue.
    assert launched.task_queue == CONNECTOR_QUEUE
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
    await ActivityEnvironment().run(
        activities.connect,
        request(cli, f'exec "{sys.executable}" "{stand_in}" "{seen}"'),
    )
    while not seen.with_suffix(".ready").exists():
        await asyncio.sleep(0.05)

    await activities.shut_down()

    assert seen.read_text() == "SIGTERM"
    assert cli.running() == []


async def test_shutting_down_stops_every_claude_code_then_raises_what_failed() -> None:
    """Such as a remote connector whose terminate call fails, which nothing would retry."""
    stopped: list[bool] = []

    class FailsToTerminate(ScriptedProcess):
        async def terminate(self) -> None:
            raise OSError("terminate failed")

    class Terminates(ScriptedProcess):
        async def terminate(self) -> None:
            stopped.append(True)

    processes: list[ClaudeCodeProcess] = [FailsToTerminate(), Terminates()]

    class Connector:
        async def connect(self, request: ConnectActivityInput) -> ClaudeCodeProcess:
            del request
            return processes.pop(0)

    activities = ConnectionActivities(Connector(), CONNECTOR_QUEUE)
    for activity_id in ("1", "2"):
        env = ActivityEnvironment()
        env.info = dataclasses.replace(env.info, activity_id=activity_id)
        await env.run(activities.connect, ConnectActivityInput([], {}))

    with pytest.raises(OSError, match="terminate failed"):
        await activities.shut_down()

    assert stopped == [True]


async def test_a_read_the_worker_shutdown_cancelled_fails_for_good(
    cli: LocalCli, activities: ConnectionActivities
) -> None:
    """As a Worker shutdown does at once, before the plugin stops Claude Code."""
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


async def test_a_read_returns_what_fits_the_payload_limit() -> None:
    """At least one output, even one over the limit on its own. The next read returns the rest."""
    big = "x" * 600_000
    run = ActivityEnvironment().run
    activities = scripted([big, {"n": 1}, {"n": 2}, big])
    handle = (await run(activities.connect, ConnectActivityInput([], {}))).handle

    reads = [
        (await run(activities.read, ReadActivityInput(handle, received))).output
        for received in (0, 1, 3)
    ]

    assert reads == [[big], [{"n": 1}, {"n": 2}], [big]]


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
    run = ActivityEnvironment().run
    activities = scripted(ConnectionResetError("dropped"), [{"n": 1}])
    handle = (await run(activities.connect, ConnectActivityInput([], {}))).handle
    read = ReadActivityInput(handle, 0)

    with pytest.raises(ConnectionResetError):
        await run(activities.read, read)

    assert await run(activities.read, read) == ReadResult([{"n": 1}])


async def test_reads_return_stderr_and_output_in_the_order_they_arrive(
    cli: LocalCli,
) -> None:
    """Up to Claude Code's exit, even stderr written just before it."""
    process = await SubprocessConnector().connect(
        request(cli, """echo '{"n": 1}'; sleep 0.5; echo warning >&2""")
    )

    output: list[Output] = []
    while batch := await process.read():
        output += batch

    assert output == [{"n": 1}, "warning"]
    # Every later read ends the same way.
    assert await process.read() == []
    await process.close()


@pytest.mark.parametrize(
    ("script", "error", "reason"),
    [
        ("exit 3", "ProcessError", "exit code 3"),
        # Not UTF-8, which the Claude Agent SDK fails to decode.
        ("printf '\\377\\n'", "CLIConnectionError", "can't decode byte 0xff"),
    ],
)
async def test_a_nonzero_exit_or_unreadable_output_fails_every_read_for_good(
    cli: LocalCli,
    activities: ConnectionActivities,
    script: str,
    error: str,
    reason: str,
) -> None:
    run = ActivityEnvironment().run
    handle = (await run(activities.connect, request(cli, script))).handle

    for _ in range(2):
        with pytest.raises(ApplicationError) as raised:
            await run(activities.read, ReadActivityInput(handle, 0))
        assert raised.value.type == error
        assert raised.value.non_retryable
        assert reason in raised.value.message
    await run(activities.close, CloseActivityInput(handle))


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
    activities = ConnectionActivities(SlowConnector(), CONNECTOR_QUEUE)
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
