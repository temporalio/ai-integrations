"""The types shared by a :class:`TemporalTransport`, its Activities, and how a Worker reaches Claude Code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from temporalio import workflow

Output = dict[str, Any] | str
"""A message from Claude Code's stdout, or a line of its stderr."""


class TransportActivityOptions(TypedDict, total=False):
    """Options for each Activity a :class:`TemporalTransport` runs, overriding its own.

    ``task_queue`` applies only to ``connect``. The other Activities always run on
    the Worker that started Claude Code, since only it holds the process.
    """

    connect: workflow.ActivityConfig
    """Starts Claude Code."""
    write: workflow.ActivityConfig
    """Writes to Claude Code's stdin."""
    end_input: workflow.ActivityConfig
    """Closes Claude Code's stdin."""
    read: workflow.ActivityConfig
    """Waits for Claude Code's output."""
    close: workflow.ActivityConfig
    """Closes Claude Code."""


@dataclass
class ConnectActivityInput:
    """How to start Claude Code, built from the options as in the Claude Agent SDK."""

    args: list[str]
    """Claude Code's flags."""
    env: dict[str, str]
    """The variables to add to Claude Code's environment."""
    cli_path: str | None = None
    """``options.cli_path``, or ``None`` to find Claude Code as in the Claude Agent SDK."""
    cwd: str | None = None
    """``options.cwd``."""
    user: str | None = None
    """``options.user``, the user to run Claude Code as."""
    max_buffer_size: int | None = None
    """``options.max_buffer_size``, the most a message from Claude Code can be."""


@dataclass
class Connection:
    """The connect Activity's result."""

    handle: str
    task_queue: str
    """The task queue of the other Activities, which only the launching Worker polls."""


@dataclass
class WriteActivityInput:
    """The write Activity's input."""

    handle: str
    text: str
    """One or more lines for Claude Code's stdin."""


@dataclass
class EndInputActivityInput:
    """The end input Activity's input."""

    handle: str


@dataclass
class ReadActivityInput:
    """The read Activity's input."""

    handle: str
    received: int


@dataclass
class CloseActivityInput:
    """The close Activity's input."""

    handle: str


@dataclass
class ReadResult:
    """The read Activity's result."""

    output: list[Output] = field(default_factory=list)
    """Claude Code's next output, or none once Claude Code exited with status 0."""


class ClaudeCodeProcess(Protocol):
    """A running Claude Code's stdin and stdout.

    ``write``, ``end_input`` and ``close`` must be safe to call from overlapping
    Activity attempts. The plugin calls ``read`` one at a time and cancels it
    only by closing.
    """

    async def write(self, data: str) -> None:
        """Write one or more lines to Claude Code's stdin."""
        ...

    async def end_input(self) -> None:
        """Close Claude Code's stdin. As in the Claude Agent SDK, it then exits once it's done."""
        ...

    async def read(self) -> list[Output]:
        """Wait until Claude Code writes output, then return everything it has written so far.

        Returns:
            At least one output, or an empty list once Claude Code exited with
            status 0 and all its output was read.

        Raises:
            ProcessError: If Claude Code exited with a nonzero status, as in
                the Claude Agent SDK.
        """
        ...

    async def close(self) -> None:
        """Close Claude Code's stdin and wait for it to exit, forcing it if it doesn't in time.

        Claude Code treats the end of its input as the end of the session: it
        wraps up, failing a pending tool permission, and exits.
        """
        ...

    async def terminate(self) -> None:
        """Stop Claude Code at once, as SIGTERM does, forcing it if it doesn't in time.

        Claude Code stops where it is, instead of wrapping up as it does when
        its input ends, so a resumed session continues the turn, including a
        pending tool permission. The plugin calls this, not ``close``, on
        Worker shutdown, so release here whatever ``close`` would.
        """
        ...


class Connector(Protocol):
    """Starts Claude Code and reaches it, wherever it runs.

    Raise the Claude Agent SDK's errors, such as ``CLINotFoundError``, only for
    what no retry can fix. Raise any other error, such as a dropped network
    call, to have it retried.
    """

    async def connect(self, request: ConnectActivityInput) -> ClaudeCodeProcess:
        """Start Claude Code.

        The plugin holds the returned :class:`ClaudeCodeProcess` until the
        ``close`` Activity, and runs every Activity that uses it on this
        Worker.
        """
        ...
