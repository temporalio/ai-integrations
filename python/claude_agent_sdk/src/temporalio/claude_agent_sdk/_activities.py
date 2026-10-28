"""The Activities a :class:`TemporalTransport` runs, which reach Claude Code through a :class:`Connector`."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import ClaudeSDKError, CLIConnectionError

from temporalio import activity

from ._constants import CLOSE, CONNECT, CONNECTION_LOST, END_INPUT, READ, WRITE
from ._errors import activity_error
from ._heartbeat_decorator import auto_heartbeater
from ._types import (
    ClaudeCodeProcess,
    CloseActivityInput,
    ConnectActivityInput,
    Connection,
    Connector,
    EndInputActivityInput,
    Output,
    ReadActivityInput,
    ReadResult,
    WriteActivityInput,
)


@dataclass
class _Connection:
    """A Claude Code ``connect`` started, with the output the Workflow hasn't confirmed it received."""

    process: ClaudeCodeProcess
    unreceived: list[Output] = field(default_factory=list[Output])
    """Output a read returned, kept until the next read confirms the Workflow got it."""
    received: int = 0
    """How many outputs the Workflow received before ``unreceived``."""
    reading: asyncio.Future[list[Output]] | None = None
    """The process read that's still running."""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def read_process(self) -> list[Output]:
        """Wait for the process's next output.

        A cancelled attempt leaves the read running, and the next attempt waits
        for it, so no output is lost.
        """
        if self.reading is None:
            self.reading = asyncio.ensure_future(self.process.read())
        reading = self.reading
        try:
            return await asyncio.shield(reading)
        finally:
            # Once it's done, even with an error, the next attempt reads again.
            # A retryable error, such as a dropped network call, can then succeed.
            if reading.done():
                self.reading = None


class ConnectionActivities:
    """The Activities, with the connector they use, the Claude Codes they started, and this process's own task queue."""

    def __init__(self, connector: Connector, task_queue: str) -> None:
        """Create the Activities, with no Claude Code running.

        Args:
            connector: Starts Claude Code and reaches it.
            task_queue: The task queue only this process polls, where the
                Activities after ``connect`` must run.
        """
        self._connector = connector
        self._task_queue = task_queue
        self._connections: dict[str, _Connection] = {}
        # This Worker is shutting down and stops every Claude Code it's connected
        # to, so each Activity that reaches one fails with CONNECTION_LOST.
        self._shutting_down = False

    def connected_activities(self) -> list[Callable[..., Any]]:
        """The Activities that reach a Claude Code ``connect`` started, so must run in this process."""
        return [self.write, self.end_input, self.read, self.close]

    @activity.defn(name=CONNECT)
    @auto_heartbeater
    async def connect(self, request: ConnectActivityInput) -> Connection:
        """Start Claude Code, unless this launch started it already."""
        info = activity.info()
        handle = f"{info.workflow_run_id}/{info.activity_id}"

        if handle not in self._connections:
            # Shielded, so a cancelled launch still gets the process to close it.
            connecting = asyncio.ensure_future(self._connector.connect(request))
            try:
                process = await asyncio.shield(connecting)
                self._connections[handle] = _Connection(process)
            except asyncio.CancelledError:
                # The Workflow never gets the handle, so nothing else can close it.
                with contextlib.suppress(Exception):
                    await (await connecting).close()
                raise
            except ClaudeSDKError as error:
                raise activity_error(error) from error

        return Connection(handle, self._task_queue)

    @activity.defn(name=WRITE)
    async def write(self, request: WriteActivityInput) -> None:
        """Write to Claude Code's stdin."""
        try:
            await self._connection(request.handle).process.write(request.text)
        except ClaudeSDKError as error:
            raise activity_error(error) from error

    @activity.defn(name=END_INPUT)
    async def end_input(self, request: EndInputActivityInput) -> None:
        """Close Claude Code's stdin."""
        try:
            await self._connection(request.handle).process.end_input()
        except ClaudeSDKError as error:
            raise activity_error(error) from error

    @activity.defn(name=READ)
    @auto_heartbeater
    async def read(self, request: ReadActivityInput) -> ReadResult:
        """Wait for Claude Code's output, and return what there is.

        The output stays here until a later read confirms the Workflow got it.
        The server drops the result of an attempt it timed out, so the retry
        returns the same output again.
        """
        try:
            connection = self._connection(request.handle)

            async with connection.lock:
                if request.received > connection.received:
                    del connection.unreceived[: request.received - connection.received]
                    connection.received = request.received

                if not connection.unreceived:
                    try:
                        connection.unreceived = await connection.read_process()
                    finally:
                        # Report a lost connection if a shutdown began during a
                        # read that got no output, or cancelled it. Temporal
                        # would retry a cancelled read on this Worker's task
                        # queue, which no Worker polls once it's gone.
                        cancel = activity.cancellation_details()
                        if not connection.unreceived and (
                            (cancel is not None and cancel.worker_shutdown)
                            or (self._shutting_down and not activity.is_cancelled())
                        ):
                            raise CLIConnectionError(CONNECTION_LOST)

                return ReadResult(list(connection.unreceived))
        except ClaudeSDKError as error:
            raise activity_error(error) from error

    @activity.defn(name=CLOSE)
    @auto_heartbeater
    async def close(self, request: CloseActivityInput) -> None:
        """Close Claude Code and everything it started."""
        connection = self._connections.get(request.handle)
        if connection is None:  # A retry of a close that succeeded.
            return

        try:
            await connection.process.close()
            del self._connections[request.handle]
        except ClaudeSDKError as error:
            raise activity_error(error) from error

    async def shut_down(self) -> None:
        """Stop every Claude Code this process started, leaving their sessions resumable."""
        self._shutting_down = True
        connections, self._connections = self._connections, {}

        async def stop(process: ClaudeCodeProcess) -> None:
            await process.terminate()
            await process.close()

        await asyncio.gather(
            *(stop(connection.process) for connection in connections.values()),
            return_exceptions=True,
        )

    def _raise_if_shutting_down(self) -> None:
        if self._shutting_down:
            raise CLIConnectionError(CONNECTION_LOST)

    def _connection(self, handle: str) -> _Connection:
        """The Claude Code ``connect`` started, if this Worker still holds it."""
        self._raise_if_shutting_down()
        connection = self._connections.get(handle)
        if connection is None:
            # Closed, or started by an earlier run, such as one a reset replaced.
            raise CLIConnectionError(
                "Claude Code is not running. To continue the conversation, start "
                "a new client with options.resume set to the session ID."
            )
        return connection
