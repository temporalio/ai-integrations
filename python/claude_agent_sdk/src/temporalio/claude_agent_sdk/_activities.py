"""The Activities a :class:`TemporalTransport` runs, which reach Claude Code through a :class:`Connector`."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Any

from claude_agent_sdk import ClaudeSDKError

from temporalio import activity

from ._constants import CLOSE, CONNECT, END_INPUT, READ, WRITE
from ._errors import activity_error
from ._types import (
    ClaudeCodeProcess,
    CloseActivityInput,
    ConnectActivityInput,
    Connection,
    Connector,
    EndInputActivityInput,
    ReadActivityInput,
    ReadResult,
    WriteActivityInput,
)


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
        self._processes: dict[str, ClaudeCodeProcess] = {}

    def connected_activities(self) -> list[Callable[..., Any]]:
        """The Activities that reach a Claude Code ``connect`` started, so must run in this process."""
        return [self.write, self.end_input, self.read, self.close]

    @activity.defn(name=CONNECT)
    async def connect(self, request: ConnectActivityInput) -> Connection:
        """Start Claude Code, unless this launch started it already."""
        info = activity.info()
        handle = f"{info.workflow_run_id}/{info.activity_id}"

        if handle not in self._processes:
            # Shielded, so a cancelled launch still gets the process to close it.
            connecting = asyncio.ensure_future(self._connector.connect(request))
            try:
                async with _heartbeating():
                    self._processes[handle] = await asyncio.shield(connecting)
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
            await self._processes[request.handle].write(request.text)
        except ClaudeSDKError as error:
            raise activity_error(error) from error

    @activity.defn(name=END_INPUT)
    async def end_input(self, request: EndInputActivityInput) -> None:
        """Close Claude Code's stdin."""
        try:
            await self._processes[request.handle].end_input()
        except ClaudeSDKError as error:
            raise activity_error(error) from error

    @activity.defn(name=READ)
    async def read(self, request: ReadActivityInput) -> ReadResult:
        """Wait for Claude Code's output, and return what there is."""
        try:
            process = self._processes[request.handle]
            async with _heartbeating():
                output = await process.read()
            return ReadResult(output)
        except ClaudeSDKError as error:
            raise activity_error(error) from error

    @activity.defn(name=CLOSE)
    async def close(self, request: CloseActivityInput) -> None:
        """Close Claude Code and everything it started."""
        process = self._processes.get(request.handle)
        if process is None:  # A retry of a close that succeeded.
            return

        try:
            async with _heartbeating():
                await process.close()
            del self._processes[request.handle]
        except ClaudeSDKError as error:
            raise activity_error(error) from error


@contextlib.asynccontextmanager
async def _heartbeating() -> AsyncIterator[None]:
    """Heartbeat every second while the block runs."""

    async def beat() -> None:
        while True:
            activity.heartbeat()
            await asyncio.sleep(1)

    beats = asyncio.create_task(beat())
    try:
        yield
    finally:
        beats.cancel()
