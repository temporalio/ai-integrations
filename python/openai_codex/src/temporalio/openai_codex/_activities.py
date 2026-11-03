"""The Activities a :class:`TemporalCodexClient` runs, which reach Codex through a :class:`Connector`."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from openai_codex import CodexError, TransportClosedError
from temporalio import activity
from temporalio.service import PayloadLimitsConfig

from ._constants import CLOSE, CONNECT, CONNECTION_LOST, READ, RESERVE, WRITE
from ._errors import activity_error
from ._heartbeat_decorator import auto_heartbeater
from ._types import (
    CloseActivityInput,
    CodexProcess,
    ConnectActivityInput,
    Connection,
    Connector,
    Message,
    ReadActivityInput,
    ReadResult,
    WriteActivityInput,
)


@dataclass
class _Connection:
    """A Codex ``connect`` started, with the output the Workflow hasn't confirmed it received."""

    process: CodexProcess
    unreceived: list[Message] = field(default_factory=list[Message])
    """Message a read returned, kept until the next read confirms the Workflow got it."""
    received: int = 0
    """How many outputs the Workflow received before ``unreceived``."""
    reading: asyncio.Future[list[Message]] | None = None
    """The process read that's still running."""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    writes: dict[int, asyncio.Task[None]] = field(default_factory=dict)
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def write_process(self, message: Message) -> None:
        """Keep later writes behind an earlier attempt that outlived its caller."""
        async with self.write_lock:
            await self.process.write(message)

    async def close(self) -> None:
        """Close the process and release retained read and write tasks."""
        await self.process.close()
        pending: list[asyncio.Future[Any]] = list(self.writes.values())
        if self.reading is not None:
            pending.append(self.reading)
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async def read_process(self) -> list[Message]:
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
    """One Worker's Activities, with the connector they use, the Codex processes they started, and their connector queue."""

    def __init__(self, connector: Connector, connector_queue: str) -> None:
        """Create the Activities, with no Codex running.

        Args:
            connector: Starts Codex and reaches it.
            connector_queue: The task queue only this process polls, where the
                Activities after ``connect`` must run.
        """
        self._connector = connector
        self.connector_queue = connector_queue
        # Duplicate plugin instances share this owner's connector Worker.
        self.connector_worker_running = False
        self._connections: dict[str, _Connection] = {}
        self._closed_handles: set[str] = set()
        self._connect_lock = asyncio.Lock()

    def connected_activities(self) -> list[Callable[..., Any]]:
        """The Activities that reach a Codex ``connect`` started, so must run in this process."""
        return [self.connect, self.write, self.read, self.close]

    @activity.defn(name=RESERVE)
    async def reserve(self, request: None) -> Connection:
        """Select an owner before launching, without allocating a process."""
        del request
        info = activity.info()
        return Connection(
            f"{info.workflow_run_id}/{info.activity_id}", self.connector_queue
        )

    @activity.defn(name=CONNECT)
    @auto_heartbeater
    async def connect(self, request: ConnectActivityInput) -> Connection:
        """Start Codex, unless this launch started it already."""
        info = activity.info()
        handle = request.handle or f"{info.workflow_run_id}/{info.activity_id}"

        async with self._connect_lock:
            if handle in self._closed_handles:
                raise activity_error(
                    TransportClosedError("Codex connection was closed")
                )
            if handle not in self._connections:
                # Shielded, so a cancelled launch still gets the process to close it.
                connecting = asyncio.ensure_future(self._connector.connect(request))
                try:
                    process = await asyncio.shield(connecting)
                    if handle in self._closed_handles:
                        await process.close()
                        raise TransportClosedError("Codex connection was closed")
                    self._connections[handle] = _Connection(process)
                except asyncio.CancelledError:
                    # Finish a launch that outlived cancellation, then dispose of it.
                    with contextlib.suppress(Exception):
                        await (await connecting).close()
                    raise
                except CodexError as error:
                    raise activity_error(error) from error
                except (FileNotFoundError, PermissionError) as error:
                    raise activity_error(error) from error

        return Connection(handle, self.connector_queue)

    @activity.defn(name=WRITE)
    @auto_heartbeater
    async def write(self, request: WriteActivityInput) -> None:
        """Write to Codex's stdin."""
        try:
            connection = self._connection(request.handle)
            # One task per logical write. Shielding retains in-flight writes when
            # attempts overlap or the completion is lost. Completed tasks remain
            # cached until close, including permanent failures.
            task = connection.writes.get(request.operation_id)
            if task is None:
                task = asyncio.create_task(connection.write_process(request.message))
                connection.writes[request.operation_id] = task
            try:
                await asyncio.shield(task)
            except Exception as error:
                if (
                    not isinstance(error, CodexError)
                    and connection.writes.get(request.operation_id) is task
                ):
                    connection.writes.pop(request.operation_id, None)
                raise
        except CodexError as error:
            raise activity_error(error) from error

    @activity.defn(name=READ)
    @auto_heartbeater
    async def read(self, request: ReadActivityInput) -> ReadResult:
        """Wait for Codex's output, and return as much as fits Temporal's payload limit.

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
                        cancel = activity.cancellation_details()
                        if cancel is not None and cancel.worker_shutdown:
                            raise TransportClosedError(CONNECTION_LOST)

                return ReadResult(_fit_payload(connection.unreceived))
        except CodexError as error:
            raise activity_error(error) from error

    @activity.defn(name=CLOSE)
    @auto_heartbeater
    async def close(self, request: CloseActivityInput) -> None:
        """Close Codex and everything it started."""
        # A cancelled launch can still commit after this Activity runs. The
        # known identity lets that launch close its process instead of storing it.
        self._closed_handles.add(request.handle)
        connection = self._connections.get(request.handle)
        if connection is None:  # A retry of a close that succeeded.
            return

        try:
            await connection.close()
            self._connections.pop(request.handle, None)
        except CodexError as error:
            raise activity_error(error) from error

    async def shut_down(self) -> None:
        """Close every Codex this Worker started."""
        connections, self._connections = self._connections, {}

        results = await asyncio.gather(
            *(connection.close() for connection in connections.values()),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, BaseException):
                raise result

    def _connection(self, handle: str) -> _Connection:
        """The Codex ``connect`` started, if this Worker still holds it."""
        connection = self._connections.get(handle)
        if connection is None:
            # Closed, or started by an earlier run, such as one a reset replaced.
            raise TransportClosedError(CONNECTION_LOST)
        return connection


def _fit_payload(outputs: list[Message]) -> list[Message]:
    """The most of ``outputs``, from the start, that fits Temporal's payload limit, and at least one."""
    limit = PayloadLimitsConfig().payloads_warn_size
    size = 2  # The surrounding JSON list.
    for count, output in enumerate(outputs):
        size += len(json.dumps(output).encode()) + (2 if count else 0)  # ", "
        if count and size > limit:
            return outputs[:count]
    return list(outputs)
