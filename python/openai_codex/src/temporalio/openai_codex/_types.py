"""Activity inputs and extension points for a Codex connection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from openai_codex import CodexConfig
from temporalio import workflow

Message = dict[str, Any]


class CodexActivityOptions(TypedDict, total=False):
    """Activity overrides; only connect may select a task queue."""

    connect: workflow.ActivityConfig
    write: workflow.ActivityConfig
    read: workflow.ActivityConfig
    close: workflow.ActivityConfig


@dataclass
class ConnectActivityInput:
    """Launch configuration, resolved on the Activity Worker."""

    config: CodexConfig = field(default_factory=CodexConfig)
    handle: str | None = None
    """Reserved connection identity, also useful for idempotent remote launches."""


@dataclass
class Connection:
    """A connection and the queue of the Worker that owns it."""

    handle: str
    task_queue: str


@dataclass
class WriteActivityInput:
    """A write whose operation ID is stable across Activity attempts."""

    handle: str
    operation_id: int
    message: Message


@dataclass
class ReadActivityInput:
    """Read output, acknowledging the number of messages already received."""

    handle: str
    received: int


@dataclass
class ReadResult:
    """An ordered batch, retained until the next acknowledgement."""

    output: list[Message] = field(default_factory=list)


@dataclass
class CloseActivityInput:
    """The connection to close."""

    handle: str


class CodexProcess(Protocol):
    """A Codex app-server connection owned by one Worker.

    Reads are serialized and survive cancelled Activity attempts. Close must
    tolerate overlapping calls and unblock outstanding reads and writes.
    A write that raises a retryable exception must not have committed its message;
    use CodexError for ambiguous or permanent failures.
    """

    async def write(self, message: Message) -> None:
        """Send one JSON-RPC message."""
        ...

    async def read(self) -> list[Message]:
        """Read the next batch, or return an empty list on EOF."""
        ...

    async def close(self) -> None:
        """Close the connection and release its resources."""
        ...


class Connector(Protocol):
    """Launch or connect to Codex from an Activity.

    Raise CodexError for permanent failures and other exceptions for failures
    safe to retry. The returned process stays on this Worker.
    """

    async def connect(self, request: ConnectActivityInput) -> CodexProcess:
        """Create one connection with the requested launch configuration."""
        ...
