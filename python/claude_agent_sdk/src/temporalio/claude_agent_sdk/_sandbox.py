"""The interface a sandbox provider implements to run Claude Code for a Workflow.

The CLI runs detached in the sandbox, with its stdin and stdout in files. Activities
write its stdin with :meth:`Sandbox.write_files` and follow its stdout with
:meth:`Sandbox.stream`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class SandboxError(Exception):
    """The sandbox is gone, or could not run a command. The message says why.

    It fails the Activity, and the retry builds the sandbox again with its factory.
    """


@dataclass
class ExecResult:
    """How a command in the sandbox ended."""

    exit_code: int
    stdout: bytes
    stderr: bytes


class Sandbox(ABC):
    """Where the CLI runs. Built by a :data:`SandboxFactory`.

    Every Activity attempt builds one with the factory, which decides how it
    reaches the session's sandbox, such as after the sandbox was lost. Claude
    Code keeps its session files in :attr:`home`, and a CLI started with
    ``resume`` resumes from them.
    """

    @property
    @abstractmethod
    def cli_path(self) -> str:
        """The Claude Code executable in the sandbox."""

    @property
    @abstractmethod
    def home(self) -> str:
        """A directory outside the workspace that holds the CLI's config, plugins, stdin, and stdout."""

    @abstractmethod
    async def start(self, *, env: Mapping[str, str], **kwargs: Any) -> None:
        """Start the sandbox for a session's CLI.

        ``env`` is the CLI's environment. The commands the Activities run use
        ``sh``, ``cat``, ``mkdir``, ``mv``, ``rm``, ``sleep``, ``tail``, and ``wc``.
        """

    @abstractmethod
    async def exec(self, argv: Sequence[str], **kwargs: Any) -> ExecResult:
        """Run ``argv`` in the workspace, without a shell, and return how it ended.

        Raises :class:`SandboxError` if the sandbox is gone.
        """

    @abstractmethod
    def stream(self, argv: Sequence[str], **kwargs: Any) -> AsyncIterator[bytes]:
        """Run ``argv`` in the workspace, without a shell, and yield its stdout as it comes.

        Ends when the command exits. Closing it early stops the command. Raises
        :class:`SandboxError` if the sandbox is gone.
        """

    @abstractmethod
    async def write_files(
        self, files: Mapping[str, tuple[bytes, int]], **kwargs: Any
    ) -> None:
        """Write each file in ``files``, a path in the sandbox, with its content and mode.

        Each file must appear whole, never partly written, for example by
        writing it under another name and renaming it. Creates the directories
        they are in, as needed. Raises :class:`SandboxError` if the sandbox is
        gone.
        """

    @abstractmethod
    async def terminate(self, **kwargs: Any) -> None:
        """End the sandbox and everything running in it."""


@dataclass(frozen=True)
class SandboxContext:
    """The Workflow an Activity runs a sandbox for, passed to a :data:`SandboxFactory`."""

    sandbox: str
    """The name the factory is registered under on ``ClaudeAgentSDKPlugin``."""
    namespace: str
    workflow_id: str
    run_id: str
    first_execution_run_id: str
    """The Workflow's first run ID. The same through a reset and continue-as-new."""


SandboxFactory = Callable[[SandboxContext], Sandbox | Awaitable[Sandbox]]
"""Builds the :class:`Sandbox` for each Activity attempt. Registered on :class:`ClaudeAgentSDKPlugin` by name."""
