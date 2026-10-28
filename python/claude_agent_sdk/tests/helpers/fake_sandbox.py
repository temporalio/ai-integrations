"""An in-process ``SandboxRuntime``: each command is a local subprocess.

Records what the runtime was asked to do. The provider-neutral suite runs against it.
A provider test can reuse that suite with the same recording attributes.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from temporalio.claude_agent_sdk.sandbox import LAUNCH_SCRIPT, generation_is_stray

CLI_VERSION = "2.1.294"


class FakeProcess:
    """A ``SandboxProcess`` over a local subprocess."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        fail_writes: bool = False,
        *,
        fail_wait: bool = False,
        fail_stdout: bool = False,
    ) -> None:
        """Bind ``process``. ``fail_writes`` raises from ``write``."""
        assert process.stdin and process.stdout and process.stderr
        self.process = process
        self.fail_writes = fail_writes
        self.fail_wait = fail_wait
        self.fail_stdout = fail_stdout

    async def write(self, data: bytes) -> None:
        """Write ``data`` to stdin, or fail when the runtime asked writes to drop."""
        assert self.process.stdin
        if self.fail_writes:
            raise RuntimeError("Failed to write to exec stdin")
        self.process.stdin.write(data)
        await self.process.stdin.drain()

    async def end_input(self) -> None:
        """Close stdin."""
        assert self.process.stdin
        self.process.stdin.write_eof()
        await self.process.stdin.drain()

    async def _chunks(
        self, stream: asyncio.StreamReader | None
    ) -> AsyncIterator[bytes]:
        assert stream
        while chunk := await stream.read(65536):
            yield chunk

    def stdout(self) -> AsyncIterator[bytes]:
        """Stdout chunks, or a stream that fails on the first read."""
        if self.fail_stdout:
            return _BrokenStdout()
        return self._chunks(self.process.stdout)

    def stderr(self) -> AsyncIterator[bytes]:
        """Stderr chunks."""
        return self._chunks(self.process.stderr)

    async def wait(self) -> int:
        """The process exit code, or a failure to observe it."""
        if self.fail_wait:
            raise RuntimeError("wait failed")
        return await self.process.wait()


class _BrokenStdout:
    """An stdout iterator whose first read fails."""

    def __aiter__(self) -> _BrokenStdout:
        return self

    async def __anext__(self) -> bytes:
        raise RuntimeError("stdout failed")


class _FixedExit:
    """A process that has already finished with ``code``. The probe uses ``wait`` only."""

    def __init__(self, code: int) -> None:
        self._code = code

    async def write(self, data: bytes) -> None:
        del data

    async def end_input(self) -> None:
        return

    def stdout(self) -> AsyncIterator[bytes]:
        return self._empty()

    def stderr(self) -> AsyncIterator[bytes]:
        return self._empty()

    async def _empty(self) -> AsyncIterator[bytes]:
        items: list[bytes] = []
        for item in items:
            yield item

    async def wait(self) -> int:
        return self._code


class _Version:
    """What ``cli --version`` prints. Not a subprocess."""

    def __init__(self, text: str) -> None:
        self._text = text.encode()

    async def write(self, data: bytes) -> None:
        """Ignore stdin. The version probe does not write."""
        del data

    async def end_input(self) -> None:
        """Ignore stdin EOF."""

    async def _once(self) -> AsyncIterator[bytes]:
        yield self._text

    def stdout(self) -> AsyncIterator[bytes]:
        """The version text, once."""
        return self._once()

    def stderr(self) -> AsyncIterator[bytes]:
        """No stderr."""
        return self._empty()

    async def _empty(self) -> AsyncIterator[bytes]:
        items: list[bytes] = []
        for item in items:
            yield item

    async def wait(self) -> int:
        """The probe succeeded."""
        return 0


class FakeSandbox:
    """One started sandbox. ``execs`` records argv and the env the runtime passed."""

    def __init__(
        self,
        runtime: FakeSandboxRuntime,
        session_id: str | None,
        snapshot: str | None,
        generation: int | None = None,
    ) -> None:
        """Remember which session, snapshot, and lease generation this sandbox started from."""
        self.runtime = runtime
        self.session_id = session_id
        self.started_from = snapshot
        self.generation = generation
        self.generation_tag = None if generation is None else str(generation)
        self.execs: list[tuple[tuple[str, ...], Mapping[str, str | None]]] = []
        self.terminated = 0
        self.processes: list[asyncio.subprocess.Process] = []

    async def exec(
        self, argv: Sequence[str], *, env: Mapping[str, str | None] | None = None
    ) -> Any:
        """Run ``argv`` locally, or raise ``runtime.exec_error`` when set."""
        self.execs.append((tuple(argv), env or {}))
        if self.runtime.exec_error is not None:
            raise self.runtime.exec_error
        if tuple(argv[1:]) == ("--version",):
            return _Version(self.runtime.cli_version)
        if (
            self.runtime.probe_exit is not None
            and len(argv) >= 3
            and "test -d" in argv[2]
        ):
            return _FixedExit(self.runtime.probe_exit)
        cli = tuple(argv[:3]) == ("sh", "-c", LAUNCH_SCRIPT)
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                "PATH": os.environ["PATH"],
                **self.runtime.image_env,
                **{k: v for k, v in (env or {}).items() if v},
            },
        )
        self.processes.append(process)
        return FakeProcess(
            process,
            self.runtime.fail_writes and cli,
            fail_wait=self.runtime.fail_wait and cli,
            fail_stdout=self.runtime.fail_stdout and cli,
        )

    async def snapshot(self) -> str:
        """The next ``snap-N`` id, or ``runtime.snapshot_error``."""
        if self.runtime.snapshot_error is not None:
            raise self.runtime.snapshot_error
        snapshot = f"snap-{len(self.runtime.snapshots) + 1}"
        self.runtime.snapshots.append((self, snapshot))
        return snapshot

    async def terminate(self) -> None:
        """Kill commands still running. Does not drop recorded snapshots."""
        self.terminated += 1
        for process in self.processes:
            if process.returncode is None:
                process.kill()


class FakeSandboxRuntime:
    """A ``SandboxRuntime``. ``sandboxes`` and ``snapshots`` record the calls."""

    def __init__(self, cli_path: str, workspace_root: str) -> None:
        """Bind the CLI inside the sandbox and the session directory parent."""
        self.cli_path = cli_path
        self.workspace_root = workspace_root
        self.cli_version = f"{CLI_VERSION} (Claude Code)\n"
        self.image_env: dict[str, str] = {}
        """Variables every command in the sandbox sees (a provider's secrets and image ENV)."""
        self._supports_native_tools = True
        self.fail_writes = False
        self.fail_wait = False
        self.fail_stdout = False
        self.probe_exit: int | None = None
        self.exec_error: Exception | None = None
        self.sandboxes: list[FakeSandbox] = []
        self.snapshots: list[tuple[FakeSandbox, str]] = []
        self.snapshot_error: Exception | None = None
        self.sweep_started: asyncio.Event | None = None
        self.sweep_release: asyncio.Event | None = None
        self.create_started: asyncio.Event | None = None
        self.create_release: asyncio.Event | None = None

    @property
    def supports_native_tools(self) -> bool:
        """Whether native tools are allowed. Tests set this to fail closed."""
        return self._supports_native_tools

    @supports_native_tools.setter
    def supports_native_tools(self, value: bool) -> None:
        self._supports_native_tools = value

    def cwd(self, session_id: str) -> str:
        """The session directory. The same path on every sandbox."""
        return f"{self.workspace_root}/{session_id}"

    async def start(
        self,
        *,
        session_id: str | None,
        snapshot: str | None,
        generation: int | None = None,
    ) -> FakeSandbox:
        """Record a new sandbox started from ``snapshot`` at ``generation``."""
        if self.create_started is not None:
            self.create_started.set()
        if self.create_release is not None:
            await self.create_release.wait()
        sandbox = FakeSandbox(self, session_id, snapshot, generation)
        self.sandboxes.append(sandbox)
        return sandbox

    async def terminate_strays(self, session_id: str, *, generation: int) -> None:
        """Terminate this session's sandboxes older than ``generation``.

        Snapshots stay. A snapshot named by workflow state is not deleted. Equal and newer
        generations are left running, including one created while this sweep was blocked.
        """
        if self.sweep_started is not None:
            self.sweep_started.set()
        if self.sweep_release is not None:
            await self.sweep_release.wait()
        for sandbox in self.sandboxes:
            if (
                sandbox.session_id == session_id
                and not sandbox.terminated
                and generation_is_stray(sandbox.generation_tag, generation)
            ):
                await sandbox.terminate()
