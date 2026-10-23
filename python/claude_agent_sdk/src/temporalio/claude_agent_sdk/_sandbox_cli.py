"""Claude Code in a sandbox, run and followed through files with the scripts in ``scripts/``."""

from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from . import _shell
from ._home import SandboxHome
from ._payloads import SandboxSession
from ._sandbox import ExecResult, Sandbox


class SandboxCli:
    """The CLI in a sandbox: there is one per sandbox.

    Its stdin and stdout are files, so the CLI outlives the Activity that
    launched it, and a retried Activity picks up where the last attempt was.
    """

    def __init__(self, sandbox: Sandbox, session: SandboxSession) -> None:
        """Reach the CLI in ``session``'s sandbox, launched or not."""
        self._sandbox = sandbox
        self._session = session
        self.files = SandboxHome(sandbox.home).cli

    async def launch(self, args: Sequence[str]) -> tuple[int, str]:
        """Start the CLI, if not started yet.

        Returns the pid of the shell that runs it, and the sandbox's boot ID.
        """
        files = self.files
        plugins = [
            flag
            for path in self._session.plugin_dirs
            for flag in ("--plugin-dir", path)
        ]
        argv = _shell.command(
            "launch",
            files.stdin,
            files.stdout,
            files.log,
            files.pid,
            files.new_pid,
            files.exited,
            _shell.script("run_cli"),
            self._sandbox.cli_path,
            *args,
            *plugins,
        )
        result = await self._sandbox.exec(argv)
        _check(result, "Launching Claude Code")
        pid, _, boot = result.stdout.decode().partition(" ")
        return int(pid), boot.strip()

    async def write_stdin(self, number: int, text: bytes) -> None:
        """Write ``text`` as stdin file ``number``. A retry writes the same file again."""
        path = f"{self.files.stdin}/{number}.jsonl"
        await self._sandbox.write_files({path: (text, 0o644)})

    async def exit_status(self) -> int | None:
        """The CLI's exit status, or ``None`` if it didn't exit by itself, such as when it was lost with its sandbox."""
        result = await self._sandbox.exec(["cat", self.files.exited])
        status = result.stdout.strip()
        return int(status) if result.exit_code == 0 and status else None

    def follow_stdout(self, offset: int) -> AsyncIterator[bytes]:
        """The stdout file from byte ``offset`` as it grows, until the CLI exits."""
        session = self._session
        argv = _shell.command(
            "follow_stdout",
            self.files.stdout,
            str(offset + 1),
            self.files.exited,
            str(session.cli_pid),
            session.cli_boot,
        )
        return self._sandbox.stream(argv)

    async def read_stdout(self, offset: int) -> bytes:
        """The stdout file from byte ``offset``, or nothing if there is none."""
        argv = _shell.command("read_stdout", self.files.stdout, str(offset + 1))
        result = await self._sandbox.exec(argv)
        _check(result, "Reading Claude Code's output")
        return result.stdout

    async def log(self) -> bytes:
        """The CLI's stderr."""
        return (await self._sandbox.exec(["cat", self.files.log])).stdout


async def write_plugins(sandbox: Sandbox, plugins: Sequence[str]) -> list[str]:
    """Write the ``plugins`` directories into the sandbox's home. Returns their paths there."""
    home = SandboxHome(sandbox.home)
    dirs = [f"{home.plugins}/{index}" for index in range(len(plugins))]
    if plugins:
        # Off the event loop, so a large plugin does not hold up other Activities' heartbeats.
        files = await asyncio.to_thread(_read_files, plugins, dirs)
        await sandbox.write_files(files)
    return dirs


def _read_files(
    plugins: Sequence[str], dirs: Sequence[str]
) -> dict[str, tuple[bytes, int]]:
    """Each file in ``plugins``, by its path under the matching one of ``dirs``, with its content and mode.

    Follows symbolic links, so each file is written with its target's content.
    """
    files: dict[str, tuple[bytes, int]] = {}
    for plugin, target in zip(plugins, dirs):
        if not os.path.isdir(plugin):
            raise NotADirectoryError(f"Not a plugin directory: {plugin}")
        for folder, _, names in os.walk(plugin, followlinks=True):
            for name in names:
                path = Path(folder, name)
                relative = path.relative_to(plugin).as_posix()
                mode = stat.S_IMODE(path.stat().st_mode)
                files[f"{target}/{relative}"] = (path.read_bytes(), mode)
    return files


def _check(result: ExecResult, doing: str) -> None:
    """Raise an error to retry if the command failed."""
    if result.exit_code:
        raise RuntimeError(f"{doing} failed: {result.stderr.decode(errors='replace')}")
