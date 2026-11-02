"""Async stdio connector using the SDK's own bundled-runtime resolver."""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from pathlib import Path
from typing import Any

from openai_codex import CodexError, TransportClosedError
from openai_codex.client import (
    _installed_codex_path_dirs,
    _prepend_path_dirs,
    _resolve_codex_bin,
)

from ._types import ConnectActivityInput, Message


class SubprocessConnector:
    """Run the configured Codex runtime in a subprocess of the Worker."""

    async def connect(self, request: ConnectActivityInput) -> CodexProcessConnection:
        """Match upstream launch arguments, environment, PATH, and runtime selection."""
        config = request.config
        path_dirs: tuple[Path, ...] = ()
        if config.launch_args_override is not None:
            args = list(config.launch_args_override)
        else:
            args = [str(_resolve_codex_bin(config))]
            if config.codex_bin is None:
                path_dirs = _installed_codex_path_dirs()
            for value in config.config_overrides:
                args.extend(["--config", value])
            args.extend(["app-server", "--listen", "stdio://"])
        env = {**os.environ, **(config.env or {})}
        _prepend_path_dirs(env, path_dirs)
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=config.cwd,
                env=env,
                limit=16 * 1024 * 1024,
            )
        except (FileNotFoundError, PermissionError):
            raise
        except (OSError, ValueError) as error:
            raise CodexError(str(error)) from error
        return CodexProcessConnection(proc)


class CodexProcessConnection:
    """A draining subprocess connection with bounded stderr diagnostics."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        """Start the output pumps immediately."""
        self._process = process
        self._output: asyncio.Queue[Message | BaseException] = asyncio.Queue()
        self._stderr: deque[str] = deque(maxlen=400)
        self._write_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        self._stdout_task = asyncio.create_task(self._drain_stdout())

    async def _drain_stderr(self) -> None:
        assert self._process.stderr is not None
        while line := await self._process.stderr.readline():
            self._stderr.append(line.decode("utf-8", errors="replace").rstrip())

    async def _drain_stdout(self) -> None:
        assert self._process.stdout is not None
        try:
            while line := await self._process.stdout.readline():
                try:
                    message: Any = json.loads(line)
                except (ValueError, UnicodeError) as error:
                    raise CodexError(f"Invalid JSON-RPC line: {line!r}") from error
                if not isinstance(message, dict):
                    raise CodexError(f"Invalid JSON-RPC payload: {message!r}")
                self._output.put_nowait(message)
            raise TransportClosedError(
                "Codex process closed stdout. stderr_tail="
                + "\n".join(self._stderr)[-2000:]
            )
        except CodexError as error:
            self._output.put_nowait(error)
        except Exception as error:
            terminal = TransportClosedError(f"Failed to read Codex output: {error}")
            terminal.__cause__ = error
            self._output.put_nowait(terminal)

    async def write(self, message: Message) -> None:
        """Serialize writes; an ambiguous broken pipe permanently ends the session."""
        async with self._write_lock:
            if self._closed or self._process.stdin is None:
                raise TransportClosedError("Codex process is not running")
            try:
                self._process.stdin.write((json.dumps(message) + "\n").encode())
                await self._process.stdin.drain()
            except (OSError, RuntimeError) as error:
                raise TransportClosedError(str(error)) from error

    async def read(self) -> list[Message]:
        """Return queued output before reporting an exit or malformed message."""
        output: list[Message] = []
        item = await self._output.get()
        while not isinstance(item, BaseException):
            output.append(item)
            if self._output.empty():
                return output
            item = self._output.get_nowait()
        self._output.put_nowait(item)
        if not output:
            raise item
        return output

    async def close(self) -> None:
        """Terminate and reap the runtime, including when callers overlap."""
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if self._process.stdin is not None:
                self._process.stdin.close()
            if self._process.returncode is None:
                try:
                    self._process.terminate()
                    await asyncio.wait_for(self._process.wait(), 2)
                except ProcessLookupError:
                    pass
                except asyncio.TimeoutError:
                    self._process.kill()
            await self._process.wait()
            self._output.put_nowait(
                TransportClosedError("Codex process is not running")
            )
            self._stdout_task.cancel()
            self._stderr_task.cancel()
            await asyncio.gather(
                self._stdout_task, self._stderr_task, return_exceptions=True
            )
