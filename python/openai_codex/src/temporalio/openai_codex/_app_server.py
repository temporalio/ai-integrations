"""A minimal asyncio JSON-RPC client for ``codex app-server --listen stdio://``.

Deliberately not the ``openai-codex`` SDK client: that client runs server-request handlers on its
reader thread, so one pending tool call would stall every notification. Here each server request is
served in its own task, and a call that is never answered (a pending host tool) blocks nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any


class AppServerExited(RuntimeError):
    """The ``codex app-server`` process ended while a request or turn was in flight."""


def default_codex_bin() -> str:
    """Locate the Codex binary: ``$CODEX_BIN``, else the one bundled with ``openai-codex-cli-bin``."""
    if os.environ.get("CODEX_BIN"):
        return os.environ["CODEX_BIN"]
    try:
        import codex_cli_bin  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "No Codex binary found. Install the `bundled-codex` extra "
            "(`pip install 'temporalio-openai-codex[bundled-codex]'`), set CODEX_BIN, "
            "or pass CodexPlugin(codex_bin=...)."
        ) from exc
    return str(codex_cli_bin.bundled_codex_path())  # pyright: ignore[reportUnknownMemberType]


class AppServer:
    """One ``codex app-server`` child process and its JSON-RPC connection."""

    def __init__(
        self,
        *,
        codex_bin: str,
        home: str,
        config_overrides: Sequence[str],
        env: Mapping[str, str],
        cwd: str,
        on_request: Callable[[str, Any], Awaitable[Any]],
    ) -> None:
        """Prepare (but do not start) the process; ``on_request`` serves server->client requests."""
        self._argv = [codex_bin]
        for kv in config_overrides:
            self._argv += ["--config", kv]
        self._argv += ["app-server", "--listen", "stdio://"]
        self._env = {**os.environ, **env, "CODEX_HOME": home}
        self._cwd = cwd
        self._on_request = on_request
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._readers: list[asyncio.Task[None]] = []
        self.notifications: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.stderr: list[str] = []
        self.proc: asyncio.subprocess.Process

    async def start(self) -> None:
        """Spawn the process and complete the JSON-RPC handshake."""
        self.proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
            cwd=self._cwd,
            limit=1 << 24,
        )
        self._readers = [
            asyncio.create_task(self._read_stdout()),
            asyncio.create_task(self._read_stderr()),
        ]
        await self.call(
            "initialize",
            {
                "clientInfo": {
                    "name": "temporalio-openai-codex",
                    "title": "Temporal OpenAI Codex integration",
                    "version": "0",
                },
                # dynamicTools and thread/inject_items are experimental protocol surface.
                "capabilities": {"experimentalApi": True},
            },
        )
        self._send({"method": "initialized"})

    async def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        async for line in self.proc.stderr:
            self.stderr.append(line.decode(errors="replace").rstrip())
            del self.stderr[:-50]

    def _send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(message) + "\n").encode())

    async def call(self, method: str, params: Any = None) -> Any:
        """Send a request and await its result; a JSON-RPC error raises ``RuntimeError``."""
        request_id = str(uuid.uuid4())
        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = future
        message: dict[str, Any] = {"id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)
        reply = await future
        if "error" in reply:
            raise RuntimeError(f"codex {method}: {reply['error']}")
        return reply["result"]

    async def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        async for line in self.proc.stdout:
            message = json.loads(line)
            if "method" in message and "id" in message:
                task = asyncio.create_task(self._serve(message))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            elif "method" in message:
                self.notifications.put_nowait(message)
            else:
                future = self._pending.pop(message["id"], None)
                if future is not None and not future.done():
                    future.set_result(message)
        self.notifications.put_nowait({"method": "__eof__"})
        for future in self._pending.values():
            if not future.done():
                future.set_exception(AppServerExited("codex app-server exited"))

    async def _serve(self, message: dict[str, Any]) -> None:
        try:
            result = await self._on_request(message["method"], message.get("params"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported to Codex as a JSON-RPC error
            self._send(
                {"id": message["id"], "error": {"code": -32000, "message": str(exc)}}
            )
            return
        self._send({"id": message["id"], "result": result})

    def kill(self) -> None:
        """SIGKILL the process."""
        with contextlib.suppress(ProcessLookupError):
            self.proc.kill()

    async def close(self) -> None:
        """Stop the process (politely, then forcibly) and the reader tasks."""
        for task in list(self._tasks):
            task.cancel()
        if self.proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                assert self.proc.stdin is not None
                self.proc.stdin.close()
                self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 3)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    self.proc.kill()
        for reader in self._readers:
            reader.cancel()
