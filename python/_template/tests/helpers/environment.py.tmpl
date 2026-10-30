"""Bounded retries for local Temporal server startup."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import sys
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from temporalio.testing import WorkflowEnvironment

logger = logging.getLogger(__name__)

_RETRYABLE_STARTUP_ERROR = re.compile(
    r"^Failed starting Temporal dev server: (?:"
    r"ephemeral server at \S+ did not start within \d+s\."
    r"|failed to download ephemeral server executable:"
    r"|invalid ephemeral server download: could not (?:read|extract) "
    r"(?:tar|ZIP) archive(?: entry)?(?:$|:)"
    r")"
)


async def start_local_with_retry(
    start: Callable[[], Awaitable[WorkflowEnvironment]],
) -> WorkflowEnvironment:
    """Invoke a configured startup callback with up to three attempts.

    The caller binds all server options, keeping this policy independent of
    plugin configuration. Only the SDK's startup timeout, download, and archive
    read/extraction errors are retried after one second. Other errors and
    cancellation propagate immediately.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await start()
        except RuntimeError as exc:
            if attempt >= 3 or not _RETRYABLE_STARTUP_ERROR.match(str(exc)):
                raise
            logger.warning(
                "Temporal dev server startup attempt %s failed: %s", attempt, exc
            )
            await asyncio.sleep(1)


@contextmanager
def _capture_server_output(directory: Path) -> Generator[None, None, None]:
    """Give the native child persistent log files, then restore pytest's streams.

    The server inherits these descriptors when the SDK spawns it. Restoring the
    parent's descriptors after startup does not redirect the child's output.
    Windows native subprocesses also need their Win32 standard handles updated.
    """
    saved: dict[int, int] = {}
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, name in ((1, "stdout.log"), (2, "stderr.log")):
            saved[fd] = os.dup(fd)
            with (directory / name).open("ab", buffering=0) as stream:
                os.dup2(stream.fileno(), fd)
            _set_native_std_handle(fd)
        yield
    finally:
        for fd, original in saved.items():
            os.dup2(original, fd)
            os.close(original)
            _set_native_std_handle(fd)


def _set_native_std_handle(fd: int) -> None:
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        set_handle = kernel.SetStdHandle
        set_handle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
        set_handle.restype = wintypes.BOOL
        if not set_handle(-10 - fd, msvcrt.get_osfhandle(fd)):
            raise ctypes.WinError(ctypes.get_last_error())


class _ServerDiagnostics:
    """Observe SDK-owned children without taking over their lifecycle or reaping."""

    def __init__(self, directory: Path, version: str) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._existing_pids = {process.pid for process in psutil.Process().children()}
        self._processes: dict[int, psutil.Process] = {}
        self._exited: set[int] = set()
        self._waiters: list[threading.Thread] = []
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self.record(
            "session",
            worker=os.environ.get("PYTEST_XDIST_WORKER", "main"),
            python_pid=os.getpid(),
            python_version=sys.version,
            platform=platform.platform(),
            server_version=version,
        )

    def record(self, event: str, **fields: Any) -> None:
        with self._lock:
            with (self.directory / "lifecycle.jsonl").open(
                "a", encoding="utf-8"
            ) as log:
                log.write(
                    json.dumps(
                        {
                            "time": datetime.now(timezone.utc).isoformat(),
                            "event": event,
                            **fields,
                        }
                    )
                    + "\n"
                )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        for waiter in self._waiters:
            waiter.join(timeout=1)

    def _wait_windows(self, process: psutil.Process) -> None:
        # Windows waiting holds a process handle and can retain its exit code.
        # POSIX wait()/waitpid() would compete with the SDK's child reaper.
        try:
            self.record("process_exited", pid=process.pid, exit_code=process.wait())
        except psutil.Error as exc:
            self.record("observation_error", pid=process.pid, error=repr(exc))

    def _snapshot(self) -> None:
        for process in psutil.Process().children():
            if process.pid in self._processes or process.pid in self._existing_pids:
                continue
            try:
                command = process.cmdline()
                # The SDK invokes the CLI with these as its first arguments.
                # A Python test child can contain the same strings later in
                # its command without being a Temporal server.
                if command[1:3] != ["server", "start-dev"]:
                    continue
                self._processes[process.pid] = process
                self.record("process_started", pid=process.pid, command=command)
                if sys.platform == "win32":
                    waiter = threading.Thread(
                        target=self._wait_windows, args=(process,), daemon=True
                    )
                    self._waiters.append(waiter)
                    waiter.start()
            except psutil.NoSuchProcess:
                continue
        if sys.platform != "win32":
            for pid, process in self._processes.items():
                if pid not in self._exited and (
                    not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
                ):
                    self._exited.add(pid)
                    self.record(
                        "process_exited",
                        pid=pid,
                        exit_code=None,
                        detail="Exit code belongs to the SDK's POSIX child reaper.",
                    )

    def _monitor(self) -> None:
        while True:
            try:
                self._snapshot()
            except (psutil.Error, OSError) as exc:
                self.record("observation_error", error=repr(exc))
            if self._stop.wait(0.1):
                # Include a process that exited during fixture teardown.
                try:
                    self._snapshot()
                except (psutil.Error, OSError) as exc:
                    self.record("observation_error", error=repr(exc))
                return


@asynccontextmanager
async def diagnosed_local_server(
    start: Callable[[], Awaitable[WorkflowEnvironment]],
    *,
    server_version: str,
    log_root: Path = Path("junit/servers"),
) -> AsyncGenerator[WorkflowEnvironment, None]:
    """Record each worker's server output and lifecycle without adding retries."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    diagnostics = _ServerDiagnostics(
        log_root / f"{worker}-{os.getpid()}", server_version
    )
    diagnostics.start()
    attempt = 0

    async def start_with_output() -> WorkflowEnvironment:
        nonlocal attempt
        attempt += 1
        diagnostics.record("startup_requested", attempt=attempt)
        try:
            with _capture_server_output(diagnostics.directory):
                return await start()
        except BaseException as exc:
            diagnostics.record("startup_failed", attempt=attempt, error=repr(exc))
            raise

    try:
        environment = await start_local_with_retry(start_with_output)
        diagnostics.record(
            "ready", address=environment.client.service_client.config.target_host
        )
        try:
            yield environment
        finally:
            diagnostics.record("shutdown_requested")
            try:
                await environment.shutdown()
            except BaseException as exc:
                diagnostics.record("shutdown_failed", error=repr(exc))
                raise
            diagnostics.record("shutdown_completed")
    finally:
        diagnostics.close()
