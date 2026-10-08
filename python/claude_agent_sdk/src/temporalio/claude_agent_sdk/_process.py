"""Fail-closed process supervision and workspace locks (standard library only).

Run as ``python -I -S _process.py WORKER_PID LOCK ENGINE [ARGUMENT ...]``.
The supervisor holds the workspace lock until the engine and its descendants
have stopped. A missing launcher, failed lock or failed Windows job setup never
falls back to starting an unprotected engine.
"""

from __future__ import annotations

import errno
import os
import signal
import struct
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any


@contextmanager
def workspace_lock(path: Path) -> Iterator[None]:
    """Hold an OS lock without unlinking its file or tolerating lock failures.

    Args:
        path: A lock file outside the disposable workspace.

    Yields:
        Once exclusive ownership is established.
    """
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.set_inheritable(fd, False)
        if sys.platform == "win32":
            import msvcrt

            if os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        # Closing also releases the lock, including on setup failure. Never
        # unlink: another executor may already have this inode open.
        os.close(fd)


def _windows_guard(worker: int) -> tuple[Any, Any, Any]:
    """Create a kill-on-close job before starting children and pin the Worker handle."""
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [(str(i), ctypes.c_uint64) for i in range(6)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("Basic", BasicLimits),
            ("Io", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    parent = kernel.OpenProcess(0x100000, False, worker)  # SYNCHRONIZE, pins identity
    info = ExtendedLimits()
    info.Basic.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE; no child breakaway
    if (
        not job
        or not parent
        or not kernel.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info)
        )
    ):
        if job:
            kernel.CloseHandle(job)
        if parent:
            kernel.CloseHandle(parent)
        raise OSError("cannot establish Windows engine supervision")
    return kernel, job, parent


def _start_windows_engine(
    argv: list[str], guard: tuple[Any, Any, Any]
) -> subprocess.Popen[bytes]:
    """Assign a suspended engine to its job before it can create any children."""
    import ctypes
    from ctypes import wintypes

    child: Any = subprocess.Popen(argv, creationflags=0x4)  # CREATE_SUSPENDED
    try:
        if not guard[0].AssignProcessToJobObject(guard[1], int(child._handle)):
            raise OSError("cannot assign Claude engine to its job")
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)  # type: ignore[attr-defined]
        ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        ntdll.NtResumeProcess.restype = wintypes.LONG
        if ntdll.NtResumeProcess(int(child._handle)) != 0:
            raise OSError("cannot resume supervised Claude engine")
        return child
    except BaseException:
        child.kill()
        child.wait()
        raise


def _process_table() -> dict[int, tuple[int, str]]:
    """Read only process ancestry and state, including detached Bash/MCP children."""
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,stat="],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return {
        int(pid): (int(parent), state)
        for pid, parent, state in (line.split() for line in result.stdout.splitlines())
    }


class _PosixTree:
    """Retain process lifetime ownership independently of current engine ancestry."""

    def __init__(self) -> None:
        import ctypes

        self._ctypes = ctypes
        self._owned: set[int] = set()
        self._previous_subreaper = ctypes.c_int()
        if sys.platform == "darwin":
            self._lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_last_error=True)
            self._lib.proc_pidinfo.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            self._lib.proc_pidinfo.restype = ctypes.c_int
            if self._identity(os.getpid()) is None:
                raise OSError("cannot establish macOS descendant ownership")
        elif sys.platform.startswith("linux"):
            self._lib = ctypes.CDLL(None, use_last_error=True)
            self._lib.prctl.restype = ctypes.c_int
            self._prctl(
                37, ctypes.byref(self._previous_subreaper)
            )  # GET_CHILD_SUBREAPER
            self._prctl(36, ctypes.c_ulong(1))  # SET_CHILD_SUBREAPER
        else:
            raise OSError(
                "process lifetime supervision is unavailable on this platform"
            )

    def _prctl(self, option: int, value: Any) -> None:
        if self._lib.prctl(option, value, 0, 0, 0) != 0:
            code = self._ctypes.get_errno()
            raise OSError(code, "cannot establish Linux descendant ownership")

    def _identity(self, pid: int) -> tuple[int, int] | None:
        if sys.platform == "darwin":
            # PROC_PIDUNIQIDENTIFIERINFO: the parent's unique ID is fixed at
            # fork/spawn and survives reparenting; unique IDs survive exec and
            # distinguish reused PIDs. The interface's record is 56 bytes.
            # https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/proc_internal.h
            data = self._ctypes.create_string_buffer(56)
            self._ctypes.set_errno(0)
            size = self._lib.proc_pidinfo(pid, 17, 0, data, len(data))
            if size != len(data):
                code = self._ctypes.get_errno()
                if size == 0 and code in (0, errno.ESRCH):
                    return None
                raise OSError(
                    code,
                    f"cannot inspect process lifetime identity for {pid} (size {size})",
                )
            return struct.unpack_from("=QQ", data.raw, 16)
        try:
            fields = Path(f"/proc/{pid}/stat").read_bytes().rsplit(b")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            return None
        return int(fields[19]), int(fields[1])

    def _start(self, argv: list[str]) -> subprocess.Popen[bytes]:
        # The exec gate ensures we retain the engine's identity before it can
        # fork and exit, including when it runs a very short shell command.
        read_fd, write_fd = os.pipe()
        child: subprocess.Popen[bytes] | None = None
        try:
            child = subprocess.Popen(
                [sys.executable, "-I", "-S", __file__, "--exec", str(read_fd), *argv],
                start_new_session=True,
                pass_fds=(read_fd,),
            )
            identity = self._identity(child.pid)
            if identity is None:
                raise OSError("engine exited before supervision was established")
            self._owned.add(identity[0])
            os.write(write_fd, b"1")
            return child
        except BaseException:
            if child is not None:
                child.kill()
                child.wait()
            raise
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def _discover(self) -> dict[int, tuple[int, str]]:
        table = _process_table()
        identities: dict[int, tuple[int, int]] = {}
        for pid in table:
            try:
                identity = self._identity(pid)
            except PermissionError:
                # Other users' processes cannot belong to our tool tree.
                continue
            if identity is not None:
                identities[pid] = identity
        if sys.platform == "darwin":
            while True:
                new = {
                    unique
                    for unique, parent in identities.values()
                    if parent in self._owned and unique not in self._owned
                }
                if not new:
                    break
                self._owned.update(new)
            return {
                pid: (unique, table[pid][1])
                for pid, (unique, _) in identities.items()
                if unique in self._owned
            }
        # Linux reparents orphaned descendants to this subreaper, including
        # double-forked daemons. They remain discoverable after engine exit.
        found = {os.getpid()}
        while True:
            new_pids = {
                pid for pid, (parent, _) in table.items() if parent in found
            } - found
            if not new_pids:
                break
            found.update(new_pids)
        return {
            pid: (identities[pid][0], table[pid][1])
            for pid in found - {os.getpid()}
            if pid in identities
        }

    def _signal(self, pid: int, unique: int, signum: int) -> None:
        identity = self._identity(pid)
        if identity is not None and identity[0] == unique:
            with suppress(ProcessLookupError):
                os.kill(pid, signum)

    def _close(self) -> None:
        if sys.platform.startswith("linux"):
            self._prctl(36, self._ctypes.c_ulong(self._previous_subreaper.value))


def _stop_posix_tree(child: subprocess.Popen[bytes], tree: _PosixTree) -> None:
    """Stop retained descendants even after the engine exits and they reparent."""
    descendants: dict[int, int] = {}
    try:
        while True:
            found = tree._discover()
            new = {
                pid: unique
                for pid, (unique, _) in found.items()
                if descendants.get(pid) != unique
            }
            if not new:
                break
            for pid, unique in new.items():
                tree._signal(pid, unique, signal.SIGSTOP)
            descendants.update(new)
    finally:
        for pid, unique in descendants.items():
            tree._signal(pid, unique, signal.SIGKILL)
        # Ownership is tracked per process, including detached groups. A
        # redundant killpg after individual termination can fail with EPERM
        # on macOS when that group contains only zombies.
        child.kill()
        child.wait()
    # A terminated orphan can temporarily remain a zombie. It cannot execute;
    # every still-running descendant must stop before releasing the lock.
    deadline = time.monotonic() + 5
    while True:
        running = {
            pid: unique
            for pid, (unique, state) in tree._discover().items()
            if not state.startswith("Z")
        }
        if not running:
            if sys.platform.startswith("linux"):
                # Reap children adopted from the engine as well as the engine.
                while True:
                    try:
                        pid, _ = os.waitpid(-1, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if pid == 0:
                        break
            return
        for pid, unique in running.items():
            tree._signal(pid, unique, signal.SIGKILL)
        if time.monotonic() >= deadline:
            raise RuntimeError("Claude engine descendants did not stop")
        time.sleep(0.01)


def supervise(worker: int, lock: Path, argv: list[str]) -> int:
    """Run one engine, kill its process tree on Worker loss, then release its lock.

    Args:
        worker: The Worker's process id.
        lock: A shared lock for local executors using this workspace.
        argv: The engine command.

    Returns:
        The engine's exit code.
    """
    stopping = False

    def stop(signum: int, frame: object) -> None:
        nonlocal stopping
        del signum, frame
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, stop)
    with workspace_lock(lock):
        if os.name != "nt":
            # Failure to inspect detached children is a startup failure, not a
            # reason to silently weaken cleanup guarantees.
            _process_table()
        windows = _windows_guard(worker) if os.name == "nt" else None

        def worker_gone() -> bool:
            if windows is not None:
                status = windows[0].WaitForSingleObject(windows[2], 0)
                if status not in (0, 258):  # WAIT_OBJECT_0, WAIT_TIMEOUT
                    raise OSError("cannot check Worker lifetime")
                return status == 0
            return os.getppid() != worker

        if stopping or worker_gone():
            if windows is not None:
                windows[0].CloseHandle(windows[1])
                windows[0].CloseHandle(windows[2])
            return 1
        child: subprocess.Popen[bytes] | None = None
        posix = _PosixTree() if windows is None else None
        try:
            if windows is not None:
                child = _start_windows_engine(argv, windows)
            else:
                assert posix is not None
                child = posix._start(argv)
            while child.poll() is None:
                if posix is not None:
                    # Keep identities when parents disappear between scans;
                    # cleanup must not reconstruct ownership from live PPIDs.
                    posix._discover()
                if stopping or worker_gone():
                    break
                try:
                    child.wait(timeout=0.05)
                except subprocess.TimeoutExpired:
                    pass
            return child.returncode if child.returncode is not None else 1
        finally:
            if windows is not None:
                try:
                    windows[0].TerminateJobObject(windows[1], 1)
                    if child is not None:
                        child.wait()
                finally:
                    windows[0].CloseHandle(windows[1])
                    windows[0].CloseHandle(windows[2])
            elif posix is not None:
                try:
                    if child is not None:
                        _stop_posix_tree(child, posix)
                finally:
                    posix._close()


def main() -> None:
    """Start a supervised engine; setup failures exit before it can execute."""
    try:
        if len(sys.argv) >= 4 and sys.argv[1] == "--exec":
            fd = int(sys.argv[2])
            allowed = os.read(fd, 1)
            os.close(fd)
            if allowed != b"1":
                raise OSError("supervisor exited before admitting the engine")
            os.execvp(sys.argv[3], sys.argv[3:])
        if len(sys.argv) < 4:
            raise ValueError("usage: _process.py WORKER_PID LOCK ENGINE [ARGUMENT ...]")
        code = supervise(int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
    except Exception as exc:
        sys.stderr.write(f"Cannot safely start Claude Code: {exc}\n")
        code = 1
    sys.exit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
