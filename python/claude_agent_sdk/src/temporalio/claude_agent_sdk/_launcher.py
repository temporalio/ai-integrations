"""Exec the Claude Code CLI as leader of a new session and process group.

Standard library only, run by path (``python -I -S``), not imported. The
process-group id is written to the attempt's pid file before exec so the
Activity, or a later attempt, can kill the group. Because a SIGKILLed worker
cannot reap a ``setsid`` child, a watchdog forked into the CLI's group waits
on the worker (kqueue ``NOTE_EXIT``, else a pidfd; not ``prctl``) and kills
the CLI's groups when it exits. A host with neither watch exits before exec.
"""

from __future__ import annotations

import contextlib
import errno
import os
import select
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal, NoReturn

PGID_FILE_ENV = "TEMPORAL_CLAUDE_PGID_FILE"


def main() -> None:
    """Become a session leader, record the process-group id, and exec the CLI."""
    worker = os.getppid()
    if sys.platform == "win32":
        sys.exit(1)
    os.setsid()
    _record_group()
    if not _arm_watchdog(worker):
        sys.exit(1)
    os.execv(sys.argv[1], sys.argv[1:])


def _record_group() -> None:
    """Write this pid (the CLI's process group, once it execs) to the attempt's file."""
    if path := os.environ.get(PGID_FILE_ENV):
        with open(f"{path}.tmp", "w") as handle:
            handle.write(str(os.getpid()))
        os.replace(f"{path}.tmp", path)


def _watch_kind(
    *, has_kqueue: bool, has_pidfd: bool
) -> Literal["kqueue", "pidfd"] | None:
    """``kqueue`` where the host has it, otherwise ``pidfd``, otherwise no watch."""
    if has_kqueue:
        return "kqueue"
    if has_pidfd:
        return "pidfd"
    return None


def parent_death_watch() -> Literal["kqueue", "pidfd"] | None:
    """This host's parent-exit watch, or None when a turn must not start a CLI."""
    return _watch_kind(
        has_kqueue=hasattr(select, "kqueue"),
        has_pidfd=getattr(os, "pidfd_open", None) is not None,
    )


def _host_watch() -> Literal["kqueue", "pidfd"] | None:
    """This process's parent-exit watch, or None when the CLI must not start."""
    return parent_death_watch()


def _arm_watchdog(worker: int) -> bool:
    """Fork the watchdog. False: the watch could not arm, so the CLI must not start."""
    if sys.platform == "win32" or _host_watch() is None:
        return False
    read_fd, write_fd = os.pipe()
    try:
        pid = os.fork()
    except OSError:
        os.close(read_fd)
        os.close(write_fd)
        return False
    if pid:
        os.close(write_fd)
        try:
            return os.read(read_fd, 1) == b"1"
        finally:
            os.close(read_fd)
    _watchdog(read_fd, write_fd, worker)


def _watchdog(read_fd: int, write_fd: int, worker: int) -> NoReturn:
    """Wait in the CLI's group; kill it if the worker exits. Does not return."""
    try:
        os.close(read_fd)
        _detach_stdio()
        cli = os.getppid()
        died: Callable[[], bool] | None
        try:
            died = _arm(worker, cli)
        except OSError:
            died = None
        if died is not None:
            os.write(write_fd, b"1")
            os.close(write_fd)
            write_fd = -1
            if died():
                _kill_cli_group(cli)
    except Exception:  # noqa: BLE001 - a failed watchdog must exit, not trace on the CLI's stderr
        pass
    finally:
        if write_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(write_fd)
        os._exit(0)


def _detach_stdio() -> None:
    """Drop the SDK pipes. This process must not hold the CLI's stdout open."""
    null = os.open(os.devnull, os.O_RDWR)
    try:
        os.dup2(null, 0)
        os.dup2(null, 1)
        os.dup2(null, 2)
    finally:
        os.close(null)


def _arm(worker: int, cli: int) -> Callable[[], bool]:
    """Register a parent-exit watch. The call blocks; True means the worker exited."""
    if _host_watch() == "kqueue":
        return _arm_kqueue(worker, cli)
    return _arm_pidfd(worker, cli)


def _arm_kqueue(worker: int, cli: int) -> Callable[[], bool]:
    """macOS/BSD: ``NOTE_EXIT`` for the worker and the CLI. Exec is not an exit."""
    kqueue = getattr(select, "kqueue")
    kevent = getattr(select, "kevent")
    flags = getattr(select, "KQ_EV_ADD") | getattr(select, "KQ_EV_ONESHOT")
    filt = getattr(select, "KQ_FILTER_PROC")
    note = getattr(select, "KQ_NOTE_EXIT")
    kq = kqueue()
    watching: list[int] = []
    for pid in (worker, cli):
        try:
            kq.control([kevent(pid, filt, flags, note)], 0)
        except ProcessLookupError:
            if pid == worker:
                return lambda: True
            return lambda: False
        watching.append(pid)

    def wait() -> bool:
        while True:
            try:
                got = kq.control(None, len(watching))
            except InterruptedError:
                continue
            return any(int(event.ident) == worker for event in got)

    return wait


def _arm_pidfd(
    worker: int,
    cli: int,
    *,
    open_pidfd: Callable[[int], int] | None = None,
    poll: Callable[[], list[tuple[int, int]]] | None = None,
) -> Callable[[], bool]:
    """Linux: a pidfd becomes readable when that process exits. Not ``prctl``."""
    opener = getattr(os, "pidfd_open") if open_pidfd is None else open_pidfd
    fds: dict[int, int] = {}
    for pid in (worker, cli):
        try:
            fds[opener(pid)] = pid
        except OSError as exc:
            if exc.errno != errno.ESRCH:
                raise
            if pid == worker:
                return lambda: True
            return lambda: False
    if poll is None:
        if sys.platform == "win32":
            raise OSError(errno.ENOSYS, "select.poll")
        poller = select.poll()
        for fd in fds:
            poller.register(fd, select.POLLIN)

        def host_poll() -> list[tuple[int, int]]:
            return poller.poll()

        ready: Callable[[], list[tuple[int, int]]] = host_poll
    else:
        ready = poll

    def wait() -> bool:
        return _pidfd_wait(fds, worker, ready)

    return wait


def _pidfd_wait(
    fds: dict[int, int], worker: int, poll: Callable[[], list[tuple[int, int]]]
) -> bool:
    """True when ``worker``'s pidfd is ready. An interrupted poll is retried."""
    while True:
        try:
            events = poll()
        except InterruptedError:
            continue
        return any(fds[fd] == worker for fd, _event in events)


def _kill_cli_group(cli: int) -> None:
    """Kill marked orphans and descendant groups before killing this watchdog's group."""
    if sys.platform == "win32":
        return
    groups = _descendant_groups(cli)
    if path := os.environ.get(PGID_FILE_ENV):
        try:
            pids = marked_pids(f"{PGID_FILE_ENV}={path}")
        except (OSError, ValueError, subprocess.SubprocessError):
            pids = []
        for pid in pids:
            if pid not in (cli, os.getpid()):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)
    for group in _reap_order(groups, os.getpgrp()):
        _signal_group(group)


def marked_pids(marker: str) -> list[int]:
    """This user's processes whose initial environment has the exact attempt marker.

    Linux reads ``/proc``; elsewhere ``ps -E`` reads the environment. macOS hides the
    environment of its own system binaries, so a detached one is missed there.
    """
    if Path("/proc/self/environ").exists():
        pids = []
        for environ in Path("/proc").glob("[0-9]*/environ"):
            try:
                if marker.encode() in environ.read_bytes().split(b"\0"):
                    pids.append(int(environ.parent.name))
            except OSError:
                continue
        return pids
    out = subprocess.run(
        ["ps", "-A", "-E", "-ww", "-o", "pid=,command="],
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout
    return [
        int(line.split(None, 1)[0])
        for line in out.splitlines()
        if f" {marker} " in f" {line} "
    ]


def _reap_order(groups: set[int], own: int) -> list[int]:
    """Descendant groups other than ``own``, then ``own`` itself."""
    return [group for group in groups if group != own] + [own]


def _signal_group(pgid: int) -> None:
    """SIGKILL one process group. A group that is already gone is not an error."""
    if sys.platform == "win32":
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return


def _descendant_groups(
    pid: int, table: list[tuple[int, int, int]] | None = None
) -> set[int]:
    """Process groups of ``pid``'s descendants. Bash sessions sit outside the CLI's group."""
    if table is None:
        try:
            table = _process_table()
        except (OSError, ValueError, subprocess.SubprocessError):
            return set()
    children: dict[int, list[tuple[int, int]]] = {}
    for child, parent, pgid in table:
        children.setdefault(parent, []).append((child, pgid))
    groups: set[int] = set()
    stack = [pid]
    while stack:
        for child, pgid in children.get(stack.pop(), ()):
            groups.add(pgid)
            stack.append(child)
    return groups


def _process_table() -> list[tuple[int, int, int]]:
    """``(pid, ppid, pgid)`` of every process: ``/proc`` on Linux, ``ps`` elsewhere."""
    if os.path.exists("/proc/self/stat"):
        table: list[tuple[int, int, int]] = []
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat") as stat:
                    row = _proc_stat_row(name, stat.read())
            except OSError:
                continue
            table.append(row)
        return table
    out = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,pgid="],
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout
    return [
        (int(a), int(b), int(c))
        for a, b, c in (line.split() for line in out.splitlines())
    ]


def _proc_stat_row(name: str, stat: str) -> tuple[int, int, int]:
    """``(pid, ppid, pgid)`` from ``/proc/<pid>/stat``. ``comm`` may contain ``)``."""
    fields = stat.rsplit(")", 1)[1].split()
    return int(name), int(fields[1]), int(fields[2])


if __name__ == "__main__":
    main()
