"""Parent-exit watch used when the worker is SIGKILLed.

Darwin's host watch is kqueue, so these tests drive the Linux pidfd branch
through ``_arm_pidfd``'s opener and poll. That is not a live pidfd. The live
reap runs only where pidfd is the host watch; CI Linux runs it.
"""

from __future__ import annotations

import errno
import io
import json
import os
import select
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from temporalio.claude_agent_sdk import _activity, _launcher
from temporalio.claude_agent_sdk._models import (
    CLAUDE_TURN_REJECTED_ERROR_TYPE,
    ClaudeTurnInput,
)
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from .helpers import fixtures as fx
from .helpers import pidfd_reap

# A raise, unlike a marker, also tells type checkers that the POSIX calls below are unreachable.
if sys.platform == "win32":
    raise pytest.skip.Exception(fx.POSIX_ONLY_REASON, allow_module_level=True)

_LAUNCHER = "temporalio.claude_agent_sdk._launcher"


def test_worker_death_reaps_a_genuinely_orphaned_helper(tmp_path: Path) -> None:
    """An intermediate parent exits before worker death, defeating the descendant walk."""
    middle = tmp_path / "middle.py"
    middle.write_text(
        "import os, signal, json\n"
        "r, w = os.pipe()\n"
        "if os.fork():\n"
        "    os._exit(0)\n"
        "os.close(w)\n"
        "os.setsid()\n"
        "os.read(r, 1)\n"
        "os.close(r)\n"
        "os.write(int(os.environ['PR103_READY']), (json.dumps({'orphan': os.getpid(), 'parent': os.getppid()})+'\\n').encode())\n"
        "signal.pause()\n"
    )
    cli = tmp_path / "cli.py"
    cli.write_text(
        "import os, sys, subprocess, signal, json\n"
        "fd = int(os.environ['PR103_READY'])\n"
        "subprocess.run([sys.executable, '-I', '-S', sys.argv[1]], pass_fds=(fd,), check=True)\n"
        "os.write(fd, (json.dumps({'cli': os.getpid()})+'\\n').encode())\n"
        "signal.pause()\n"
    )
    worker_code = (
        "import os, sys, subprocess, signal\n"
        "subprocess.Popen([sys.executable, '-I', '-S', sys.argv[1], sys.executable, '-I', '-S', sys.argv[2], sys.argv[3]], pass_fds=(int(os.environ['PR103_READY']),))\n"
        "signal.pause()\n"
    )
    pgid_file = tmp_path / "attempt.pgid"
    read_fd, write_fd = os.pipe()
    worker = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            worker_code,
            _launcher.__file__,
            str(cli),
            str(middle),
        ],
        env={
            **os.environ,
            _launcher.PGID_FILE_ENV: str(pgid_file),
            "PR103_READY": str(write_fd),
        },
        pass_fds=(write_fd,),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    os.close(write_fd)
    orphan = cli_pid = 0
    close_watch: Callable[[], None] = lambda: None
    pidfds: list[int] = []
    try:
        data = b""
        messages: dict[str, int] = {}
        deadline = time.monotonic() + 5
        while "orphan" not in messages or "cli" not in messages:
            assert select.select(
                [read_fd], [], [], max(0, deadline - time.monotonic())
            )[0], "helper did not start"
            chunk = os.read(read_fd, 4096)
            assert chunk, "worker exited before reporting its helpers"
            data += chunk
            while b"\n" in data:
                line, data = data.split(b"\n", 1)
                messages.update(json.loads(line))
        orphan, cli_pid = messages["orphan"], messages["cli"]
        assert messages["parent"] != cli_pid
        table = _launcher._process_table()
        assert next(parent for pid, parent, _ in table if pid == orphan) not in {
            cli_pid,
            worker.pid,
        }
        assert orphan not in _launcher._descendant_groups(cli_pid, table)
        assert orphan in _launcher.marked_pids(f"{_launcher.PGID_FILE_ENV}={pgid_file}")
        watchdogs = [
            pid
            for pid, parent, group in table
            if parent == cli_pid and group == cli_pid
        ]
        assert len(watchdogs) == 1, watchdogs
        tracked = {cli_pid, orphan, *watchdogs}
        if hasattr(select, "kqueue"):
            kq = getattr(select, "kqueue")()
            close_watch = kq.close
            kq.control(
                [
                    getattr(select, "kevent")(
                        pid,
                        filter=getattr(select, "KQ_FILTER_PROC"),
                        flags=(
                            getattr(select, "KQ_EV_ADD")
                            | getattr(select, "KQ_EV_ONESHOT")
                        ),
                        fflags=getattr(select, "KQ_NOTE_EXIT"),
                    )
                    for pid in tracked
                ],
                0,
            )

            def wait_exits(timeout: float) -> set[int]:
                return {
                    int(event.ident)
                    for event in kq.control(None, len(tracked), timeout)
                }

        else:
            poller = select.poll()
            by_fd = {getattr(os, "pidfd_open")(pid): pid for pid in tracked}
            pidfds.extend(by_fd)
            for fd in by_fd:
                poller.register(fd, select.POLLIN)

            def wait_exits(timeout: float) -> set[int]:
                got: set[int] = set()
                for fd, _ in poller.poll(int(timeout * 1000)):
                    got.add(by_fd[fd])
                    poller.unregister(fd)
                return got

        worker.kill()
        worker.wait(timeout=3)
        exited: set[int] = set()
        deadline = time.monotonic() + 3
        while exited != tracked and time.monotonic() < deadline:
            remaining = max(0, deadline - time.monotonic())
            exited.update(wait_exits(remaining))
        assert cli_pid in exited, "worker death did not kill the CLI"
        assert set(watchdogs) <= exited, "watchdog did not finish"
        assert orphan in exited, (
            f"orphan {orphan} survived worker, CLI and watchdog death"
        )
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=3)
        for pid in (orphan, cli_pid):
            if pid:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        os.close(read_fd)
        for fd in pidfds:
            os.close(fd)
        close_watch()


class _Pidfds:
    def __init__(self, results: dict[int, int | OSError]) -> None:
        self.results = results
        self.opened: list[int] = []

    def open(self, pid: int) -> int:
        self.opened.append(pid)
        value = self.results[pid]
        if isinstance(value, OSError):
            raise value
        return value


def _poll_script(
    *steps: list[tuple[int, int]] | BaseException,
) -> Callable[[], list[tuple[int, int]]]:
    pending = list(steps)

    def poll() -> list[tuple[int, int]]:
        step = pending.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    return poll


def _unpolled() -> list[tuple[int, int]]:
    raise AssertionError("pidfd poll ran")


@pytest.mark.parametrize(
    ("has_kqueue", "has_pidfd", "kind"),
    [
        (True, True, "kqueue"),
        (True, False, "kqueue"),
        (False, True, "pidfd"),
        (False, False, None),
    ],
)
def test_watch_kind_prefers_kqueue_then_pidfd(
    has_kqueue: bool, has_pidfd: bool, kind: str | None
) -> None:
    assert _launcher._watch_kind(has_kqueue=has_kqueue, has_pidfd=has_pidfd) == kind


def test_host_watch_uses_that_order() -> None:
    kind = _launcher._watch_kind(
        has_kqueue=hasattr(select, "kqueue"),
        has_pidfd=getattr(os, "pidfd_open", None) is not None,
    )
    assert _launcher.parent_death_watch() == kind
    assert _launcher._host_watch() == kind


def _hide_parent_death_watch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(select, "kqueue", raising=False)
    monkeypatch.delattr(os, "pidfd_open", raising=False)


def test_a_host_with_no_watch_does_not_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_parent_death_watch(monkeypatch)

    def fail_fork() -> int:
        raise AssertionError("fork")

    monkeypatch.setattr(f"{_LAUNCHER}.os.fork", fail_fork)
    assert _launcher.parent_death_watch() is None
    assert _launcher._arm_watchdog(1) is False


@pytest.mark.parametrize("kind", ["kqueue", "pidfd"])
def test_kqueue_and_pidfd_remain_accepted(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    if kind == "kqueue":
        monkeypatch.setattr(select, "kqueue", lambda: None, raising=False)
        monkeypatch.delattr(os, "pidfd_open", raising=False)
    else:
        monkeypatch.delattr(select, "kqueue", raising=False)
        monkeypatch.setattr(os, "pidfd_open", lambda _pid: 0, raising=False)
    assert _launcher.parent_death_watch() == kind
    _activity._require_parent_death_watch()


async def test_a_turn_without_a_parent_death_watch_is_refused_before_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_parent_death_watch(monkeypatch)
    monkeypatch.setattr(
        "temporalio.claude_agent_sdk._activity.os.name", "posix", raising=False
    )

    def cli_started(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("CLI resolution started")

    monkeypatch.setattr(_activity.ClaudeTurnActivity, "_cli", cli_started)
    with pytest.raises(ApplicationError) as info:
        await ActivityEnvironment().run(
            _activity.ClaudeTurnActivity().run_turn, ClaudeTurnInput(prompt="p")
        )
    err = info.value
    assert err.type == CLAUDE_TURN_REJECTED_ERROR_TYPE
    assert err.non_retryable is True
    assert "parent-death watch" in str(err)


@pytest.mark.parametrize(
    ("kind", "fired"), [("pidfd", True), ("kqueue", False)], ids=["pidfd", "kqueue"]
)
def test_arm_uses_the_host_watch(
    monkeypatch: pytest.MonkeyPatch, kind: str, fired: bool
) -> None:
    monkeypatch.setattr(_launcher, "_host_watch", lambda: kind)
    seen: dict[str, tuple[int, int]] = {}

    def fake(worker: int, cli: int) -> Callable[[], bool]:
        seen["pids"] = (worker, cli)
        return lambda: fired

    monkeypatch.setattr(_launcher, f"_arm_{kind}", fake)
    assert _launcher._arm(4, 5)() is fired
    assert seen["pids"] == (4, 5)


def test_pidfd_default_opener_is_os_pidfd_open() -> None:
    if hasattr(os, "pidfd_open"):
        pytest.skip("this host has pidfd; the live reap covers the real opener")
    with pytest.raises(AttributeError):
        _launcher._arm_pidfd(1, 2)


def test_pidfd_is_true_when_the_worker_is_already_gone() -> None:
    pidfds = _Pidfds({7: OSError(errno.ESRCH, "gone")})
    wait = _launcher._arm_pidfd(7, 8, open_pidfd=pidfds.open, poll=_unpolled)
    assert wait() is True
    assert pidfds.opened == [7]


def test_pidfd_is_false_when_the_cli_is_already_gone() -> None:
    pidfds = _Pidfds({7: 70, 8: OSError(errno.ESRCH, "gone")})
    wait = _launcher._arm_pidfd(7, 8, open_pidfd=pidfds.open, poll=_unpolled)
    assert wait() is False
    assert pidfds.opened == [7, 8]


def test_pidfd_open_error_propagates() -> None:
    pidfds = _Pidfds({7: OSError(errno.EPERM, "denied")})
    with pytest.raises(OSError) as info:
        _launcher._arm_pidfd(7, 8, open_pidfd=pidfds.open)
    assert info.value.errno == errno.EPERM
    assert pidfds.opened == [7]


def test_pidfd_wait_is_true_when_the_worker_exits() -> None:
    pidfds = _Pidfds({7: 70, 8: 80})
    wait = _launcher._arm_pidfd(
        7,
        8,
        open_pidfd=pidfds.open,
        poll=_poll_script([(70, select.POLLIN), (80, select.POLLIN)]),
    )
    assert wait() is True


def test_pidfd_wait_is_false_when_only_the_cli_exits() -> None:
    pidfds = _Pidfds({7: 70, 8: 80})
    wait = _launcher._arm_pidfd(
        7, 8, open_pidfd=pidfds.open, poll=_poll_script([(80, select.POLLIN)])
    )
    assert wait() is False


def test_pidfd_wait_retries_an_interrupted_poll() -> None:
    pidfds = _Pidfds({7: 70, 8: 80})
    wait = _launcher._arm_pidfd(
        7,
        8,
        open_pidfd=pidfds.open,
        poll=_poll_script(InterruptedError(), [(70, select.POLLIN)]),
    )
    assert wait() is True


def test_descendant_groups_include_a_detached_session() -> None:
    # cli 2, a child still in that group, a grandchild in its own session, and a stranger.
    table = [(2, 1, 2), (3, 2, 2), (4, 3, 9), (5, 1, 5)]
    assert _launcher._descendant_groups(2, table) == {2, 9}


@pytest.mark.parametrize(
    "exc",
    [OSError("proc"), ValueError("ps"), subprocess.SubprocessError("ps")],
)
def test_descendant_groups_are_empty_when_the_table_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    def boom() -> list[tuple[int, int, int]]:
        raise exc

    monkeypatch.setattr(_launcher, "_process_table", boom)
    assert _launcher._descendant_groups(1) == set()


def test_linux_process_table_parses_proc_stat_past_a_comm_with_parentheses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats = {
        "2": "2 (python) S 1 2 2\n",
        "4": "4 (bash (session)) S 2 9 9\n",
    }

    def fail_ps(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("ps")

    def open_stat(path: str, *_args: object, **_kwargs: object) -> io.StringIO:
        name = path.removeprefix("/proc/").removesuffix("/stat")
        if name == "6":
            raise OSError(errno.ENOENT, "gone")
        return io.StringIO(stats[name])

    monkeypatch.setattr(
        f"{_LAUNCHER}.os.path.exists", lambda path: path == "/proc/self/stat"
    )
    monkeypatch.setattr(
        f"{_LAUNCHER}.os.listdir", lambda _path: ["2", "4", "6", "cpuinfo", "self"]
    )
    monkeypatch.setattr("builtins.open", open_stat)
    monkeypatch.setattr(f"{_LAUNCHER}.subprocess.run", fail_ps)
    assert set(_launcher._process_table()) == {(2, 1, 2), (4, 2, 9)}


def test_reap_order_signals_other_groups_then_its_own() -> None:
    order = _launcher._reap_order({2, 9, 10}, 2)
    assert order[-1] == 2
    assert set(order) == {2, 9, 10}
    assert order.count(2) == 1
    assert _launcher._reap_order(set(), 2) == [2]


def test_marked_orphans_are_killed_before_the_watchdogs_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_launcher.PGID_FILE_ENV, "/attempt")
    monkeypatch.setattr(_launcher, "marked_pids", lambda _marker: [4, 5, 9])
    monkeypatch.setattr(_launcher, "_descendant_groups", lambda _cli: {2})
    monkeypatch.setattr(os, "getpid", lambda: 5)
    monkeypatch.setattr(os, "getpgrp", lambda: 2)
    killed: list[tuple[str, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, _sig: killed.append(("pid", pid)))
    monkeypatch.setattr(
        _launcher, "_signal_group", lambda group: killed.append(("group", group))
    )

    _launcher._kill_cli_group(4)

    assert killed == [("pid", 9), ("group", 2)]


@pytest.mark.parametrize("error", [ProcessLookupError, PermissionError])
def test_reap_continues_after_a_descendant_group_is_gone(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    monkeypatch.setattr(_launcher, "_descendant_groups", lambda _pid: {9, 2})
    monkeypatch.setattr(f"{_LAUNCHER}.os.getpgrp", lambda: 2)
    seen: list[int] = []

    def killpg(pgid: int, sig: int) -> None:
        assert sig == signal.SIGKILL
        seen.append(pgid)
        if pgid == 9:
            raise error

    monkeypatch.setattr(f"{_LAUNCHER}.os.killpg", killpg)
    _launcher._kill_cli_group(4)
    assert seen[-1] == 2
    assert seen.count(2) == 1
    assert 9 in seen


@pytest.mark.skipif(
    _launcher._host_watch() != "pidfd",
    reason=(
        "live pidfd reap runs only when pidfd is the host watch; "
        "CI Linux runs it. This host does not."
    ),
)
def test_pidfd_reaps_a_detached_descendant_when_the_worker_is_killed(
    tmp_path: Path,
) -> None:
    """The launcher is exec'd with ``python -I -S``. The test does not signal the CLI."""
    pidfd_reap.reap(str(Path(_launcher.__file__)), tmp_path)
