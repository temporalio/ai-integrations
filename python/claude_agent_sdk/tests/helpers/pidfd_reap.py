"""Live pidfd reap of a detached CLI descendant. Standard library only.

A host with kqueue refuses. Darwin does not run this; CI Linux does, and so
does a Linux container that already has a Python image.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HANG = """\
import os
import time
from pathlib import Path

child = os.fork()
if child == 0:
    os.setsid()
    Path(os.environ["PIDFD_CHILD"]).write_text(str(os.getpid()))
    time.sleep(30)
    os._exit(0)
Path(os.environ["PIDFD_CLI"]).write_text(str(os.getpid()))
time.sleep(30)
"""

_WORKER = """\
import os
import subprocess
import sys
import time

subprocess.Popen(
    [sys.executable, "-I", "-S", sys.argv[1], sys.executable, "-I", "-S", sys.argv[2]],
    env={**os.environ, "TEMPORAL_CLAUDE_PGID_FILE": sys.argv[3]},
)
time.sleep(30)
"""


def _pidfd_host() -> bool:
    return not hasattr(select, "kqueue") and hasattr(os, "pidfd_open")


def _running(pid: int) -> bool:
    """False when ``pid`` is gone or a zombie. A zombie has been killed and cannot run."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def _ppid(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    raise AssertionError(f"/proc/{pid}/status has no PPid")


def _tail(path: Path) -> str:
    root = path.parent
    names = sorted(item.name for item in root.iterdir()) if root.exists() else []
    text = path.read_text()[-2000:] if path.exists() else ""
    return f"files={names}\n{text}"


def _until(path: Path, worker: subprocess.Popen[bytes], log_path: Path) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size:
            return
        if worker.poll() is not None:
            raise AssertionError(
                f"worker exited {worker.returncode} before {path.name}\n{_tail(log_path)}"
            )
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path.name}\n{_tail(log_path)}")


def reap(launcher: str, root: Path) -> None:
    """SIGKILL ``launcher``'s parent. The pidfd watchdog must reap the CLI and its session."""
    if sys.platform == "win32" or not _pidfd_host():
        raise AssertionError(
            "pidfd is not this host's parent-exit watch; CI Linux runs the live reap"
        )
    cli_script = root / "hang.py"
    worker_script = root / "worker.py"
    cli_script.write_text(_HANG)
    worker_script.write_text(_WORKER)
    cli_file = root / "cli"
    child_file = root / "child"
    pgid_file = root / "pgid"
    log_path = root / "worker.log"
    log = log_path.open("w")
    worker: subprocess.Popen[bytes] | None = None
    cli_pid = 0
    child_pid = 0
    try:
        worker = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                str(worker_script),
                launcher,
                str(cli_script),
                str(pgid_file),
            ],
            env={
                **os.environ,
                "PIDFD_CLI": str(cli_file),
                "PIDFD_CHILD": str(child_file),
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        _until(cli_file, worker, log_path)
        _until(child_file, worker, log_path)
        _until(pgid_file, worker, log_path)
        cli_pid = int(cli_file.read_text())
        child_pid = int(child_file.read_text())
        assert int(pgid_file.read_text()) == cli_pid
        assert os.getpgid(cli_pid) == cli_pid
        assert os.getpgid(child_pid) != cli_pid
        assert _ppid(cli_pid) == worker.pid
        assert _ppid(child_pid) == cli_pid
        os.kill(worker.pid, signal.SIGKILL)
        worker.wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not _running(cli_pid) and not _running(child_pid):
                return
            time.sleep(0.05)
        raise AssertionError(
            f"cli running={_running(cli_pid)} "
            f"descendant running={_running(child_pid)}\n{_tail(log_path)}"
        )
    finally:
        log.close()
        if worker is not None and worker.poll() is None:
            os.kill(worker.pid, signal.SIGKILL)
            worker.wait(timeout=5)
        for pid in (cli_pid, child_pid):
            if not pid:
                continue
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def main() -> None:
    """Run the live reap against the launcher path in ``argv``."""
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(tempfile.mkdtemp())
    root.mkdir(parents=True, exist_ok=True)
    try:
        reap(sys.argv[1], root)
    except AssertionError as exc:
        raise SystemExit(str(exc)) from None
    print("pidfd reap: cli and detached descendant were reaped", flush=True)


if __name__ == "__main__":
    main()
