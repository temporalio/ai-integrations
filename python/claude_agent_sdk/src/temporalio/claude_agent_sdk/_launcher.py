"""Starts Claude Code so that it ends when the Worker process that started it ends.

On Linux and macOS the runner gives the SDK a short shell script as ``cli_path``,
and the script runs this file (standard library only, so it starts in tens of
milliseconds) with the real engine and the SDK's arguments.

When a Worker process dies (a crash, an out-of-memory kill, SIGKILL), nothing in it
can stop its engines, and an engine would go on with its turn, built-in tools
included, while Temporal runs the step again on another Worker. Tested: after a
SIGKILL of its parent, Claude Code kept running Bash commands for 30 seconds, and
waited about an hour for a model API that did not answer.

- Linux: the engine gets SIGTERM when the Worker thread that started it ends
  (``PR_SET_PDEATHSIG``; the SDK starts engines from the Worker's event loop
  thread, which lives as long as the Worker). This file then becomes the engine
  (exec): no process is added. On SIGTERM Claude Code ends its Bash commands, then
  exits.
- Other systems (macOS), or Linux without ``prctl``: this file stays as the
  engine's parent. It passes on SIGTERM, SIGINT and SIGHUP, and once its own parent
  is not the Worker any more (the Worker is gone), it sends the engine SIGTERM, and
  SIGKILL after ``GRACE_SECONDS``.

Windows has no such script (it would run through cmd.exe); there the runner puts each
engine in a job object that ends it, and the processes it starts, with the Worker
process.
"""

from __future__ import annotations

import os
import signal
import sys
import time

WORKER_PID = "TCA_WORKER_PID"
"""The Worker's process id, from the runner. Removed before the engine starts."""

CHECK = "--tca-launcher-check"
"""As the only argument after the engine: print "ok" and start nothing (the runner's check)."""

GRACE_SECONDS = 10.0
"""How long the engine has to stop once the Worker is gone (supervising only)."""

TERM_GRACE_SECONDS = 3.0
"""How long the engine has after a SIGTERM it was passed (supervising only).

The SDK kills the process it started 5 seconds after its SIGTERM (claude-agent-sdk
0.2): the supervisor ends the engine first, so that kill never leaves it running.
"""

POLL_SECONDS = 0.2
"""How often a supervising launcher looks at its parent."""

_PR_SET_PDEATHSIG = 1


def _worker_gone(worker: int | None) -> bool:
    """Whether the launcher's parent is no longer the Worker (it was adopted)."""
    return worker is not None and os.getppid() != worker


def _end_with_parent_linux() -> bool:
    """Ask Linux for SIGTERM when the parent thread ends. Returns whether it is set."""
    try:
        import ctypes  # some Python builds have none: then the supervisor does it

        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0) == 0
    except (ImportError, OSError, AttributeError):
        return False


def _exec(argv: list[str]) -> None:
    """Become the engine, with the signal handling a new process gets.

    Python ignores SIGPIPE and SIGXFSZ, and an ignored signal stays ignored across
    exec; the SDK starts the engine with them restored, and so does this.
    """
    for name in ("SIGPIPE", "SIGXFSZ", "SIGXFZ"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_DFL)
    try:
        os.execvp(argv[0], argv)
    except OSError as err:
        sys.stderr.write(f"Claude Code cannot start ({argv[0]}): {err}\n")
        sys.exit(127)


def supervise(argv: list[str], worker: int | None) -> int:
    """Run the engine as a child and end it when the Worker is gone.

    Args:
        argv: The engine and its arguments.
        worker: The Worker's process id, or None to only pass signals on.

    Returns:
        The engine's exit code (128 plus the signal, if a signal ended it).
    """
    import subprocess

    started: list[subprocess.Popen[bytes]] = []
    early: list[int] = []  # signals that came before the engine started
    kill_at: list[float] = []  # when to SIGKILL the engine, once it has to stop

    def kill_after(seconds: float) -> None:
        if not kill_at:
            kill_at.append(time.monotonic() + seconds)

    def pass_on(signum: int, frame: object) -> None:
        del frame
        if started:
            started[0].send_signal(signum)  # nothing happens once it has exited
        else:
            early.append(signum)
        if signum == signal.SIGTERM:
            kill_after(TERM_GRACE_SECONDS)

    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        signal.signal(getattr(signal, name), pass_on)
    try:
        child = subprocess.Popen(argv)
    except OSError as err:
        sys.stderr.write(f"Claude Code cannot start ({argv[0]}): {err}\n")
        return 127
    started.append(child)
    for signum in early:
        child.send_signal(signum)
    # The engine holds the SDK's pipes now. Letting go of them here means the SDK
    # sees them close when the engine exits, not when this process does.
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    os.close(devnull)
    stopping = False
    while True:
        try:
            code = child.wait(timeout=POLL_SECONDS)
            break
        except subprocess.TimeoutExpired:
            pass
        if not stopping and _worker_gone(worker):
            stopping = True
            child.terminate()
            kill_after(GRACE_SECONDS)
        if kill_at and time.monotonic() > kill_at[0]:
            child.kill()
    return 128 - code if code < 0 else code


def main() -> None:
    """Start the engine named in ``sys.argv[1]`` with the arguments after it."""
    if len(sys.argv) < 2:
        sys.exit("usage: _launcher.py ENGINE [ARGUMENT ...]")
    if sys.argv[2:] == [CHECK]:
        print("ok")
        return
    raw = os.environ.pop(WORKER_PID, "")
    worker = int(raw) if raw.isdigit() else None
    argv = sys.argv[1:]
    if sys.platform.startswith("linux") and _end_with_parent_linux():
        if _worker_gone(worker):
            sys.exit(128 + int(signal.SIGTERM))  # it ended before the engine started
        _exec(argv)
    sys.exit(supervise(argv, worker))


if __name__ == "__main__":
    main()
