"""Claude Code ends with the Worker process that started it.

When a Worker process dies (a crash, an out-of-memory kill, SIGKILL), its engine must
not go on with its turn, running built-in tools, while Temporal runs the step again on
another Worker. On Linux and macOS the engine starts through a launcher
(``_launcher``); on Windows each engine joins a job object that ends it with the
Worker process. And the hook denies every call once the Worker is gone: nothing holds
its lock on ``worker.lock`` any more.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.claude_agent_sdk import (
    ClaudeAgentSdkRunner,
    SegmentInput,
    _defer_hook,
    _launcher,
    _runner,
)
from temporalio.client import Client
from temporalio.exceptions import ApplicationError
from tests.helpers.fake_messages_api import engine_env, start_with_policy
from tests.helpers.processes import alive, descendants
from tests.refund.policy import refund_policy
from tests.refund.workflows import RefundAgentWorkflow
from tests.test_crash import kill, start_worker, wait_until
from tests.test_engine import TOOLS as REFUND_TOOLS
from tests.test_workflow_engine import hang_on_request, shared_settings

pytestmark = pytest.mark.timeout(240)
posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="the launcher script is for Linux and macOS"
)

FAKE_ENGINE = """\
import json, os, signal, sys, time
if sys.argv[1:] == ["-v"]:
    print(json.dumps(["version", os.environ.get("TCA_WORKER_PID")]))
    sys.exit(0)
if sys.argv[1:2] == ["exit"]:
    print(json.dumps([sys.argv[2:], os.environ.get("TCA_WORKER_PID")]))
    sys.exit(7)
if sys.argv[2:3] == ["ignore-term"]:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(sys.argv[1], "w") as f:
    f.write(str(os.getpid()))
time.sleep(120)
"""
"""Stands in for Claude Code: writes its process id, then waits (or prints and exits)."""

FAKE_WORKER = """\
import json, os, subprocess, sys, time
subprocess.Popen(json.loads(sys.argv[1]), env={**os.environ, "TCA_WORKER_PID": str(os.getpid())})
time.sleep(120)
"""
"""Stands in for a Worker process: starts the launcher like the SDK does, then waits."""

SUPERVISE = """\
import os, runpy, sys
launcher = runpy.run_path(sys.argv[1])
worker = os.environ.pop("TCA_WORKER_PID", "")
sys.exit(launcher["supervise"](sys.argv[2:], int(worker) if worker else None))
"""
"""Runs the launcher's supervising mode (what macOS uses) on any system."""


def fake_engine(tmp_path: Path) -> Path:
    path = tmp_path / "fake_claude"
    path.write_text(f"#!{sys.executable}\n{FAKE_ENGINE}", encoding="utf-8")
    path.chmod(0o755)
    return path


async def gone(pid: int, timeout: float) -> None:
    await wait_until(lambda: not alive(pid), timeout=timeout)


# ---- the hook ----


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("TCA_ALLOW_ID", "TCA_ANSWERED_IDS", "TCA_TOOL_ACTIVITIES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TCA_HOOK_DIR", str(tmp_path))
    return tmp_path


def decision(name: str, tool_use_id: str = "t1", **event: Any) -> str:
    out = _defer_hook.decide({"tool_name": name, "tool_use_id": tool_use_id, **event})
    return str(out.get("permissionDecision", "run"))


def test_hook_decides_as_usual_while_the_worker_holds_its_lock(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TCA_TOOL_ACTIVITIES", "Bash")
    lock = _runner._hold_worker_lock(str(run_dir), required=True)
    try:
        assert decision("Glob", "t1") == "run"
        assert decision("Bash", "t2") == "defer"
    finally:
        _runner._release_worker_lock(lock)


def test_hook_stops_once_the_worker_is_gone(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing holds the lock: the Worker died. Built-in tools are denied; a durable
    call still defers, which ends the run (no one runs it); a tool step runs nothing."""
    monkeypatch.setenv("TCA_TOOL_ACTIVITIES", "Bash")
    _runner._release_worker_lock(_runner._hold_worker_lock(str(run_dir), required=True))
    assert decision("Glob", "t1") == "deny"
    assert decision("mcp__durable__count", "t2") == "defer"
    monkeypatch.setenv("TCA_ALLOW_ID", "t3")
    assert decision("Bash", "t3") == "deny"
    denied = {p.name: p.read_text() for p in (run_dir / "denied").iterdir()}
    assert denied == {"t1": "stopped", "t3": "stopped"}


def test_hook_without_a_lock_file_decides_as_before(run_dir: Path) -> None:
    assert not (run_dir / _defer_hook.WORKER_LOCK).exists()
    assert decision("Glob") == "run"


def refuse_locks(fd: int) -> None:
    """Like some network and FUSE file systems."""
    del fd
    raise OSError(errno.ENOLCK, "No locks available")


def test_where_files_cannot_be_locked_the_step_stops(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the lock the hook could not tell that a dead Worker's engine should
    stop, so by default (``engine_cleanup="required"``) the step stops, before the
    engine starts, with an error that says what to fix."""
    monkeypatch.setattr(_runner, "_lock_file", refuse_locks)
    with pytest.raises(ApplicationError) as raised:
        _runner._hold_worker_lock(str(run_dir), required=True)
    assert raised.value.type == _runner.CLEANUP_UNAVAILABLE
    assert not raised.value.non_retryable  # another Worker can run the step
    assert "cannot lock files" in str(raised.value) and "TMPDIR" in str(raised.value)
    assert 'engine_cleanup="best_effort"' in str(raised.value)
    assert not (run_dir / _defer_hook.WORKER_LOCK).exists()


def test_where_files_cannot_be_locked_a_best_effort_step_still_runs(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``engine_cleanup="best_effort"``: the step runs, without the lock file (so the
    hook takes the Worker to be there), and a warning says so."""
    monkeypatch.setattr(_runner, "_lock_file", refuse_locks)
    monkeypatch.setattr(_runner, "_warned", set())
    with pytest.warns(UserWarning, match="cannot lock files"):
        lock = _runner._hold_worker_lock(str(run_dir), required=False)
    assert lock is None and not (run_dir / _defer_hook.WORKER_LOCK).exists()
    _runner._release_worker_lock(lock)
    assert decision("Glob") == "run"


@posix_only
def test_hooks_that_test_the_lock_together_both_see_the_worker_gone(
    run_dir: Path,
) -> None:
    """Hooks of calls Claude Code runs together test the lock at the same time: one
    hook's test must not look like the Worker to another."""
    _runner._release_worker_lock(_runner._hold_worker_lock(str(run_dir), required=True))
    other = os.open(run_dir / _defer_hook.WORKER_LOCK, os.O_RDWR)
    try:
        if sys.platform != "win32":
            import fcntl

            fcntl.flock(other, fcntl.LOCK_SH | fcntl.LOCK_NB)  # another hook, testing
        assert _defer_hook._worker_gone(str(run_dir))
    finally:
        os.close(other)


# ---- the launcher (Linux and macOS) ----


@posix_only
def test_launcher_script_checks_itself_and_starts_the_engine_unchanged(
    tmp_path: Path,
) -> None:
    """The runner's check starts nothing; the SDK's version probe goes straight to the
    engine; otherwise the engine gets every argument as it was, not the launcher's
    variable, and its exit code comes back."""
    script = _runner._write_launch_script(str(fake_engine(tmp_path)))
    check = subprocess.run(
        [script, _launcher.CHECK], capture_output=True, text=True, timeout=60
    )
    assert check.returncode == 0 and check.stdout.strip() == "ok", check.stderr
    worker = {**os.environ, _launcher.WORKER_PID: str(os.getpid())}
    probe = subprocess.run(
        [script, "-v"], capture_output=True, text=True, timeout=60, env=worker
    )
    # The variable is still there: the launcher (which removes it) did not run.
    assert json.loads(probe.stdout) == ["version", str(os.getpid())], probe.stderr
    ran = subprocess.run(
        [script, "exit", "a b", "it's", "--resume"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, _launcher.WORKER_PID: str(os.getpid())},
    )
    assert ran.returncode == 7, ran.stderr
    assert json.loads(ran.stdout) == [["a b", "it's", "--resume"], None]


@posix_only
@pytest.mark.parametrize(
    "how", ["script", "supervise"], ids=["as-on-this-system", "supervising"]
)
async def test_launcher_ends_the_engine_when_its_worker_dies(
    tmp_path: Path, how: str
) -> None:
    """A Worker dies like a power cut; the engine it started is gone within seconds.

    ``as-on-this-system`` uses the runner's script (on Linux the parent-death signal,
    on macOS the supervisor); ``supervising`` runs the supervisor on any system.
    """
    engine, pid_file = fake_engine(tmp_path), tmp_path / "engine.pid"
    if how == "script":
        argv = [_runner._write_launch_script(str(engine)), str(pid_file)]
    else:
        launcher = str(Path(_runner.__file__).with_name("_launcher.py"))
        argv = [sys.executable, "-c", SUPERVISE, launcher, str(engine), str(pid_file)]
    worker = subprocess.Popen([sys.executable, "-c", FAKE_WORKER, json.dumps(argv)])
    try:
        await wait_until(
            lambda: pid_file.exists() and pid_file.read_text().isdigit(), timeout=60
        )
        engine_pid = int(pid_file.read_text())
        assert alive(engine_pid)
        kill(worker)
        await gone(engine_pid, timeout=15)
    finally:
        if worker.poll() is None:
            kill(worker)


@posix_only
async def test_a_launcher_script_a_cleaner_removed_is_written_again(
    tmp_path: Path,
) -> None:
    """An idle Worker's temporary files may be cleaned up. The next engine start
    writes the script again, in a new private folder, never in a folder of the
    same name that someone else may have made since."""
    (tmp_path / "work").mkdir()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path / "work"), env={"ANTHROPIC_API_KEY": "x"}
    )
    first = await runner._launch_path()
    assert first is not None and os.access(first, os.X_OK)
    folder = os.path.dirname(first)
    shutil.rmtree(folder)
    os.mkdir(folder)
    os.chmod(folder, 0o777)  # someone else's now
    try:
        second = await runner._launch_path()
    finally:
        os.rmdir(folder)
    assert second is not None and os.path.dirname(second) != folder
    assert os.access(second, os.X_OK) and _runner._private_folder(
        os.path.dirname(second)
    )


@posix_only
def test_the_launcher_works_without_ctypes(tmp_path: Path) -> None:
    """Some Python builds have no ctypes: the launcher supervises instead."""
    launcher = str(Path(_runner.__file__).with_name("_launcher.py"))
    code = (
        "import runpy, sys; sys.modules['ctypes'] = None; "
        "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    ran = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, launcher, str(fake_engine(tmp_path))]
        + ["exit", "a b"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, _launcher.WORKER_PID: str(os.getpid())},
    )
    assert ran.returncode == 7, ran.stderr
    assert json.loads(ran.stdout) == [["a b"], None]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the launcher execs on Linux only"
)
def test_the_engine_gets_default_signal_handling(tmp_path: Path) -> None:
    """Python ignores SIGPIPE, and an ignored signal stays ignored across exec: the
    engine, and the commands it starts, must get it as the SDK gives it (default),
    or a pipeline like ``yes | head -1`` never ends its first command."""
    engine = tmp_path / "engine.sh"
    engine.write_text("#!/bin/sh\nexec grep SigIgn /proc/self/status\n")
    engine.chmod(0o755)
    script = _runner._write_launch_script(str(engine))
    ran = subprocess.run([script], capture_output=True, text=True, timeout=60)
    ignored = int(ran.stdout.split()[-1], 16)
    if sys.platform != "win32":
        for signum in (signal.SIGPIPE, signal.SIGXFSZ):
            assert not ignored & (1 << (signum - 1)), ran.stdout


@posix_only
async def test_the_supervisor_ends_an_engine_that_ignores_sigterm(
    tmp_path: Path,
) -> None:
    """The SDK kills what it started 5 seconds after its SIGTERM. The supervisor ends
    the engine before that, so the SDK's kill never leaves the engine running."""
    engine, pid_file = fake_engine(tmp_path), tmp_path / "engine.pid"
    launcher = str(Path(_runner.__file__).with_name("_launcher.py"))
    supervisor = subprocess.Popen(
        [sys.executable, "-c", SUPERVISE, launcher, str(engine), str(pid_file)]
        + ["ignore-term"]
    )
    try:
        await wait_until(
            lambda: pid_file.exists() and pid_file.read_text().isdigit(), timeout=60
        )
        engine_pid = int(pid_file.read_text())
        started = time.monotonic()
        supervisor.send_signal(signal.SIGTERM)
        code = await asyncio.to_thread(supervisor.wait, 15)
        assert time.monotonic() - started < 5  # the SDK's own kill comes at 5 s
        if sys.platform != "win32":
            assert code == 128 + int(signal.SIGKILL)
        await gone(engine_pid, timeout=5)
    finally:
        if supervisor.poll() is None:
            kill(supervisor)


@posix_only
async def test_supervising_launcher_passes_signals_on(tmp_path: Path) -> None:
    """The SDK stops the engine with SIGTERM: through the supervisor too."""
    engine, pid_file = fake_engine(tmp_path), tmp_path / "engine.pid"
    launcher = str(Path(_runner.__file__).with_name("_launcher.py"))
    supervisor = subprocess.Popen(
        [sys.executable, "-c", SUPERVISE, launcher, str(engine), str(pid_file)]
    )
    try:
        await wait_until(
            lambda: pid_file.exists() and pid_file.read_text().isdigit(), timeout=60
        )
        engine_pid = int(pid_file.read_text())
        supervisor.send_signal(signal.SIGTERM)
        assert supervisor.wait(timeout=15) == 128 + int(signal.SIGTERM)
        await gone(engine_pid, timeout=5)
    finally:
        if supervisor.poll() is None:
            kill(supervisor)


EARLY_SIGNAL = """\
import os, runpy, signal, subprocess, sys
launcher = runpy.run_path(sys.argv[1])
start = subprocess.Popen
def popen(argv):
    os.kill(os.getpid(), signal.SIGHUP)  # the SDK stops it while it starts
    return start(argv)
subprocess.Popen = popen
sys.exit(launcher["supervise"](sys.argv[2:], None))
"""


@posix_only
def test_a_signal_before_the_engine_started_reaches_the_engine(
    tmp_path: Path,
) -> None:
    engine, pid_file = fake_engine(tmp_path), tmp_path / "engine.pid"
    launcher = str(Path(_runner.__file__).with_name("_launcher.py"))
    ran = subprocess.run(
        [sys.executable, "-c", EARLY_SIGNAL, launcher, str(engine), str(pid_file)],
        capture_output=True,
        timeout=60,
    )
    if sys.platform != "win32":
        assert ran.returncode == 128 + int(signal.SIGHUP), ran.stderr


@posix_only
def test_an_engine_that_cannot_start_is_reported(tmp_path: Path) -> None:
    launcher = str(Path(_runner.__file__).with_name("_launcher.py"))
    missing = str(tmp_path / "no-claude-here")
    ran = subprocess.run(
        [sys.executable, "-c", SUPERVISE, launcher, missing],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert ran.returncode == 127
    assert ran.stderr.startswith(f"Claude Code cannot start ({missing}): ")


# ---- Windows ----


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows only")
async def test_windows_ends_an_engine_with_its_worker_and_nothing_else() -> None:
    """The engine's job ends it with the Worker process; the Worker's other child
    processes are not in it, and go on."""
    code = (
        "import subprocess, sys, time\n"
        "from temporalio.claude_agent_sdk import _runner\n"
        "sleep = [sys.executable, '-c', 'import time; time.sleep(120)']\n"
        "engine, other = subprocess.Popen(sleep), subprocess.Popen(sleep)\n"
        "_runner._end_with_worker(engine.pid)\n"
        "print(engine.pid, other.pid, flush=True)\n"
        "time.sleep(120)\n"
    )
    worker = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    other = 0
    try:
        assert worker.stdout is not None
        engine, other = (int(pid) for pid in worker.stdout.readline().split())
        assert alive(engine) and alive(other)
        kill(worker)
        await gone(engine, timeout=15)
        assert alive(other)
    finally:
        if worker.poll() is None:
            kill(worker)
        if other and alive(other):
            subprocess.run(["taskkill", "/F", "/PID", str(other)], capture_output=True)


# ---- the real engine ----


async def test_the_sdk_lists_its_engines_and_on_windows_they_are_in_the_job(
    tmp_path: Path,
) -> None:
    """On Windows the runner finds engines in the SDK's own list of the processes it
    started (``_ACTIVE_CHILDREN``) and puts them in the Worker's job as they start.
    On every system, this fails if a claude-agent-sdk version stops keeping it."""
    from claude_agent_sdk._internal.transport import subprocess_cli

    api = start_with_policy(refund_policy)
    arrived, release = hang_on_request(api, 1)  # the engine runs, waiting for Claude
    (tmp_path / "work").mkdir()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path / "work"), env=engine_env(api, str(tmp_path / "cfg"))
    )
    step = asyncio.ensure_future(
        runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="Order A-1001 arrived broken, I want my money back.",
                tools=REFUND_TOOLS,
                transcript=[],
            ),
            1,
        )
    )
    try:
        assert await asyncio.to_thread(arrived.wait, 120)
        pids = [
            getattr(child, "pid", None) for child in subprocess_cli._ACTIVE_CHILDREN
        ]
        assert pids and all(isinstance(pid, int) and alive(pid) for pid in pids)
        if sys.platform == "win32":
            assert all(_runner._in_worker_job(int(pid or 0)) for pid in pids)
    finally:
        release.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(step, 120)
        api.stop()


async def test_real_engine_ends_with_its_worker(
    client: Client, address: str, shop_dir: Path, tmp_path: Path
) -> None:
    """The model call hangs, and the Worker is killed like a power cut: within seconds
    no process it started is left. (Before, Claude Code waited for the model as long
    as the API took, up to about an hour, then went on with its turn.)"""
    api = start_with_policy(refund_policy)
    arrived, release = hang_on_request(api, 1)  # the first model call never answers
    queue = f"lifetime-{uuid.uuid4().hex[:8]}"
    (tmp_path / "work").mkdir()
    env = {
        **shared_settings(tmp_path, shop_dir, "held"),
        **engine_env(api, str(tmp_path / "claude-config")),
    }
    worker = await start_worker(address, queue, env, tmp_path / "w1.log", real=True)
    handle = await client.start_workflow(
        RefundAgentWorkflow.run,
        "Order A-1001 arrived broken, I want my money back.",
        id=queue,
        task_queue=queue,
    )
    try:
        assert await asyncio.to_thread(arrived.wait, 120)
        started = descendants(worker.pid)
        assert started, "Claude Code runs under the Worker process"
        kill(worker)
        deadline = time.monotonic() + 20
        while any(alive(pid) for pid in started) and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        assert not [pid for pid in started if alive(pid)], (
            tmp_path / "w1.log"
        ).read_text(errors="replace")[-3000:]
    finally:
        release.set()
        if worker.poll() is None:
            kill(worker)
        with contextlib.suppress(Exception):
            await handle.terminate()
        api.stop()


def refund_step() -> SegmentInput:
    return SegmentInput(
        session_id=str(uuid.uuid4()),
        prompt="Order A-1001 arrived broken, I want my money back.",
        tools=REFUND_TOOLS,
        transcript=[],
    )


def noexec_launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Launcher scripts that cannot run, like those in a folder mounted noexec."""
    written: list[str] = []

    def write(engine: str) -> str:
        script = tmp_path / f"launch-{len(written)}"
        script.write_text("#!/bin/sh\nexit 0\n")  # not executable
        written.append(engine)
        return str(script)

    monkeypatch.setattr(_runner, "_launch_scripts", {})
    monkeypatch.setattr(_runner, "_write_launch_script", write)
    return written


@pytest.mark.parametrize(
    "cause",
    [
        pytest.param("launcher", marks=posix_only),
        "lock",
        "job",
    ],
)
async def test_real_engine_a_step_without_its_cleanup_stops_before_claude_code_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cause: str
) -> None:
    """The default ``engine_cleanup="required"``: a step whose engine could not end
    with its Worker process stops before Claude Code starts (no model request), and
    the error says what to fix. A launcher check that failed is made again by the
    next step, so a fixed Worker goes on. (A Windows job object that cannot be
    created is simulated here, on every system.)"""
    written = noexec_launcher(tmp_path, monkeypatch) if cause == "launcher" else []
    if cause == "lock":
        monkeypatch.setattr(_runner, "_lock_file", refuse_locks)
    if cause == "job":
        monkeypatch.setattr(
            _runner, "_worker_job", lambda: "cannot create a job object (error 5)"
        )
    api = start_with_policy(refund_policy)
    (tmp_path / "work").mkdir()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path / "work"), env=engine_env(api, str(tmp_path / "cfg"))
    )
    try:
        for _ in range(2):
            with pytest.raises(ApplicationError) as raised:
                await runner.run(refund_step(), 1)
            assert raised.value.type == _runner.CLEANUP_UNAVAILABLE
            assert not raised.value.non_retryable
            assert {
                "launcher": "cannot run",
                "lock": "cannot lock files",
                "job": "cannot create a job object",
            }[cause] in str(raised.value)
    finally:
        api.stop()
    assert api.requests == []  # Claude Code never ran
    assert len(written) == (2 if cause == "launcher" else 0)  # checked again


@pytest.mark.parametrize(
    "cause",
    [
        pytest.param("launcher", marks=posix_only),
        "lock",
        "job",
    ],
)
async def test_real_engine_best_effort_cleanup_runs_the_step_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cause: str
) -> None:
    """``engine_cleanup="best_effort"`` keeps the old behavior: the step runs, and a
    warning says that a dead Worker's engine would finish its turn."""
    if cause == "launcher":
        noexec_launcher(tmp_path, monkeypatch)
    elif cause == "lock":
        monkeypatch.setattr(_runner, "_lock_file", refuse_locks)
    else:
        monkeypatch.setattr(
            _runner,
            "_engines_end_with_worker",
            lambda: {0: _runner._job_problem("cannot put Claude Code in the job")},
        )
    monkeypatch.setattr(_runner, "_warned", set())
    api = start_with_policy(refund_policy)
    (tmp_path / "work").mkdir()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path / "work"),
        env=engine_env(api, str(tmp_path / "cfg")),
        engine_cleanup="best_effort",
    )
    try:
        with pytest.warns(UserWarning, match="Going on anyway, because engine_cleanup"):
            out = await runner.run(refund_step(), 1)
    finally:
        api.stop()
    assert out.deferred is not None and out.deferred.name == "look_up_order"
    assert api.errors == []


async def test_real_engine_an_engine_outside_the_workers_job_is_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows can fail to put an engine in the Worker's job object only after the
    engine started. By default the step then ends the engine and stops, so no engine
    runs that would outlive its Worker. (The failure is simulated here, on every
    system.)"""
    from claude_agent_sdk._internal.transport import subprocess_cli

    started: list[int] = []

    def outside_the_job() -> dict[int, str]:
        failed: dict[int, str] = {}
        for child in subprocess_cli._ACTIVE_CHILDREN:
            pid = getattr(child, "pid", None)
            if isinstance(pid, int) and alive(pid):
                started.append(pid)
                failed[pid] = _runner._job_problem(
                    "cannot put Claude Code in the job object (Windows error 5)"
                )
        return failed

    monkeypatch.setattr(_runner, "_engines_end_with_worker", outside_the_job)
    monkeypatch.setattr(_runner, "_JOB_POLL", True)  # as on Windows
    api = start_with_policy(refund_policy)
    (tmp_path / "work").mkdir()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path / "work"), env=engine_env(api, str(tmp_path / "cfg"))
    )
    try:
        with pytest.raises(ApplicationError) as raised:
            await runner.run(refund_step(), 1)
    finally:
        api.stop()
    assert raised.value.type == _runner.CLEANUP_UNAVAILABLE
    assert "Windows error 5" in str(raised.value)
    assert started, "the engine had started"
    assert api.requests == []  # ended within milliseconds: it asked Claude nothing
    await wait_until(lambda: not [pid for pid in started if alive(pid)], timeout=15)
