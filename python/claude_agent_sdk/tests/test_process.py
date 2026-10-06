"""Fail closed on supervisor/lock setup errors and stop whole orphan process trees."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions, PermissionResultAllow, ResultMessage

from temporalio.claude_agent_sdk import _process
from temporalio.claude_agent_sdk._managed import (
    SupervisedClient,
    SupervisedTransport,
    supervised_query,
)
from tests.helpers.fake_messages_api import FakeMessagesAPI, engine_env, history_of
from tests.hybrid.test_workflow import until


def test_missing_launcher_refuses_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(_process, "__file__", str(tmp_path / "missing.py"))
    transport = SupervisedTransport("", ClaudeAgentOptions(), tmp_path / "lock")
    with pytest.raises(RuntimeError, match="supervisor is missing"):
        transport._build_command()


def test_lock_failure_never_starts_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spawned: list[Any] = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda *args, **kwargs: spawned.append(args)
    )
    with _process.workspace_lock(tmp_path / "lock"):
        with pytest.raises(OSError):
            _process.supervise(os.getppid(), tmp_path / "lock", ["engine"])
    assert spawned == []


def test_windows_guard_failure_never_starts_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(worker: int) -> Any:
        del worker
        raise OSError("injected job setup failure")

    # Exercise the mandatory guard on this platform too, before Popen.
    with monkeypatch.context() as scoped:
        scoped.setattr(
            _process,
            "workspace_lock",
            lambda path: nullcontext(),
        )
        scoped.setattr(os, "name", "nt")
        scoped.setattr(_process, "_windows_guard", fail)
        with pytest.raises(OSError, match="job setup failure"):
            _process.supervise(os.getppid(), tmp_path, ["engine"])


@pytest.mark.skipif(os.name == "nt", reason="POSIX descendant inspection")
def test_missing_descendant_inspection_never_starts_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spawned: list[Any] = []

    def fail() -> Any:
        raise OSError("process inspection unavailable")

    monkeypatch.setattr(_process, "_process_table", fail)
    monkeypatch.setattr(
        subprocess, "Popen", lambda *args, **kwargs: spawned.append(args)
    )
    with pytest.raises(OSError, match="inspection unavailable"):
        _process.supervise(os.getppid(), tmp_path / "lock", ["engine"])
    assert spawned == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX process lifetime ownership")
def test_missing_lifetime_ownership_never_starts_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spawned: list[Any] = []

    def fail() -> Any:
        raise OSError("lifetime ownership unavailable")

    monkeypatch.setattr(_process, "_PosixTree", fail)
    monkeypatch.setattr(_process, "_process_table", lambda: {})
    monkeypatch.setattr(
        subprocess, "Popen", lambda *args, **kwargs: spawned.append(args)
    )
    with pytest.raises(OSError, match="lifetime ownership unavailable"):
        _process.supervise(os.getppid(), tmp_path / "lock", ["engine"])
    assert spawned == []


def test_reused_pid_is_not_signalled(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = object.__new__(_process._PosixTree)
    monkeypatch.setattr(tree, "_identity", lambda pid: (222, 0))
    signalled: list[Any] = []
    monkeypatch.setattr(os, "kill", lambda *args: signalled.append(args))
    tree._signal(123, 111, signal.SIGTERM)
    assert signalled == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec gate")
def test_engine_cannot_execute_before_identity_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _process._PosixTree()
    monkeypatch.setattr(tree, "_identity", lambda pid: None)
    marker = tmp_path / "engine-effect"
    try:
        with pytest.raises(OSError, match="supervision was established"):
            tree._start(
                [
                    sys.executable,
                    "-c",
                    "import sys; from pathlib import Path; Path(sys.argv[1]).touch()",
                    str(marker),
                ]
            )
        assert not marker.exists()
    finally:
        tree._close()


@pytest.mark.parametrize(
    "failure",
    [
        "worker-loss",
        "engine-exit",
        "exit-immediately",
        pytest.param(
            "engine-sigkill",
            marks=pytest.mark.skipif(os.name == "nt", reason="POSIX SIGKILL"),
        ),
        pytest.param(
            "double-fork",
            marks=pytest.mark.skipif(
                not sys.platform.startswith("linux"), reason="Linux subreaper boundary"
            ),
        ),
    ],
)
async def test_failure_stops_engine_and_detached_bash_mcp_children(
    tmp_path: Path,
    failure: str,
) -> None:
    probe = Path(__file__).parent / "helpers/process_probe.py"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(probe),
            "worker",
            str(tmp_path),
            str(_process.__file__),
            failure,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    unrelated_root = tmp_path / "unrelated"
    unrelated_root.mkdir()
    unrelated = subprocess.Popen(
        [
            sys.executable,
            str(probe),
            "child",
            str(unrelated_root),
            str(_process.__file__),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pids: list[int] = []
    try:

        async def ready() -> Any:
            assert proc.poll() is None
            log = tmp_path / "pids.jsonl"
            if not log.exists():
                return None
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            return (
                rows
                if len(rows) == 3
                and (
                    failure == "exit-immediately" or (tmp_path / "effects.log").exists()
                )
                else None
            )

        rows = await until(ready)
        pids = [row["pid"] for row in rows]
        if failure == "worker-loss":
            proc.kill()
            proc.wait()
        elif failure == "engine-exit":
            (tmp_path / "exit-engine").touch()
        elif failure == "engine-sigkill":
            os.kill(
                next(row["pid"] for row in rows if row["role"] == "engine"),
                signal.SIGKILL,
            )

        async def stopped() -> bool:
            try:
                with _process.workspace_lock(tmp_path / "workspace.lock"):
                    return True
            except OSError:
                return False

        await until(stopped)
        effects = tmp_path / "effects.log"
        before = effects.read_bytes() if effects.exists() else b""
        await asyncio.sleep(0.3)
        assert (effects.read_bytes() if effects.exists() else b"") == before, (
            tmp_path / "supervisor.stderr"
        ).read_text()
        assert unrelated.poll() is None
        unrelated_effects = unrelated_root / "effects.log"
        unrelated_before = unrelated_effects.read_bytes()
        await asyncio.sleep(0.1)
        assert len(unrelated_effects.read_bytes()) > len(unrelated_before)
        assert (tmp_path / "supervisor.stderr").read_text() == ""
        # The lock can be taken only after the old executor is terminated.
        # A real replacement process must be able to start under that lock.
        marker = tmp_path / "replacement"
        command = [
            sys.executable,
            "-I",
            "-S",
            str(_process.__file__),
            str(os.getpid()),
            str(tmp_path / "workspace.lock"),
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('safe')",
            str(marker),
        ]
        result = await asyncio.to_thread(
            subprocess.run, command, capture_output=True, timeout=5
        )
        assert result.returncode == 0, result.stderr
        assert marker.read_text() == "safe"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        unrelated.terminate()
        unrelated.wait(timeout=5)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("mode", ["client", "query"])
async def test_supervision_preserves_permission_callback(
    tmp_path: Path, mode: str
) -> None:
    invoked: list[str] = []
    victim = tmp_path / "permission-effect"
    victim.write_text("remove only after permission")

    async def allow(
        name: str, arguments: dict[str, Any], context: Any
    ) -> PermissionResultAllow:
        del context
        invoked.append(name)
        return PermissionResultAllow(updated_input=arguments)

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        if history_of(body)[2]:
            return [{"type": "text", "text": "DONE"}]
        return [
            {
                "type": "tool_use",
                "id": "toolu_permission",
                "name": "Bash",
                "input": {"command": f"rm {shlex.quote(str(victim))}"},
            }
        ]

    api = FakeMessagesAPI(policy, primary_tools={"Bash"}).start()
    options = ClaudeAgentOptions(
        cwd=str(tmp_path),
        env=engine_env(api, str(tmp_path / "cfg")),
        tools=["Bash"],
        permission_mode="default",
        can_use_tool=allow,
        setting_sources=[],
    )
    sdk = SupervisedClient(options, tmp_path / "lock")
    try:
        if mode == "client":
            await sdk.connect()
            await sdk.query("work")
            messages = [message async for message in sdk.receive_response()]
        else:
            messages = [
                message
                async for message in supervised_query(prompt="work", options=options)
            ]
        assert invoked == ["Bash"]
        assert not victim.exists()
        assert any(
            isinstance(message, ResultMessage) and message.result == "DONE"
            for message in messages
        )
        assert api.errors == []
    finally:
        await sdk.disconnect()
        api.stop()
