"""The Docker sandbox, against a stand-in for its sbx CLI."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from samples.docker_sandbox import DockerSandbox
from temporalio.claude_agent_sdk import SandboxError

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the fake sbx is a sh script"
)


@pytest.fixture
def fake_sbx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Puts an sbx that runs a script first on the PATH."""

    def fake(script: str) -> None:
        (tmp_path / "sbx").write_text(f"#!/bin/sh\n{script}\n")
        (tmp_path / "sbx").chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    return fake


async def test_a_new_sandbox_gets_its_env_on_stdin_not_on_the_command(
    tmp_path: Path, fake_sbx: Callable[[str], None]
) -> None:
    fake_sbx(
        f'echo "$@" >> "{tmp_path}/calls"\n'
        f'case $1 in ls) echo "[]" ;; *) cat >> "{tmp_path}/stdin" ;; esac'
    )
    sandbox = DockerSandbox("run", env={"ANTHROPIC_API_KEY": "s'ecret"})

    await sandbox.start(env={"A": "1"})

    assert "ecret" not in (tmp_path / "calls").read_text()
    stdin = (tmp_path / "stdin").read_text()
    assert stdin == "export A=1\nexport ANTHROPIC_API_KEY='s'\"'\"'ecret'\n"


async def test_a_failed_command_in_a_gone_sandbox_raises(
    fake_sbx: Callable[[str], None],
) -> None:
    ls = (
        '[{"name": "claude-agent-up", "status": "running"},'
        ' {"name": "claude-agent-down", "status": "stopped"}]'
    )
    fake_sbx(f"case $1 in ls) echo '{ls}' ;; exec) exit 1 ;; esac")

    assert (await DockerSandbox("up").exec(["false"])).exit_code == 1
    # A stopped sandbox is gone too: the CLI stopped with it.
    for gone in ("down", "gone"):
        with pytest.raises(SandboxError, match="not running"):
            await DockerSandbox(gone).exec(["false"])


async def test_a_sandbox_that_cannot_be_stopped_raises(
    fake_sbx: Callable[[str], None],
) -> None:
    ls = '[{"name": "claude-agent-up", "status": "running"}]'
    fake_sbx(f"case $1 in ls) echo '{ls}' ;; *) exit 1 ;; esac")

    await DockerSandbox("gone").terminate()
    with pytest.raises(SandboxError):
        await DockerSandbox("up").terminate()


async def test_a_session_keeps_one_sandbox_until_deleted(
    tmp_path: Path, fake_sbx: Callable[[str], None]
) -> None:
    ls = '[{"name": "claude-agent-run", "status": "running"}]'
    fake_sbx(
        f'echo "$@" >> "{tmp_path}/calls"\n'
        f'made="{tmp_path}/made"\n'
        "case $1 in\n"
        f"  ls) if [ -e \"$made\" ]; then echo '{ls}'; else echo '[]'; fi ;;\n"
        '  create) touch "$made" ;;\n'
        '  rm) [ -e "$made" ] && rm "$made" ;;\n'
        "  *) cat >/dev/null ;;\n"
        "esac"
    )
    sandbox = DockerSandbox("run")

    await sandbox.start(env={})
    await sandbox.start(env={})
    await sandbox.delete_session()
    await sandbox.delete_session()

    calls = [
        call
        for call in (tmp_path / "calls").read_text().splitlines()
        if not call.startswith(("ls ", "exec "))
    ]
    assert calls == [
        # Created with no folder on this machine: its files are the sandbox's own.
        "create --name claude-agent-run --no-progress claude",
        # Any host, on every start: adding the rule again does nothing.
        "policy allow network --sandbox claude-agent-run **",
        "run --detach --name claude-agent-run",
        # Started again, after stopping what the earlier start left running.
        "stop claude-agent-run",
        "policy allow network --sandbox claude-agent-run **",
        "run --detach --name claude-agent-run",
        "rm --force claude-agent-run",
        "rm --force claude-agent-run",
    ]


async def test_sbx_runs_through_the_given_command(tmp_path: Path) -> None:
    wrapper = tmp_path / "wrapper"
    wrapper.write_text(f'#!/bin/sh\necho "$@" >> "{tmp_path}/calls"\n')
    wrapper.chmod(0o755)
    sandbox = DockerSandbox("box", sbx_command=[str(wrapper), "remote-sbx"])

    await sandbox.exec(["true"])
    async for _ in sandbox.stream(["true"]):
        pass

    calls = (tmp_path / "calls").read_text().splitlines()
    assert [call.split()[:2] for call in calls] == [["remote-sbx", "exec"]] * 2
