"""The example chat in a real Docker Sandbox: ``CLAUDE_AGENT_SDK_DOCKER_E2E=1 pytest tests/test_docker_sandbox.py``.

Needs ``sbx``, signed in. Each sandbox is allowed to reach this host's scripted
Anthropic API.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence

import pytest

from samples import docker_sandbox as docker
from temporalio.claude_agent_sdk import ExecResult, Sandbox, SandboxContext
from tests.helpers.harness import ANSWERS, Harness, example_chat, say

# The sbx runner, before a test patches it.
sbx = docker.DockerSandbox._sbx

pytestmark = pytest.mark.skipif(
    os.environ.get("CLAUDE_AGENT_SDK_DOCKER_E2E") != "1" or not shutil.which("sbx"),
    reason="set CLAUDE_AGENT_SDK_DOCKER_E2E=1, with sbx installed, to run Docker Sandboxes",
)


def let_sandboxes_reach(monkeypatch: pytest.MonkeyPatch, ports: list[int]) -> None:
    """Allow each new sandbox to reach ``ports`` on this host."""

    async def patched(
        self: docker.DockerSandbox,
        args: Sequence[str],
        input: bytes = b"",
        timeout: float | None = 60.0,
    ) -> ExecResult:
        result = await sbx(self, args, input, timeout)
        if args[0] == "run" and not result.exit_code:
            # Host ports are named localhost:<port>; host.docker.internal rules don't match.
            allow = ",".join(f"localhost:{port}" for port in ports)
            name = args[args.index("--name") + 1]
            policy = ["policy", "allow", "network", "--sandbox", name, allow]
            docker._check(await sbx(self, policy))
        return result

    monkeypatch.setattr(docker.DockerSandbox, "_sbx", patched)


@pytest.mark.timeout(600)
async def test_the_example_chat_runs_in_its_docker_sandbox(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_port = harness.api.server_address[1]
    let_sandboxes_reach(monkeypatch, [api_port])
    harness.answers = ANSWERS
    env = harness.api_env("host.docker.internal")

    def factory(context: SandboxContext) -> Sandbox:
        return docker.DockerSandbox(context.first_execution_run_id, env=env)

    # The sandbox is keyed by the Workflow's first run.
    run_id = ""
    try:
        async with example_chat(harness.client, factory) as handle:
            run_id = handle.result_run_id or ""
            docker_sandbox = docker.DockerSandbox(run_id, env=env)
            assert await say(handle, "write it") == "Wrote it."
            assert await say(handle, "read it") == "Read it."
            cat = ["exec", docker._name(run_id), "cat"]
            cat.append("/home/agent/workspace/notes.txt")
            written = await sbx(docker_sandbox, cat)
            assert written.stdout == b"hi\n"
            assert harness.asked == [*ANSWERS]
    finally:
        # After the chat ends, which stops the sandbox.
        if run_id:
            await docker.DockerSandbox(run_id, env=env).delete_session()
