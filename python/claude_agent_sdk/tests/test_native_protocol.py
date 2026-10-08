"""Native identity verification, immutable rendering, and committed child forks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from claude_agent_sdk import InMemorySessionStore, project_key_for_directory
from mcp.server import Server
from mcp.types import ListResourcesResult, ReadResourceResult, Resource

from temporalio import activity
from temporalio.claude_agent_sdk import (
    ClaudeAgentSdkRunner,
    SegmentInput,
    ToolOutcome,
    _render,
)
from temporalio.claude_agent_sdk._models import ConversationRef
from temporalio.claude_agent_sdk._native import ChildStore, NativeBridge, mcp_connection


@pytest.fixture
async def bridge_and_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[NativeBridge, list[Any]]:
    accepted: list[Any] = []

    async def update(*args: Any, **kwargs: Any) -> ToolOutcome:
        del kwargs
        accepted.append(args)
        return ToolOutcome(content="should never execute")

    handle = SimpleNamespace(execute_update=update)
    monkeypatch.setattr(
        activity,
        "info",
        lambda: SimpleNamespace(workflow_id="wf", workflow_run_id="run"),
    )
    monkeypatch.setattr(
        activity,
        "client",
        lambda: SimpleNamespace(get_workflow_handle=lambda *a, **kw: handle),
    )
    inp = SegmentInput(
        str(uuid.uuid4()),
        "prompt",
        [],
        conversation=ConversationRef("query", "0", 0),
        tool_activities=["Bash"],
    )
    bridge = NativeBridge(
        inp,
        1,
        ChildStore(InMemorySessionStore(), []),
        str(tmp_path),
        inp.session_id,
        str(tmp_path),
    )
    return bridge, accepted


@pytest.mark.parametrize("reason", ["identity", "lock", "launcher"])
async def test_native_hook_refuses_unverified_calls(
    tmp_path: Path,
    bridge_and_updates: tuple[NativeBridge, list[Any]],
    reason: str,
) -> None:
    bridge, accepted = bridge_and_updates
    bridge.children.add("child")
    bridge.protected = reason != "launcher"
    if reason != "lock":
        (tmp_path / "worker.lock").touch()
    bridge.observed["original"] = ("Bash", {"command": "original command"})
    event = {
        "hook_event_name": "PreToolUse",
        "agent_id": "child",
        "tool_use_id": "original",
        "tool_name": "Bash",
        "tool_input": {
            "command": "different command"
            if reason == "identity"
            else "original command"
        },
    }
    decision = await bridge.hook(event)
    assert decision["permissionDecision"] == "deny"
    assert (
        "differs from the assistant" if reason == "identity" else "Unprotected"
    ) in decision["permissionDecisionReason"]
    assert accepted == [] and bridge.receipts == {}


async def test_external_mcp_proxy_preserves_resources_and_capabilities(
    bridge_and_updates: tuple[NativeBridge, list[Any]],
) -> None:
    bridge, accepted = bridge_and_updates

    async def listing(ctx: Any, params: Any) -> ListResourcesResult:
        del ctx, params
        return ListResourcesResult(
            resources=[Resource(name="native resource", uri="test://resource")]
        )

    async def reading(ctx: Any, params: Any) -> ReadResourceResult:
        del ctx
        assert str(params.uri) == "test://resource"
        return ReadResourceResult.model_validate(
            {"contents": [{"uri": "test://resource", "text": "recorded resource"}]}
        )

    server = Server(
        "remote",
        instructions="Original server instructions",
        on_list_resources=listing,
        on_read_resource=reading,
    )
    config = {"type": "sdk", "name": "remote", "instance": server}
    async with mcp_connection(config) as original:
        async with mcp_connection(bridge.proxy("remote", [], original)) as proxied:
            assert proxied.instructions == original.instructions
            assert proxied.server_capabilities.resources is not None
            assert proxied.server_capabilities.prompts is None
            resources = await proxied.list_resources()
            assert resources.resources[0].name == "native resource"
            value = await proxied.read_resource(resources.resources[0].uri)
            assert value.contents[0].text == "recorded resource"
    assert accepted == []


def test_native_renderer_rejects_changed_receipt(tmp_path: Path) -> None:
    file = tmp_path / "outcome.json"
    recorded = json.dumps(
        {"output": {"stdout": "out", "stderr": "err", "exitCode": 7}, "is_error": True}
    ).encode()
    file.write_bytes(recorded)
    argv = [
        sys.executable,
        "-I",
        "-S",
        _render.__file__,
        str(file),
        hashlib.sha256(recorded).hexdigest(),
    ]
    result = subprocess.run(argv, capture_output=True, check=False)
    assert (result.returncode, result.stdout, result.stderr) == (7, b"out", b"err")
    file.write_bytes(recorded.replace(b"out", b"changed"))
    refused = subprocess.run(argv, capture_output=True, check=False)
    assert refused.returncode != 0 and refused.stdout == b""
    assert b"receipt changed" in refused.stderr


async def test_child_fork_copies_only_committed_prefix(tmp_path: Path) -> None:
    store = InMemorySessionStore()
    runner = ClaudeAgentSdkRunner(
        session_store=store,
        cwd=str(tmp_path),
        env={"ANTHROPIC_API_KEY": "test-not-real"},
    )
    session, copied = str(uuid.uuid4()), str(uuid.uuid4())
    path, checkpoint = "subagents/agent-child", str(uuid.uuid4())
    key = {
        "project_key": project_key_for_directory(str(tmp_path)),
        "session_id": session,
        "subpath": path,
    }
    committed = {
        "type": "assistant",
        "isSidechain": True,
        "uuid": checkpoint,
        "sessionId": session,
        "message": {"content": [{"type": "text", "text": "committed child"}]},
    }
    unfinished = {
        **committed,
        "uuid": str(uuid.uuid4()),
        "message": {"content": [{"type": "text", "text": "unfinished attempt"}]},
    }
    await store.append(cast(Any, key), cast(Any, [committed, unfinished]))
    inp = SegmentInput(
        session, None, [], child_subpaths=[path], child_checkpoints={path: checkpoint}
    )
    await runner._copy_children(inp, copied)  # type: ignore[reportPrivateUsage]
    entries = await store.load(cast(Any, {**key, "session_id": copied}))
    assert entries == [{**committed, "sessionId": copied}]
