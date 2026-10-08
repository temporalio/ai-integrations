"""Real-engine coverage of original-ID native child Activity delivery."""

from __future__ import annotations

import asyncio
import dataclasses
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from mcp.server import Server
from mcp.types import CallToolResult, ListToolsResult, Tool, ToolAnnotations

from temporalio.claude_agent_sdk import ClaudeAgentPlugin
from temporalio.claude_agent_sdk._models import DeferredCall, NativeRequest, ToolOutcome
from temporalio.claude_agent_sdk._native import NativeBridge
from temporalio.client import (
    Client,
    WorkflowFailureError,
    WorkflowUpdateFailedError,
    WorkflowUpdateStage,
)
from temporalio.worker import Replayer, Worker
from tests.helpers.fake_messages_api import FakeMessagesAPI
from tests.subagents.workflows import NativeWorkflow, child_echo
from tests.test_engine_tools import activity_types, make_runner

pytestmark = pytest.mark.timeout(240)


def native_api(
    tool: str, arguments: dict[str, Any], *, background: bool = False
) -> FakeMessagesAPI:
    """Ask for a real child and require its original recorded result."""
    requests = 0

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        nonlocal requests
        requests += 1
        uses = [
            b
            for m in body.get("messages", [])
            for b in (m.get("content") if isinstance(m.get("content"), list) else [])
            if b.get("type") == "tool_use"
        ]
        if requests == 1:
            return [
                {
                    "type": "tool_use",
                    "id": "toolu_native_parent",
                    "name": "Agent",
                    "input": {
                        "description": "native child",
                        "prompt": "Execute the requested child tool",
                        "subagent_type": "general-purpose",
                        "run_in_background": background,
                    },
                }
            ]
        if any(b["name"] == "Agent" for b in uses):
            return [{"type": "text", "text": "parent done"}]
        if not uses:
            return [
                {
                    "type": "tool_use",
                    "id": "toolu_native_child",
                    "name": tool,
                    "input": arguments,
                }
            ]
        return [{"type": "text", "text": "child done"}]

    api = FakeMessagesAPI(decide)
    api.start()
    return api


@pytest.mark.parametrize("tool", ["mcp__durable__child_echo", "Bash"])
@pytest.mark.parametrize("mode", ["held", "store"])
@pytest.mark.parametrize("background", [False, True])
async def test_native_child_runs_as_activity(
    client: Client, tmp_path: Path, tool: str, mode: str, background: bool
) -> None:
    effects = tmp_path / "effect.txt"
    arguments: dict[str, Any] = (
        {"n": 7}
        if tool.startswith("mcp__")
        else {
            "command": f"echo once >> '{effects.as_posix()}' && printf 'recorded shell output'"
        }
    )
    api = native_api(tool, arguments, background=background)
    runner = make_runner(tmp_path, api, mode)
    queue = f"native-{uuid.uuid4().hex[:8]}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[child_echo],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
        ):
            result = await client.execute_workflow(
                NativeWorkflow.run, {}, id=queue, task_queue=queue
            )
            kinds = await activity_types(client.get_workflow_handle(queue))
            history = await client.get_workflow_handle(queue).fetch_history()
        await Replayer(
            workflows=[NativeWorkflow], plugins=[ClaudeAgentPlugin(runner)]
        ).replay_workflow(history)
    finally:
        api.stop()
    assert result["result"] == "parent done"
    assert len(result["calls"]) == 1
    assert result["calls"][0]["id"] == "toolu_native_child"
    assert result["calls"][0]["child"]
    assert (
        "child_echo" if tool.startswith("mcp__") else "run_claude_tool_step",
        "tool-toolu_native_child",
    ) in kinds
    if tool == "Bash":
        assert effects.read_text().splitlines() == ["once"]
    assert result["state"]["native_calls"]["toolu_native_child"]["delivered"]
    assert result["state"]["child_subpaths"]
    assert bool(result["state"]["child_conversations"]) == (mode == "held")
    assert not api.errors


@pytest.mark.parametrize("error", [False, True])
async def test_external_native_mcp_contract(
    client: Client, tmp_path: Path, error: bool
) -> None:
    executions: list[dict[str, Any]] = []
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
        "additionalProperties": False,
    }
    spec = Tool(
        name="external",
        description="external native tool",
        input_schema=schema,
        annotations=ToolAnnotations.model_validate(
            {
                "readOnlyHint": False,
                "destructiveHint": True,
                "idempotentHint": False,
                "openWorldHint": True,
            }
        ),
    )

    async def listing(ctx: Any, params: Any) -> ListToolsResult:
        del ctx, params
        return ListToolsResult(tools=[spec])

    async def calling(ctx: Any, params: Any) -> CallToolResult:
        del ctx
        executions.append(params.arguments)
        return CallToolResult.model_validate(
            {
                "content": [
                    {
                        "type": "text",
                        "text": "external recorded error"
                        if error
                        else "external recorded result",
                    }
                ],
                "isError": error,
            }
        )

    server = Server("external", on_list_tools=listing, on_call_tool=calling)
    api = native_api("mcp__remote__external", {"n": 9})
    runner = make_runner(
        tmp_path,
        api,
        extra_options={
            "mcp_servers": {
                "remote": {"type": "sdk", "name": "remote", "instance": server}
            }
        },
    )
    queue = f"native-mcp-{uuid.uuid4().hex[:8]}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[child_echo],
            plugins=[ClaudeAgentPlugin(runner)],
        ):
            result = await client.execute_workflow(
                NativeWorkflow.run, {}, id=queue, task_queue=queue
            )
    finally:
        api.stop()
    assert executions == [{"n": 9}]
    assert (
        result["state"]["native_calls"]["toolu_native_child"]["outcome"]["is_error"]
        == error
    )
    expected = "external recorded error" if error else "external recorded result"
    assert expected in str(
        result["state"]["native_calls"]["toolu_native_child"]["outcome"]
    )
    offered = [
        tool
        for body in api.requests
        for tool in body.get("tools", [])
        if tool["name"] == "mcp__remote__external"
    ]
    assert offered and offered[0]["input_schema"] == schema
    assert not api.errors


@pytest.mark.parametrize("tool", ["mcp__durable__child_echo", "Bash"])
async def test_native_child_error_delivery(
    client: Client, tmp_path: Path, tool: str
) -> None:
    api = native_api(
        tool,
        {"error": True}
        if tool.startswith("mcp__")
        else {"command": "printf 'recorded child error' >&2; exit 7"},
    )
    runner = make_runner(tmp_path, api)
    queue = f"native-error-{uuid.uuid4().hex[:8]}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[child_echo],
            plugins=[ClaudeAgentPlugin(runner)],
        ):
            result = await client.execute_workflow(
                NativeWorkflow.run, {}, id=queue, task_queue=queue
            )
    finally:
        api.stop()
    recorded = result["state"]["native_calls"]["toolu_native_child"]
    assert recorded["outcome"]["is_error"] and recorded["delivered"]
    outputs = [
        b
        for body in api.requests
        for m in body.get("messages", [])
        for b in (m.get("content") if isinstance(m.get("content"), list) else [])
        if b.get("type") == "tool_result"
        and b.get("tool_use_id") == "toolu_native_child"
    ]
    assert outputs and all(b.get("is_error") for b in outputs)
    assert "recorded child error" in str(outputs[0]["content"])
    assert not api.errors


@pytest.mark.parametrize("tool", ["mcp__durable__child_echo", "Bash"])
@pytest.mark.parametrize("approved", [True, False])
async def test_native_child_approval(
    client: Client, tmp_path: Path, tool: str, approved: bool
) -> None:
    effects = tmp_path / "approved.txt"
    arguments: dict[str, Any] = (
        {"n": 8}
        if tool.startswith("mcp__")
        else {"command": f"echo once >> '{effects.as_posix()}' && printf 'approved'"}
    )
    api = native_api(tool, arguments)
    runner = make_runner(tmp_path, api)
    queue = f"native-approval-{uuid.uuid4().hex[:8]}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[child_echo],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
        ):
            handle = await client.start_workflow(
                NativeWorkflow.run, {"approval": True}, id=queue, task_queue=queue
            )
            pending: list[dict[str, Any]] = []
            for _ in range(600):
                pending = await handle.query(NativeWorkflow.approvals)
                if pending:
                    break
                await asyncio.sleep(0.05)
            assert pending[0]["id"] == "toolu_native_child"
            assert pending[0]["input"] == arguments
            assert pending[0]["child"]
            assert not effects.exists()
            await handle.signal(
                NativeWorkflow.decide, args=[pending[0]["id"], approved]
            )
            result = await handle.result()
    finally:
        api.stop()
    assert result["calls"][0]["status"] == ("done" if approved else "rejected")
    if tool == "Bash":
        assert effects.exists() == approved
    assert not api.errors


async def test_separate_tool_queue_with_one_segment_slot(
    client: Client, tmp_path: Path
) -> None:
    api = native_api("Bash", {"command": "printf 'separate queue result'"})
    runner = make_runner(tmp_path, api)
    queue = f"native-queues-{uuid.uuid4().hex[:8]}"
    try:
        async with (
            Worker(
                client,
                task_queue=queue,
                workflows=[NativeWorkflow],
                plugins=[ClaudeAgentPlugin(runner)],
                max_concurrent_activities=1,
            ),
            Worker(
                client,
                task_queue=f"{queue}-tools",
                activities=[child_echo],
                plugins=[ClaudeAgentPlugin(runner)],
                max_concurrent_activities=1,
            ),
        ):
            result = await client.execute_workflow(
                NativeWorkflow.run,
                {"tool_queue": f"{queue}-tools"},
                id=queue,
                task_queue=queue,
            )
    finally:
        api.stop()
    assert result["state"]["native_calls"]["toolu_native_child"]["delivered"]


@pytest.mark.parametrize("mode", ["held", "store"])
@pytest.mark.parametrize("disable_after", [False, True])
async def test_child_state_survives_continue_as_new(
    client: Client, tmp_path: Path, mode: str, disable_after: bool
) -> None:
    api = native_api("mcp__durable__child_echo", {"n": 12})
    runner = make_runner(tmp_path, api, mode)
    queue = f"native-handover-{uuid.uuid4().hex[:8]}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[child_echo],
            plugins=[ClaudeAgentPlugin(runner)],
        ):
            result = await client.execute_workflow(
                NativeWorkflow.run,
                {"again": True, "disable_after": disable_after},
                id=queue,
                task_queue=queue,
            )
    finally:
        api.stop()
    assert result["state"]["runs"] == 2
    assert result["state"]["segments"] == 2
    assert result["state"]["tool_calls"] == 1
    assert result["state"]["child_subpaths"]
    assert result["state"]["child_checkpoints"]
    assert result["state"]["native_calls"]["toolu_native_child"]["delivered"]
    assert not api.errors


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell requires Windows")
@pytest.mark.parametrize("error", [False, True])
async def test_native_powershell_delivery(
    client: Client, tmp_path: Path, error: bool
) -> None:
    effects = tmp_path / "powershell.txt"
    command = (
        f"Add-Content -LiteralPath '{effects.as_posix()}' -Value once; "
        "[Console]::Out.Write('recorded PowerShell output')"
        + ("; exit 7" if error else "")
    )
    api = native_api("PowerShell", {"command": command})
    runner = make_runner(
        tmp_path,
        api,
        extra_options={"env": {"CLAUDE_CODE_USE_POWERSHELL_TOOL": "1"}},
    )
    queue = f"native-powershell-{uuid.uuid4().hex[:8]}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[child_echo],
            plugins=[ClaudeAgentPlugin(runner)],
        ):
            result = await client.execute_workflow(
                NativeWorkflow.run, {}, id=queue, task_queue=queue
            )
    finally:
        api.stop()
    recorded = result["state"]["native_calls"]["toolu_native_child"]
    assert recorded["delivered"] and recorded["outcome"]["is_error"] == error
    assert "recorded PowerShell output" in str(recorded["outcome"])
    assert effects.read_text().splitlines() == ["once"]
    assert not api.errors


async def test_cancelling_child_approval_does_not_execute(
    client: Client, tmp_path: Path
) -> None:
    api = native_api("mcp__durable__child_echo", {"n": 42})
    runner = make_runner(tmp_path, api)
    queue = f"native-cancel-{uuid.uuid4().hex[:8]}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[child_echo],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
        ):
            handle = await client.start_workflow(
                NativeWorkflow.run, {"approval": True}, id=queue, task_queue=queue
            )
            for _ in range(600):
                if await handle.query(NativeWorkflow.approvals):
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("Child did not reach its approval wait")
            await handle.cancel()
            with pytest.raises(WorkflowFailureError):
                await asyncio.wait_for(handle.result(), 20)
            state = await handle.query(NativeWorkflow.state)
            assert state is not None
            recorded = state.native_calls["toolu_native_child"]
            assert not recorded.delivered and recorded.outcome is None
            assert not any(
                name == "child_echo" for name, _ in await activity_types(handle)
            )
    finally:
        api.stop()


async def test_failure_before_acceptance_can_retry(
    client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = NativeBridge.update
    attempts: list[int] = []

    async def fail_once(self: NativeBridge, call: Any = None, child: str = "") -> Any:
        if call is None:
            attempts.append(self.request.attempt)
            if self.request.attempt == 1:
                raise RuntimeError("lost before any child request was accepted")
        return await original(self, call, child)

    monkeypatch.setattr(NativeBridge, "update", fail_once)
    api = native_api("mcp__durable__child_echo", {"n": 13})
    runner = make_runner(tmp_path, api)
    queue = f"native-preloss-{uuid.uuid4().hex[:8]}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[child_echo],
            plugins=[ClaudeAgentPlugin(runner)],
        ):
            result = await client.execute_workflow(
                NativeWorkflow.run, {}, id=queue, task_queue=queue
            )
    finally:
        api.stop()
    assert attempts == [1, 2]
    assert result["state"]["tool_calls"] == 1
    assert not api.errors


async def test_lost_engine_after_activity_completion_fails_closed(
    client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = NativeBridge.execute
    completed, release = asyncio.Event(), asyncio.Event()

    async def held_delivery(self: NativeBridge, tid: str) -> Any:
        result = await original(self, tid)
        completed.set()
        await release.wait()
        return result

    monkeypatch.setattr(NativeBridge, "execute", held_delivery)
    api = native_api("mcp__durable__child_echo", {"n": 14})
    runner = make_runner(tmp_path, api)
    queue = f"native-postloss-{uuid.uuid4().hex[:8]}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[child_echo],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
        ):
            handle = await client.start_workflow(
                NativeWorkflow.run, {"timeout": 20}, id=queue, task_queue=queue
            )
            await asyncio.wait_for(completed.wait(), 30)
            from claude_agent_sdk._internal.transport import subprocess_cli

            children = list(subprocess_cli._ACTIVE_CHILDREN)
            assert children
            try:
                for process in children:
                    if process.returncode is None:
                        process.terminate()
                await asyncio.wait_for(
                    asyncio.gather(*(p.wait() for p in children)), 15
                )
                release.set()  # the engine has exited; this callback cannot deliver
                with pytest.raises(WorkflowFailureError):
                    await asyncio.wait_for(handle.result(), 60)
            finally:
                release.set()
            state = await handle.query(NativeWorkflow.state)
            assert state is not None
            recorded = state.native_calls["toolu_native_child"]
            assert recorded.outcome is not None and recorded.outcome.content == {
                "n": 14
            }
            assert not recorded.delivered
            kinds = await activity_types(handle)
            assert kinds.count(("child_echo", "tool-toolu_native_child")) == 1
            assert (
                len(
                    [
                        r
                        for r in api.requests
                        if any(
                            b.get("type") == "tool_result"
                            and b.get("tool_use_id") == "toolu_native_child"
                            for m in r.get("messages", [])
                            for b in (
                                m.get("content")
                                if isinstance(m.get("content"), list)
                                else []
                            )
                        )
                    ]
                )
                == 0
            )
    finally:
        release.set()
        api.stop()


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("children", [1, 2])
async def test_parent_pause_waits_for_child_deliveries(
    client: Client, tmp_path: Path, background: bool, children: int
) -> None:
    api = native_api("mcp__durable__child_echo", {"n": 21}, background=background)
    decide = api.decide
    child_number = 0

    def parallel(body: dict[str, Any]) -> list[dict[str, Any]]:
        nonlocal child_number
        blocks = decide(body)
        if blocks[0].get("name") == "Agent":
            if children == 2:
                blocks.append({**blocks[0], "id": "toolu_native_second_parent"})
            blocks.append(
                {
                    "type": "tool_use",
                    "id": "toolu_native_parent_echo",
                    "name": "mcp__durable__child_echo",
                    "input": {"n": 22},
                }
            )
        elif blocks[0].get("name") == "mcp__durable__child_echo":
            child_number += 1
            if child_number == 2:
                blocks[0] = {
                    **blocks[0],
                    "id": "toolu_native_second_child",
                    "input": {"n": 23},
                }
        return blocks

    api.decide = parallel
    runner = make_runner(tmp_path, api)
    queue = f"native-parallel-{uuid.uuid4().hex[:8]}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[child_echo],
            plugins=[ClaudeAgentPlugin(runner)],
        ):
            result = await client.execute_workflow(
                NativeWorkflow.run, {}, id=queue, task_queue=queue
            )
    finally:
        api.stop()
    assert result["state"]["tool_calls"] == children + 1
    assert result["state"]["segments"] == 2
    assert result["state"]["native_calls"]["toolu_native_child"]["delivered"]
    expected = {
        "toolu_native_child",
        "toolu_native_parent_echo",
    }
    if children == 2:
        expected.add("toolu_native_second_child")
        assert result["state"]["native_calls"]["toolu_native_second_child"]["delivered"]
    assert {c["id"] for c in result["calls"]} == expected
    assert not api.errors


async def test_native_image_blocks_are_preserved(
    client: Client, tmp_path: Path
) -> None:
    api = native_api("mcp__durable__child_echo", {"image": True})
    runner = make_runner(tmp_path, api)
    queue = f"native-image-{uuid.uuid4().hex[:8]}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[child_echo],
            plugins=[ClaudeAgentPlugin(runner)],
        ):
            result = await client.execute_workflow(
                NativeWorkflow.run, {}, id=queue, task_queue=queue
            )
    finally:
        api.stop()
    assert result["state"]["native_calls"]["toolu_native_child"]["delivered"]
    blocks = result["state"]["native_calls"]["toolu_native_child"]["outcome"]["blocks"]
    assert blocks[1]["source"]["media_type"] == "image/png"


async def test_native_duplicates_share_execution_and_conflicts_fail(
    client: Client, tmp_path: Path
) -> None:
    api = native_api("mcp__durable__child_echo", {"n": 31})
    runner = make_runner(tmp_path, api)
    queue = f"native-duplicates-{uuid.uuid4().hex[:8]}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[child_echo],
            plugins=[ClaudeAgentPlugin(runner)],
        ):
            handle = await client.start_workflow(
                NativeWorkflow.run, {"approval": True}, id=queue, task_queue=queue
            )
            for _ in range(600):
                if await handle.query(NativeWorkflow.approvals):
                    break
                await asyncio.sleep(0.05)
            request: NativeRequest | None = None
            async for event in handle.fetch_history_events():
                if event.HasField(
                    "workflow_execution_update_accepted_event_attributes"
                ):
                    attributes = (
                        event.workflow_execution_update_accepted_event_attributes
                    )
                    if (
                        attributes.accepted_request.input.name
                        == "__claude_agent_native_tool"
                    ):
                        values = await client.data_converter.decode(
                            attributes.accepted_request.input.args.payloads,
                            [NativeRequest],
                        )
                        if values[0].call is not None:
                            request = values[0]
            assert request is not None and request.call is not None
            duplicate = await handle.start_update(
                "__claude_agent_native_tool",
                request,
                wait_for_stage=WorkflowUpdateStage.ACCEPTED,
                result_type=ToolOutcome,
            )
            conflict = dataclasses.replace(
                request,
                call=DeferredCall(request.call.id, request.call.name, {"n": 32}),
            )
            with pytest.raises(WorkflowUpdateFailedError):
                await handle.execute_update("__claude_agent_native_tool", conflict)
            with pytest.raises(WorkflowUpdateFailedError):
                await handle.execute_update(
                    "__claude_agent_native_tool",
                    dataclasses.replace(request, attempt=request.attempt - 1),
                )
            await handle.signal(NativeWorkflow.decide, args=[request.call.id, True])
            assert (await duplicate.result()).content == {"n": 31}
            result = await handle.result()
            kinds = await activity_types(handle)
    finally:
        api.stop()
    assert result["state"]["tool_calls"] == 1
    assert kinds.count(("child_echo", "tool-toolu_native_child")) == 1
