"""Native context, plan transitions, and provider helpers across durable turns."""

from __future__ import annotations

import asyncio
import io
import json
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ToolAnnotations, create_sdk_mcp_server, tool

from temporalio.claude_agent_sdk import ClaudeAgentPlugin, ClaudeAgentSdkRunner
from temporalio.claude_agent_sdk._replay import ReplayModel
from temporalio.client import Client, WorkflowUpdateFailedError
from temporalio.worker import Worker
from tests.helpers.fake_messages_api import FakeMessagesAPI, engine_env, history_of
from tests.native_batches.activities import record
from tests.native_batches.workflows import NativeWorkflow


async def execute(
    client: Client,
    folder: Path,
    tools: list[str],
    api: FakeMessagesAPI,
    *,
    options: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run a stock-engine native Workflow with a deterministic local provider."""
    runner = ClaudeAgentSdkRunner(
        cwd=str(folder),
        env=env or engine_env(api, str(folder / "cfg")),
        extra_options=options,
    )
    queue = "native-features-" + uuid.uuid4().hex
    async with Worker(
        client,
        task_queue=queue,
        workflows=[NativeWorkflow],
        activities=[record],
        plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
    ):
        return await client.execute_workflow(
            NativeWorkflow.run,
            args=["exercise native tools", tools],
            id=queue,
            task_queue=queue,
        )


async def test_skill_instructions_survive_next_turn(
    client: Client, tmp_path: Path
) -> None:
    """Skill instructions are native extra user context, beyond its tool result."""
    skill = tmp_path / ".claude" / "skills" / "durable-test"
    skill.mkdir(parents=True)
    skill.joinpath("SKILL.md").write_text(
        "---\nname: durable-test\ndescription: Test a durable skill\n---\nRemember skill-sentinel-481 on the next turn.\n"
    )

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, texts, history = history_of(body)
        if not history:
            return [
                {
                    "type": "tool_use",
                    "id": "skill_original",
                    "name": "Skill",
                    "input": {"skill": "durable-test"},
                }
            ]
        assert not history[-1].is_error, history
        assert any("skill-sentinel-481" in text for text in texts), texts
        return [{"type": "text", "text": "skill remembered"}]

    api = FakeMessagesAPI(policy).start()
    try:
        assert (
            await execute(
                client,
                tmp_path,
                ["Skill"],
                api,
                options={"setting_sources": ["project"]},
            )
            == "skill remembered"
        )
    finally:
        api.stop()
    assert api.errors == []


async def test_notebook_and_search_tools_are_native(
    client: Client, tmp_path: Path
) -> None:
    """Notebook changes and file search results survive ordinary durable turns."""
    notebook = tmp_path / "tiny.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "id": "original",
                        "source": ["print('before')"],
                        "metadata": {},
                        "outputs": [],
                        "execution_count": None,
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )
    note = tmp_path / "search.txt"
    note.write_text("needle-481\n")
    calls = [
        {
            "type": "tool_use",
            "id": "glob_original",
            "name": "Glob",
            "input": {"pattern": "*.txt", "path": str(tmp_path)},
        },
        {
            "type": "tool_use",
            "id": "grep_original",
            "name": "Grep",
            "input": {
                "pattern": "needle-481",
                "path": str(note),
                "output_mode": "content",
            },
        },
        {
            "type": "tool_use",
            "id": "notebook_read_original",
            "name": "Read",
            "input": {"file_path": str(notebook)},
        },
        {
            "type": "tool_use",
            "id": "notebook_original",
            "name": "NotebookEdit",
            "input": {
                "notebook_path": str(notebook),
                "cell_id": "original",
                "new_source": "print('after')",
                "cell_type": "code",
                "edit_mode": "replace",
            },
        },
    ]

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if len(history) < len(calls):
            return [calls[len(history)]]
        if any(h.is_error for h in history):
            return [{"type": "text", "text": str(history)}]
        assert "search.txt" in str(history[0].content)
        assert "needle-481" in str(history[1].content)
        return [{"type": "text", "text": "native files done"}]

    api = FakeMessagesAPI(policy).start()
    try:
        assert (
            await execute(
                client, tmp_path, ["Glob", "Grep", "Read", "NotebookEdit"], api
            )
            == "native files done"
        )
    finally:
        api.stop()
    assert "after" in str(json.loads(notebook.read_text())["cells"][0]["source"])
    assert api.errors == []


@pytest.mark.parametrize("approve", [True, False])
async def test_plan_decision_is_validated_and_recorded(
    client: Client, tmp_path: Path, approve: bool
) -> None:
    """Plan approval is a boolean Update before native execution can proceed."""

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if not history:
            return [
                {
                    "type": "tool_use",
                    "id": "enter_plan_original",
                    "name": "EnterPlanMode",
                    "input": {},
                }
            ]
        if len(history) == 1:
            return [
                {
                    "type": "tool_use",
                    "id": "plan_original",
                    "name": "ExitPlanMode",
                    "input": {},
                }
            ]
        assert history[-1].is_error == (not approve), history
        return [{"type": "text", "text": "plan decided"}]

    api = FakeMessagesAPI(policy).start()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path),
        env=engine_env(api, str(tmp_path / "cfg")),
    )
    queue = "native-plan-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeWorkflow],
            activities=[record],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
        ):
            handle = await client.start_workflow(
                NativeWorkflow.run,
                args=["decide the plan", ["EnterPlanMode", "ExitPlanMode"]],
                id=queue,
                task_queue=queue,
            )

            async def pending() -> None:
                while not await handle.query(NativeWorkflow.interactions):
                    await asyncio.sleep(0.05)

            await asyncio.wait_for(pending(), 20)
            with pytest.raises(WorkflowUpdateFailedError):
                await handle.execute_update(
                    NativeWorkflow.respond, args=["plan_original", "yes"]
                )
            assert await handle.execute_update(
                NativeWorkflow.respond, args=["plan_original", approve]
            )
            assert await handle.result() == "plan decided"
    finally:
        api.stop()
    assert api.errors == []


async def test_web_search_uses_the_original_helper_transport(
    client: Client, tmp_path: Path
) -> None:
    """Native WebSearch forwards its real provider request instead of replaying it."""
    helpers: list[dict[str, Any]] = []

    def helper(body: dict[str, Any]) -> list[dict[str, Any]]:
        if not any(
            t.get("type", "").startswith("web_search") for t in body.get("tools", [])
        ):
            return [{"type": "text", "text": "ok"}]
        helpers.append(body)
        return [
            {
                "type": "server_tool_use",
                "id": "srv_search_original",
                "name": "web_search",
                "input": {"query": "durable search"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srv_search_original",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.com/durable",
                        "title": "Durable search sentinel",
                        "encrypted_content": "mock-provider-content",
                        "page_age": "today",
                    }
                ],
            },
            {
                "type": "text",
                "text": "Durable search sentinel https://example.com/durable",
            },
        ]

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if not history:
            return [
                {
                    "type": "tool_use",
                    "id": "search_original",
                    "name": "WebSearch",
                    "input": {"query": "durable search"},
                }
            ]
        return [{"type": "text", "text": str(history[-1].content)}]

    api = FakeMessagesAPI(policy, helper_decide=helper).start()
    try:
        result = await execute(client, tmp_path, ["WebSearch"], api)
        assert "Durable search sentinel" in result, result
        assert len(helpers) == 1
        assert "search_original" in json.dumps(api.requests[-1])
    finally:
        api.stop()
    assert api.errors == []


async def test_mcp_annotations_keep_parallel_reads_and_write_barriers(
    client: Client, tmp_path: Path
) -> None:
    """Read-only MCP calls overlap; a modifying MCP call separates the groups."""
    active: dict[int, set[str]] = {0: set(), 1: set()}
    barriers = [asyncio.Event(), asyncio.Event()]
    finished: set[str] = set()
    changed = False

    @tool(
        "read",
        "Read native state",
        {"id": str, "phase": int},
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def read(args: dict[str, Any]) -> dict[str, Any]:
        phase, tid = args["phase"], args["id"]
        assert changed == bool(phase)
        active[phase].add(tid)
        if len(active[phase]) == 2:
            barriers[phase].set()
        await asyncio.wait_for(barriers[phase].wait(), 5)
        finished.add(tid)
        return {"content": [{"type": "text", "text": tid}]}

    @tool(
        "write",
        "Modify native state",
        {"type": "object", "properties": {}},
        annotations=ToolAnnotations(readOnlyHint=False),
    )
    async def write(args: dict[str, Any]) -> dict[str, Any]:
        del args
        nonlocal changed
        assert finished == {"before1", "before2"}
        changed = True
        return {"content": [{"type": "text", "text": "changed"}]}

    calls = [
        {
            "type": "tool_use",
            "id": tid,
            "name": "mcp__state__read",
            "input": {"id": tid, "phase": phase},
        }
        for phase, tid in [(0, "before1"), (0, "before2"), (1, "after1"), (1, "after2")]
    ]
    calls.insert(
        2,
        {
            "type": "tool_use",
            "id": "write_original",
            "name": "mcp__state__write",
            "input": {},
        },
    )

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        return calls if not history else [{"type": "text", "text": "MCP done"}]

    api = FakeMessagesAPI(policy).start()
    try:
        assert (
            await execute(
                client,
                tmp_path,
                [],
                api,
                options={
                    "mcp_servers": {
                        "state": create_sdk_mcp_server("state", tools=[read, write])
                    }
                },
            )
            == "MCP done"
        )
        assert finished == {"before1", "before2", "after1", "after2"}
    finally:
        api.stop()
    assert api.errors == []


async def test_web_fetch_keeps_native_http_and_helper_calls(
    client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real native fetch reads a mocked HTTPS page and invokes its provider."""
    fetched: list[str] = []
    helpers: list[dict[str, Any]] = []
    original = ReplayModel._forwarder

    class Response(io.BytesIO):
        status = 200

        def __init__(self, data: bytes, content_type: str) -> None:
            super().__init__(data)
            self.headers = {
                "content-type": content_type,
                "content-length": str(len(data)),
            }

    def forwarder(model: ReplayModel) -> Any:
        delegate = original(model)

        class Forwarder:
            def open(self, request: Any, timeout: int) -> Any:
                parsed = urllib.parse.urlsplit(request.full_url)
                if parsed.hostname == "example.com":
                    fetched.append(request.full_url)
                    return Response(
                        b"<html><body>page-body-sentinel: durable fetch survived.</body></html>",
                        "text/html",
                    )
                if (
                    parsed.hostname in ("127.0.0.1", "localhost")
                    and "/web/domain_info" not in parsed.path
                ):
                    return delegate.open(request, timeout=timeout)
                return Response(
                    b'{"can_fetch":true,"allowed":true}', "application/json"
                )

        return Forwarder()

    monkeypatch.setattr(ReplayModel, "_forwarder", forwarder)

    def helper(body: dict[str, Any]) -> list[dict[str, Any]]:
        if "page-body-sentinel" in json.dumps(body.get("messages", [])):
            helpers.append(body)
            return [{"type": "text", "text": "Fetched page-body-sentinel"}]
        return [{"type": "text", "text": "ok"}]

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if not history:
            return [
                {
                    "type": "tool_use",
                    "id": "fetch_original",
                    "name": "WebFetch",
                    "input": {
                        "url": "https://example.com/durable",
                        "prompt": "Quote the sentinel.",
                    },
                }
            ]
        return [{"type": "text", "text": str(history[-1].content)}]

    api = FakeMessagesAPI(policy, helper_decide=helper).start()
    try:
        result = await execute(client, tmp_path, ["WebFetch"], api)
        assert "Fetched page-body-sentinel" in result, result
        assert fetched and helpers
    finally:
        api.stop()
    assert api.errors == []
