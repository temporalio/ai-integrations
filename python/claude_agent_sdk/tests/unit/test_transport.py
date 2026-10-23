"""TemporalTransport's handling of the CLI's stdout, with the test sending it as watch_output does."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    PermissionResultAllow,
    ToolPermissionContext,
)

from temporalio import workflow
from temporalio.claude_agent_sdk import TemporalTransport
from temporalio.claude_agent_sdk._payloads import CliOutput
from temporalio.client import WorkflowHandle, WorkflowQueryFailedError
from temporalio.contrib.workflow_streams import WorkflowTopicHandle
from tests.helpers.memory_sandbox import MemoryHarness


def step(message_id: str, block: dict[str, Any]) -> dict[str, Any]:
    """A model step on stdout."""
    message = {"id": message_id, "model": "claude", "content": [block]}
    return {"type": "assistant", "message": message, "parent_tool_use_id": None}


def flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


ASSISTANT = step(
    "msg_1", {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}
)
TOOL_RESULT = {
    "type": "user",
    "message": {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "hi"}],
    },
    "parent_tool_use_id": None,
}
ANSWER = step("msg_2", {"type": "text", "text": "Done."})
RESULT = {"type": "result", "subtype": "success", "is_error": False, "result": "Done."}


@dataclasses.dataclass
class Stdin:
    """What the transport has for the CLI, and how it launches it."""

    outbox: list[dict[str, Any]]
    """Lines waiting for the CLI's stdin."""
    argv: list[str]
    """The flags of the CLI, with its config dir in ``/cfg``."""
    env: dict[str, str]


@workflow.defn
class TransportWorkflow:
    """Writes ``lines``, and reads what reaches the Claude Agent SDK up to a result.

    Its options have ``settings`` and ``env``. The transport is never
    connected: the test sends the CLI's stdout itself.
    """

    def __init__(self) -> None:
        self.transport: TemporalTransport | None = None

    @workflow.run
    async def run(
        self,
        lines: list[dict[str, Any]],
        settings: str | None = None,
        env: dict[str, str] | None = None,
    ) -> list[Any]:
        options = ClaudeAgentOptions(
            allowed_tools=["Read"],
            can_use_tool=self.allow,
            enable_file_checkpointing=True,
            settings=settings,
            env=env or {},
        )
        transport = TemporalTransport(options, sandbox="none")
        self.transport = transport
        await transport.write("".join(json.dumps(line) + "\n" for line in lines))
        messages: list[Any] = []
        async for message in transport.read_messages():
            messages.append(message)
            if message["type"] == "result":
                break
        return messages

    async def allow(
        self, _tool: str, _input: dict[str, Any], _context: ToolPermissionContext
    ) -> PermissionResultAllow:
        return PermissionResultAllow()

    @workflow.query
    def stdin(self) -> Stdin:
        assert self.transport is not None
        transport = self.transport
        return Stdin(
            outbox=list(transport._outbox),
            argv=transport._command.args("/cfg"),
            env=transport._command.env(),
        )


class Cli:
    """Sends lines of stdout to the Workflow, each in a Signal of its own, as watch_output does."""

    def __init__(self, handle: WorkflowHandle[Any, Any]) -> None:
        self.handle = handle
        self.lines: list[str] = []

    def add(self, *messages: dict[str, Any]) -> None:
        self.lines += [json.dumps(message) for message in messages]

    async def send(self, first: int, last: int) -> None:
        """Send lines ``first`` up to ``last``."""
        offsets = [0]
        for line in self.lines:
            offsets.append(offsets[-1] + len(line.encode()) + 1)
        for at in range(first, last):
            output = CliOutput(offsets[at], offsets[at + 1], self.lines[at])
            await self.handle.signal("other", output)

    async def deliver(self, *messages: dict[str, Any]) -> None:
        """Send ``messages``, after the lines sent before."""
        first = len(self.lines)
        self.add(*messages)
        await self.send(first, len(self.lines))

    async def stdin(self) -> Stdin:
        return await self.handle.query(TransportWorkflow.stdin)


async def query_once_running(handle: WorkflowHandle[Any, Any], query: Any) -> Any:
    """The query's answer, once the Workflow has made what it asks about."""
    while True:
        with contextlib.suppress(WorkflowQueryFailedError):
            return await handle.query(query)
        await asyncio.sleep(0.05)


@contextlib.asynccontextmanager
async def transport_workflow(
    harness: MemoryHarness, lines: list[Any], *args: Any
) -> AsyncIterator[Cli]:
    """Start a TransportWorkflow that writes ``lines``, and its CLI once it has a transport."""
    async with harness.start(TransportWorkflow, lines, *args) as handle:
        await query_once_running(handle, TransportWorkflow.stdin)
        yield Cli(handle)


async def test_the_cli_gets_the_sdks_flags_and_env_and_keeps_its_session_files(
    harness: MemoryHarness,
) -> None:
    init = {"type": "system", "subtype": "init", "session_id": "sid-1"}
    async with transport_workflow(harness, []) as cli:
        stdin = await cli.stdin()
        await cli.deliver(init, RESULT)
        messages = await cli.handle.result()

    assert messages == [init, RESULT]
    assert json.loads(flag(stdin.argv, "--settings"))["cleanupPeriodDays"] == 36500
    # The transport starts and resumes no session of its own.
    assert not any(a.startswith(("--session-id", "--resume")) for a in stdin.argv)
    # As the Claude Agent SDK's own transport sets it for enable_file_checkpointing.
    assert stdin.env["CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING"] == "true"


async def test_the_options_settings_and_env_are_kept(harness: MemoryHarness) -> None:
    settings = {"permissions": {"deny": ["Bash"]}, "cleanupPeriodDays": 7}
    env = {"MY_SETTING": "kept"}
    async with transport_workflow(harness, [], json.dumps(settings), env) as cli:
        stdin = await cli.stdin()
        await cli.deliver(RESULT)
        await cli.handle.result()

    assert json.loads(flag(stdin.argv, "--settings")) == {
        "permissions": {"deny": ["Bash", "Edit(//cfg/projects/**/*.jsonl)"]},
        "cleanupPeriodDays": 7,
    }
    assert stdin.env["MY_SETTING"] == "kept"


async def test_each_line_reaches_the_sdk_once_in_order(harness: MemoryHarness) -> None:
    """A line comes twice when a watch_output is retried, or ahead of the lines before it when a stale attempt sends it. Only the next one is taken: the retry sends the rest in order."""
    async with transport_workflow(harness, []) as cli:
        cli.add(ASSISTANT, TOOL_RESULT, ANSWER, RESULT)
        await cli.send(2, 4)  # Ahead of the offset: ignored.
        await cli.send(0, 2)
        await cli.send(0, 2)  # Again: ignored.
        await cli.send(2, 4)  # Sent again, in order.
        messages = await cli.handle.result()

    assert messages == [ASSISTANT, TOOL_RESULT, ANSWER, RESULT]


async def test_the_sdks_requests_go_under_ids_the_workflow_picks(
    harness: MemoryHarness,
) -> None:
    """The Claude Agent SDK's IDs are random, so a replayed Workflow would send different ones."""
    interrupt = {
        "type": "control_request",
        "request_id": "sdk_random",
        "request": {"subtype": "interrupt"},
    }
    async with transport_workflow(harness, [interrupt]) as cli:
        [sent] = (await cli.stdin()).outbox
        response = {"subtype": "success", "request_id": sent["request_id"]}
        await cli.deliver({"type": "control_response", "response": response}, RESULT)
        messages = await cli.handle.result()

    assert sent["request_id"] == "temporal_1"
    assert messages[0]["response"]["request_id"] == "sdk_random"


def test_sandbox_settings_need_settings_as_json() -> None:
    options = ClaudeAgentOptions(
        sandbox={"enabled": True}, settings="/etc/claude/settings.json"
    )
    with pytest.raises(ValueError, match="JSON string"):
        TemporalTransport(options, sandbox="none")


def test_partial_messages_need_a_stream() -> None:
    options = ClaudeAgentOptions(include_partial_messages=True)
    with pytest.raises(ValueError, match="stream"):
        TemporalTransport(options, sandbox="none")


def test_a_stream_needs_partial_messages() -> None:
    topic = WorkflowTopicHandle[Any](cast(Any, None), "partial", cast(Any, Any))
    with pytest.raises(ValueError, match="stream"):
        TemporalTransport(ClaudeAgentOptions(), sandbox="none", stream=topic)


async def test_a_line_that_is_not_an_object_with_a_type_is_skipped(
    harness: MemoryHarness,
) -> None:
    async with transport_workflow(harness, []) as cli:
        cli.lines += ["5", "[1]", '"text"', "null", json.dumps({"no": "type"})]
        cli.add(RESULT)
        await cli.send(0, len(cli.lines))
        messages = await cli.handle.result()

    assert messages == [RESULT]
