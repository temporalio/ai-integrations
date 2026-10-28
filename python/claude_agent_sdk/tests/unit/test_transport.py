"""TemporalTransport in a Workflow, with a shell script as the CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from temporalio import workflow
from temporalio.claude_agent_sdk import TemporalTransport
from temporalio.client import WorkflowFailureError
from temporalio.contrib.workflow_streams import WorkflowTopicHandle
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from tests.helpers.harness import CLAUDE, LocalCli
from tests.helpers.workflows import start_workflow

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not CLAUDE.is_file(),
    reason="needs sh and the Claude Agent SDK's bundled Claude Code",
)

RESULT = {"type": "result"}


@workflow.defn
class TransportWorkflow:
    """Writes ``lines`` to the CLI at ``cli_path``, and reads what reaches the Claude Agent SDK up to a result."""

    @workflow.run
    async def run(
        self,
        cli_path: str,
        lines: list[dict[str, Any]],
        max_buffer_size: int | None = None,
    ) -> list[Any]:
        options = ClaudeAgentOptions(cli_path=cli_path, max_buffer_size=max_buffer_size)
        transport = TemporalTransport(options)
        await transport.connect()
        for line in lines:
            await transport.write(json.dumps(line) + "\n")
        messages: list[Any] = []
        async for message in transport.read_messages():
            messages.append(message)
            if message["type"] == "result":
                break
        await transport.close()
        return messages


@pytest.fixture
def cli(tmp_path: Path) -> LocalCli:
    return LocalCli(tmp_path)


async def run_transport(
    env: WorkflowEnvironment,
    cli: LocalCli,
    script: str,
    lines: list[Any],
    max_buffer_size: int | None = None,
) -> list[Any]:
    """Run a TransportWorkflow whose CLI runs ``script``, and return what reached the Claude Agent SDK."""
    cli.set_cli(script)
    async with start_workflow(
        env.client, TransportWorkflow, str(cli.cli_path), lines, max_buffer_size
    ) as handle:
        return await handle.result()


def test_the_cli_gets_the_options_env() -> None:
    options = ClaudeAgentOptions(
        enable_file_checkpointing=True, env={"MY_SETTING": "kept"}
    )

    env = TemporalTransport(options)._connect_input().env

    assert env["MY_SETTING"] == "kept"
    # As the Claude Agent SDK's own transport sets it for enable_file_checkpointing.
    assert env["CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING"] == "true"


@pytest.mark.parametrize("partial", [True, False])
def test_a_stream_goes_with_partial_messages_and_only_them(partial: bool) -> None:
    topic = WorkflowTopicHandle[Any](cast(Any, None), "partial", cast(Any, Any))
    options = ClaudeAgentOptions(include_partial_messages=partial)
    with pytest.raises(ValueError, match="stream"):
        TemporalTransport(options, stream=None if partial else topic)


def test_settings_are_json_not_a_file_path() -> None:
    options = ClaudeAgentOptions(settings="settings.json")
    with pytest.raises(ValueError, match="JSON object"):
        TemporalTransport(options)


async def test_the_sdks_requests_go_under_ids_the_workflow_picks(
    env: WorkflowEnvironment, cli: LocalCli
) -> None:
    """The Claude Agent SDK's IDs are random, so a replayed Workflow would send different ones."""
    interrupt = {
        "type": "control_request",
        "request_id": "sdk_random",
        "request": {"subtype": "interrupt"},
    }
    sent = cli.root / "sent"
    # Answers the request it reads under the ID it was sent.
    answer = (
        f'read line; echo "$line" > "{sent}"; '
        """id=$(echo "$line" | sed 's/.*"request_id": "\\([^"]*\\)".*/\\1/'); """
        """echo '{"type": "control_response", "response": {"subtype": "success", "request_id": "'$id'"}}'; """
        f"echo '{json.dumps(RESULT)}'"
    )

    messages = await run_transport(env, cli, answer, [interrupt])

    assert json.loads(sent.read_text())["request_id"] != "sdk_random"
    assert messages[0]["response"]["request_id"] == "sdk_random"


async def test_after_the_launch_the_activities_run_on_the_workers_own_queue(
    env: WorkflowEnvironment, cli: LocalCli
) -> None:
    """Only the Worker that started the subprocess can reach it."""
    cli.set_cli("exec cat")
    async with start_workflow(
        env.client, TransportWorkflow, str(cli.cli_path), [RESULT]
    ) as handle:
        assert await handle.result() == [RESULT]

    queues: dict[str, str] = {}
    async for event in handle.fetch_history_events():
        scheduled = event.activity_task_scheduled_event_attributes
        if scheduled.activity_type.name:
            queues[scheduled.activity_type.name] = scheduled.task_queue.name
    shared = queues.pop("claude-agent-sdk-connect")
    assert set(queues) == {
        "claude-agent-sdk-write",
        "claude-agent-sdk-read",
        "claude-agent-sdk-close",
    }
    [own] = set(queues.values())
    assert own != shared
    assert own.startswith("claude-agent-sdk-")


async def test_a_line_that_is_not_a_json_object_is_skipped(
    env: WorkflowEnvironment, cli: LocalCli
) -> None:
    """As the Claude Agent SDK skips it, such as ``[SandboxDebug] ...``."""
    script = f"echo '[SandboxDebug] x'; echo '{json.dumps(RESULT)}'"

    assert await run_transport(env, cli, script, []) == [RESULT]


async def test_a_line_over_max_buffer_size_fails_as_in_the_sdk(
    env: WorkflowEnvironment, cli: LocalCli
) -> None:
    script = f"echo '{json.dumps(RESULT)}'"

    with pytest.raises(WorkflowFailureError) as raised:
        await run_transport(env, cli, script, [], max_buffer_size=10)

    error = raised.value.cause
    assert isinstance(error, ApplicationError)
    assert error.type == "CLIJSONDecodeError"
    assert "maximum buffer size of 10 bytes" in error.message
