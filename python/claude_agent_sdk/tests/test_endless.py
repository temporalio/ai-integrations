"""Endless sessions: the agent continues as new before a Workflow's history hits its limit.

These tests use a second dev server with the history limits lowered (the defaults are
51,200 events, with Continue-As-New suggested from 4,096), so the limit is reached in
seconds instead of hours.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from temporalio.claude_agent_sdk import (
    AgentState,
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    FileSessionStore,
    SegmentRunner,
    _workflow,
)
from temporalio.claude_agent_sdk._conversation import entry_text
from temporalio.claude_agent_sdk.testing import ScriptedClaude
from temporalio.client import Client, WorkflowFailureError, WorkflowHandle
from temporalio.converter import DataConverter
from temporalio.exceptions import ApplicationError
from temporalio.service import RPCError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from tests.conftest import LIMIT
from tests.endless.activities import ALL
from tests.endless.policy import count_policy
from tests.endless.workflows import (
    AutoChatWorkflow,
    ChatWorkflow,
    LongTaskWorkflow,
    TaskOptions,
)
from tests.helpers.fake_messages_api import engine_env, start_with_policy
from tests.helpers.workers import FAIL_FAST
from tests.refund import shop

pytestmark = pytest.mark.timeout(240)


def counted() -> list[str]:
    """The step numbers ``count`` really executed, in order."""
    return [e["detail"] for e in shop.executions("count")]


async def run_lengths(client: Client, workflow_id: str) -> list[int]:
    """History length of every run of a Workflow, first to last."""
    lengths: list[int] = []
    async for execution in client.list_workflows(f'WorkflowId = "{workflow_id}"'):
        handle = client.get_workflow_handle(workflow_id, run_id=execution.run_id)
        lengths.append(len((await handle.fetch_history()).events))
    return lengths


async def replay_every_run(client: Client, workflow_id: str) -> None:
    """Replay each run's history: decisions taken from the history replay the same way."""
    replayer = Replayer(
        workflows=[LongTaskWorkflow, ChatWorkflow, AutoChatWorkflow],
        plugins=[ClaudeAgentPlugin(ScriptedClaude(count_policy))],
    )
    async for execution in client.list_workflows(f'WorkflowId = "{workflow_id}"'):
        handle = client.get_workflow_handle(workflow_id, run_id=execution.run_id)
        await replayer.replay_workflow(await handle.fetch_history())


def scripted_worker(
    client: Client,
    queue: str,
    tmp: Path,
    runner: SegmentRunner | None = None,
    mode: str = "held",
) -> Worker:
    """A Worker for the long-running agents (the scripted runner holds the
    conversation in the Workflow, unless ``mode`` is ``store``)."""
    folder = tmp / "fake" if mode == "store" else None
    plugin = ClaudeAgentPlugin(
        runner or ScriptedClaude(count_policy, folder), heartbeat_every=1.0
    )
    return Worker(
        client,
        task_queue=queue,
        workflows=[LongTaskWorkflow, ChatWorkflow, AutoChatWorkflow],
        activities=ALL,
        plugins=[plugin],
        **FAIL_FAST,
    )


# Temporal applies a Workflow's argument types only when the caller passes as many
# arguments as the run method declares, so the tests pass state=None explicitly.


@pytest.mark.usefixtures("shop_dir")
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_endless_agent_runs_past_the_history_limit(
    limited: WorkflowEnvironment, tmp_path: Path, mode: str
) -> None:
    client, queue = limited.client, f"endless-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, tmp_path, mode=mode):
        handle = await client.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 120", TaskOptions(), None],
            id=queue,
            task_queue=queue,
        )
        result = await asyncio.wait_for(handle.result(), 200)
        progress = await handle.query(LongTaskWorkflow.progress)
        lengths = await run_lengths(client, queue)
    assert result == "counted to 120"
    assert counted() == [str(n) for n in range(1, 121)]  # every step ran exactly once
    assert progress == {"runs": len(lengths), "tool_calls": 120}
    assert len(lengths) >= 3, lengths  # it continued as new several times
    assert max(lengths) < LIMIT, lengths  # no run came near the hard limit


@pytest.mark.usefixtures("shop_dir")
async def test_the_same_agent_without_continue_as_new_is_terminated_at_the_limit(
    limited: WorkflowEnvironment, tmp_path: Path
) -> None:
    client, queue = limited.client, f"stock-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, tmp_path):
        handle = await client.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 120", TaskOptions(continue_as_new=False), None],
            id=queue,
            task_queue=queue,
        )
        with pytest.raises(WorkflowFailureError) as err:
            await asyncio.wait_for(handle.result(), 200)
    reason = f"{err.value} {err.value.cause}"
    assert "exceeds limit" in reason or "Terminated" in reason, reason
    assert 0 < len(counted()) < 120  # it stopped part way


@pytest.mark.usefixtures("shop_dir")
async def test_an_agent_that_cannot_continue_as_new_stops_before_the_limit(
    limited: WorkflowEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The conversation outgrows what a new run's input carries (here 2 KB stands in for
    the 2 MB a long conversation outgrows without External Storage). The agent keeps
    going in its run, then fails the task with an error that says what to change,
    before the server ends the Workflow at its limit."""
    monkeypatch.setattr(_workflow, "PAYLOAD_LIMIT_BYTES", 2000)
    monkeypatch.setattr(_workflow, "_HISTORY_EVENTS", LIMIT)  # this server's limit
    monkeypatch.setattr(_workflow, "_ROOM_EVENTS", 50)
    client, queue = limited.client, f"full-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, tmp_path):
        handle = await client.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 120", TaskOptions(), None],
            id=queue,
            task_queue=queue,
        )
        with pytest.raises(WorkflowFailureError) as err:
            await asyncio.wait_for(handle.result(), 200)
        lengths = await run_lengths(client, queue)
        await replay_every_run(client, queue)
    cause = err.value.cause
    assert isinstance(cause, ApplicationError), cause
    assert "close to Temporal's limits" in cause.message
    assert "cannot continue as new" in cause.message
    assert "External Storage" in cause.message
    assert max(lengths) < LIMIT, lengths  # it failed in good order, not terminated
    assert LIMIT - 100 < lengths[0], lengths  # and only near the limit
    assert 0 < len(counted()) < 120


@pytest.mark.usefixtures("shop_dir")
async def test_a_chat_that_cannot_continue_as_new_stops_before_the_limit(
    limited: WorkflowEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turns without tool calls: the agent checks before every step, not only after
    tool calls, so a chat is not terminated by the server either."""
    monkeypatch.setattr(_workflow, "PAYLOAD_LIMIT_BYTES", 2000)
    monkeypatch.setattr(_workflow, "_HISTORY_EVENTS", LIMIT)
    monkeypatch.setattr(_workflow, "_ROOM_EVENTS", 50)
    client, queue = limited.client, f"fullchat-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, tmp_path):
        handle = await client.start_workflow(
            ChatWorkflow.run, 200, id=queue, task_queue=queue
        )
        for _ in range(200):
            try:
                await handle.signal(ChatWorkflow.send, "count to 0")
            except RPCError:  # the run already closed
                break
        with pytest.raises(WorkflowFailureError) as err:
            await asyncio.wait_for(handle.result(), 200)
        lengths = await run_lengths(client, queue)
    cause = err.value.cause
    assert isinstance(cause, ApplicationError), cause
    assert "close to Temporal's limits" in cause.message
    assert max(lengths) < LIMIT, lengths


@pytest.mark.usefixtures("shop_dir")
async def test_an_agent_that_cannot_continue_as_new_stops_before_the_size_limit(
    client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same, for the history's size (here 150 KB stands in for 50 MB)."""
    monkeypatch.setattr(_workflow, "PAYLOAD_LIMIT_BYTES", 2000)
    monkeypatch.setattr(_workflow, "_HISTORY_BYTES", 150 * 1024)
    monkeypatch.setattr(_workflow, "_ROOM_BYTES", 10 * 1024)
    queue = f"fullsize-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, tmp_path):
        handle = await client.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 500", TaskOptions(after_events=30), None],
            id=queue,
            task_queue=queue,
        )
        with pytest.raises(WorkflowFailureError) as err:
            await asyncio.wait_for(handle.result(), 200)
        description = await handle.describe()
    cause = err.value.cause
    assert isinstance(cause, ApplicationError), cause
    assert "close to Temporal's limits" in cause.message and "MB)" in cause.message
    size = description.raw_description.workflow_execution_info.history_size_bytes
    assert size < 150 * 1024, size
    assert 0 < len(counted()) < 500


@pytest.mark.usefixtures("shop_dir")
async def test_an_agent_continues_as_new_when_its_run_is_nearly_full(
    limited: WorkflowEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continue-As-New set for a history longer than the server allows: the agent
    continues as new before the limit anyway, instead of being terminated."""
    monkeypatch.setattr(_workflow, "_HISTORY_EVENTS", LIMIT)
    monkeypatch.setattr(_workflow, "_ROOM_EVENTS", 50)
    client, queue = limited.client, f"nearly-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, tmp_path):
        handle = await client.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 120", TaskOptions(after_events=LIMIT - 1), None],
            id=queue,
            task_queue=queue,
        )
        result = await asyncio.wait_for(handle.result(), 200)
        lengths = await run_lengths(client, queue)
        await replay_every_run(client, queue)
    assert result == "counted to 120"
    assert counted() == [str(n) for n in range(1, 121)]
    assert len(lengths) >= 2 and max(lengths) < LIMIT, lengths


async def wait_for_approval(handle: WorkflowHandle[Any, Any]) -> dict[str, Any]:
    """Wait until the agent asks for an approval."""
    for _ in range(600):
        pending = await handle.query(LongTaskWorkflow.pending_approvals)
        if pending:
            return pending[0]
        await asyncio.sleep(0.1)
    raise TimeoutError("no approval request")


@pytest.mark.usefixtures("shop_dir")
async def test_an_approval_after_continue_as_new_reaches_the_new_run(
    limited: WorkflowEnvironment, tmp_path: Path
) -> None:
    client, queue = limited.client, f"approve-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, tmp_path):
        handle = await client.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 20 and publish", TaskOptions(after_events=40), None],
            id=queue,
            task_queue=queue,
        )
        pending = await wait_for_approval(handle)  # sent to the latest run
        await handle.execute_update(LongTaskWorkflow.review, args=[pending["id"], True])
        result = await asyncio.wait_for(handle.result(), 100)
        progress = await handle.query(LongTaskWorkflow.progress)
    assert result == "counted to 20"
    assert progress["runs"] > 1 and progress["tool_calls"] == 21
    assert len(shop.executions("publish")) == 1


@pytest.mark.usefixtures("shop_dir")
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_chat_keeps_one_session_across_continue_as_new(
    limited: WorkflowEnvironment, tmp_path: Path, mode: str
) -> None:
    client, queue = limited.client, f"chat-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, tmp_path, mode=mode):
        handle = await client.start_workflow(
            ChatWorkflow.run, 3, id=queue, task_queue=queue
        )
        for text in ("count to 5", "count to 10", "count to 15"):
            await handle.signal(ChatWorkflow.send, text)
        replies = await asyncio.wait_for(handle.result(), 100)
        progress = await handle.query(ChatWorkflow.progress)
    # Each answer builds on the earlier turns: the session carried over.
    assert replies == ["counted to 5", "counted to 10", "counted to 15"]
    assert counted() == [str(n) for n in range(1, 16)]
    assert progress["runs"] > 1


@pytest.mark.usefixtures("shop_dir")
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_chat_hands_over_at_the_start_of_a_message(
    client: Client, tmp_path: Path, mode: str
) -> None:
    """Messages without tool calls, and no check between them in the Workflow: the
    agent continues as new before a message's first step, and the new run sends that
    message to Claude, once."""
    seen: list[str] = []

    def answer(prompt: str, history: list[Any]) -> Any:
        seen.append(prompt)
        return count_policy(prompt, history)

    folder = tmp_path / "fake" if mode == "store" else None
    queue = f"autochat-{uuid.uuid4().hex[:8]}"
    messages = [f"message {n}" for n in range(1, 9)]
    runner = ScriptedClaude(answer, folder)
    async with scripted_worker(client, queue, tmp_path, runner=runner):
        handle = await client.start_workflow(
            AutoChatWorkflow.run, 8, id=queue, task_queue=queue
        )
        for text in messages:
            await handle.signal(AutoChatWorkflow.send, text)
        replies = await asyncio.wait_for(handle.result(), 100)
        progress = await handle.query(AutoChatWorkflow.progress)
        await replay_every_run(client, queue)
    assert replies == ["counted to 0"] * 8
    assert seen == messages  # each message reached Claude once, in order
    assert progress["runs"] > 1  # with no tool calls, only at the start of a message


async def test_agent_state_survives_the_data_converter() -> None:
    from temporalio.claude_agent_sdk import ToolOutcome

    state = AgentState(
        session_id="s",
        segment_index=7,
        task_prompt="count to 3",
        pending={"toolu_1": ToolOutcome({"n": 1})},
        recent_call_ids=["toolu_1"],
        tool_calls=1,
        runs=2,
        conversation=[
            entry_text({"uuid": "u1", "type": "user", "message": {"content": "مرحبا"}})
        ],
        external_storage=True,
    )
    converter = DataConverter.default
    payloads = await converter.encode([state])
    (decoded,) = await converter.decode(payloads, [AgentState])
    assert decoded == state
    assert isinstance(decoded.pending["toolu_1"], ToolOutcome)


@pytest.mark.timeout(300)
@pytest.mark.usefixtures("shop_dir")
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_real_engine_session_continues_across_runs(
    limited: WorkflowEnvironment, tmp_path: Path, mode: str
) -> None:
    """The real engine resumes the same Claude session in each new run."""
    api = start_with_policy(count_policy)
    (tmp_path / "work").mkdir()
    runner = ClaudeAgentSdkRunner(
        session_store=(
            FileSessionStore(tmp_path / "sessions") if mode == "store" else None
        ),
        cwd=str(tmp_path / "work"),
        env=engine_env(api, str(tmp_path / "cfg")),
    )
    client, queue = limited.client, f"real-{uuid.uuid4().hex[:8]}"
    try:
        async with scripted_worker(client, queue, tmp_path, runner):
            handle = await client.start_workflow(
                LongTaskWorkflow.run,
                args=["count to 12", TaskOptions(after_events=40), None],
                id=queue,
                task_queue=queue,
            )
            result = await asyncio.wait_for(handle.result(), 280)
            progress = await handle.query(LongTaskWorkflow.progress)
            chat = await client.start_workflow(
                ChatWorkflow.run, 2, id=f"{queue}-chat", task_queue=queue
            )
            for text in ("count to 2", "count to 4"):
                await chat.signal(ChatWorkflow.send, text)
            replies = await asyncio.wait_for(chat.result(), 280)
    finally:
        api.stop()
    assert result == "counted to 12"
    assert progress["runs"] > 1 and progress["tool_calls"] == 12
    assert replies == ["counted to 2", "counted to 4"]  # turn 2 paused and resumed too
    executed = shop.executions("count")
    # 12 steps for the task and 4 for the chat, each tool call exactly once.
    assert sorted(int(e["detail"]) for e in executed) == sorted(
        [*range(1, 13), *range(1, 5)]
    )
    keys = Counter(e["key"] for e in executed)
    assert all(times == 1 for times in keys.values()), keys
    assert api.errors == []
    assert runner.stub_calls == 0
