"""Record the golden histories in ``tests/histories``.

Run ``python -m tests.record_histories [scenario ...]`` (no names: every scenario).
Each scenario runs with the scripted runner on a local dev server; every run of its
Workflow is saved as JSON, overwriting that scenario's files. ``test_replay.py``
replays them, so a change to the Workflow code that would break Workflows already
running fails a test. Record a new scenario by its name, and keep the other files.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from temporalio.claude_agent_sdk import (
    TOOL_CALL_INTERRUPTED,
    TOOL_CALL_NOT_RUN,
    ClaudeAgentPlugin,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
)
from temporalio.claude_agent_sdk.testing import ScriptedClaude
from temporalio.client import Client, WorkflowHandle
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from tests import DEV_SERVER_DOWNLOAD_VERSION
from tests.endless.activities import ALL as COUNTING
from tests.endless.policy import count_policy
from tests.endless.workflows import ChatWorkflow, LongTaskWorkflow, TaskOptions
from tests.engine_tools.policy import edit_policy, shell_policy
from tests.engine_tools.workflows import ShellOptions, ShellWorkflow
from tests.lifecycle.workflows import TasksWorkflow, UpdateChatWorkflow
from tests.parallel.policy import parallel_policy
from tests.parallel.workflows import ParallelWorkflow
from tests.refund import shop
from tests.refund.activities import ALL as SHOP
from tests.refund.policy import refund_policy
from tests.refund.workflows import MANAGER, RefundAgentWorkflow

HISTORIES = Path(__file__).with_name("histories")
WORKFLOWS = [
    RefundAgentWorkflow,
    ParallelWorkflow,
    LongTaskWorkflow,
    ChatWorkflow,
    ShellWorkflow,
    TasksWorkflow,
    UpdateChatWorkflow,
]
"""Every Workflow the golden histories use (the replay test needs them all)."""

REFUND = "Order A-1001 arrived broken, I want my money back."


class ExplodingRunner:
    """The scripted runner, but a segment with the prompt "explode" fails for good."""

    def __init__(self, inner: ScriptedClaude) -> None:
        """Wrap ``inner``."""
        self.inner = inner

    async def run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        """Run the segment, or fail it for good."""
        if inp.prompt == "explode":
            raise ApplicationError("the engine exploded", non_retryable=True)
        return await self.inner.run(inp, attempt)


async def _first_pending(handle: WorkflowHandle[Any, Any], query: Any) -> str:
    for _ in range(600):
        pending = await handle.query(query)
        if pending:
            return pending[0]["id"]
        await asyncio.sleep(0.05)
    raise TimeoutError("no approval request")


async def refund(client: Client, queue: str) -> str:
    """A refund with a manager's approval: one durable call at a time."""
    handle = await client.start_workflow(
        RefundAgentWorkflow.run, REFUND, id=queue, task_queue=queue
    )
    tool_use_id = await _first_pending(handle, RefundAgentWorkflow.pending_approvals)
    await handle.execute_update(
        RefundAgentWorkflow.review, args=[tool_use_id, True, MANAGER]
    )
    await handle.result()
    return queue


async def refund_rejected(client: Client, queue: str) -> str:
    """The manager rejects the refund."""
    handle = await client.start_workflow(
        RefundAgentWorkflow.run, REFUND, id=queue, task_queue=queue
    )
    tool_use_id = await _first_pending(handle, RefundAgentWorkflow.pending_approvals)
    await handle.execute_update(
        RefundAgentWorkflow.review, args=[tool_use_id, False, MANAGER]
    )
    await handle.result()
    return queue


async def refund_unknown_order(client: Client, queue: str) -> str:
    """The order does not exist: the tool fails, and Claude answers."""
    await client.execute_workflow(
        RefundAgentWorkflow.run,
        "Order Z-9999 arrived broken, I want my money back.",
        id=queue,
        task_queue=queue,
    )
    return queue


async def cancel_during_tool(client: Client, queue: str) -> str:
    """The Workflow is cancelled while the approved refund runs."""
    os.environ["REFUND_DELAY"] = "3"
    try:
        handle = await client.start_workflow(
            RefundAgentWorkflow.run, REFUND, id=queue, task_queue=queue
        )
        tool_use_id = await _first_pending(
            handle, RefundAgentWorkflow.pending_approvals
        )
        await handle.execute_update(
            RefundAgentWorkflow.review, args=[tool_use_id, True, MANAGER]
        )
        while not shop.read("refunds.jsonl"):
            await asyncio.sleep(0.05)
        await handle.cancel()
        try:
            await handle.result()
        except Exception:  # cancelled
            pass
    finally:
        os.environ.pop("REFUND_DELAY", None)
    return queue


async def cancel_waiting_for_approval(client: Client, queue: str) -> str:
    """The Workflow is cancelled while a call waits for approval."""
    handle = await client.start_workflow(
        RefundAgentWorkflow.run, REFUND, id=queue, task_queue=queue
    )
    await _first_pending(handle, RefundAgentWorkflow.pending_approvals)
    await handle.cancel()
    try:
        await handle.result()
    except Exception:  # cancelled
        pass
    return queue


async def live_continue_as_new(client: Client, queue: str) -> str:
    """Live output across two Continue-As-New."""
    await client.execute_workflow(
        LongTaskWorkflow.run,
        args=["count to 8", TaskOptions(after_events=40, live=True), None],
        id=queue,
        task_queue=queue,
    )
    return queue


async def failed_segment(client: Client, queue: str) -> str:
    """A segment fails for good, then a task stops at max_segments, then one ends."""
    await client.execute_workflow(
        TasksWorkflow.run,
        ["count to 5", "explode", "count to 1"],
        id=queue,
        task_queue=queue,
    )
    return queue


async def update_chat(client: Client, queue: str) -> str:
    """A chat over Updates that continues as new between messages."""
    await client.start_workflow(
        UpdateChatWorkflow.run, args=[None, False], id=queue, task_queue=queue
    )
    latest = client.get_workflow_handle(queue)
    for text in ("count to 2", "count to 4", "count to 6", "count to 8"):
        await latest.execute_update(UpdateChatWorkflow.ask, text)
    await latest.terminate()
    return queue


async def signal_approval(client: Client, queue: str) -> str:
    """Approval by Signal, after a Signal for a call that does not exist."""
    handle = await client.start_workflow(
        LongTaskWorkflow.run,
        args=["count to 2 and publish", TaskOptions(), None],
        id=queue,
        task_queue=queue,
    )
    tool_use_id = await _first_pending(handle, LongTaskWorkflow.pending_approvals)
    await handle.signal(LongTaskWorkflow.review_by_signal, args=["nope", True])
    await handle.signal(LongTaskWorkflow.review_by_signal, args=[tool_use_id, True])
    await handle.result()
    return queue


async def chat_without_tools(client: Client, queue: str) -> str:
    """A chat whose messages call no tool, continuing as new between messages."""
    handle = await client.start_workflow(
        ChatWorkflow.run, 12, id=queue, task_queue=queue
    )
    for _ in range(12):
        await handle.signal(ChatWorkflow.send, "count to 0")
    await handle.result()
    return queue


async def parallel(client: Client, queue: str) -> str:
    """Three calls in one message, one of them waiting for approval; live output."""
    handle = await client.start_workflow(
        ParallelWorkflow.run,
        args=[["parallel 2 and publish"], None],
        id=queue,
        task_queue=queue,
    )
    tool_use_id = await _first_pending(handle, ParallelWorkflow.pending_approvals)
    await handle.execute_update(ParallelWorkflow.review, args=[tool_use_id, True])
    await handle.result()
    return queue


async def continue_as_new(client: Client, queue: str) -> str:
    """A long task that continues as new twice, carrying its conversation."""
    handle = await client.start_workflow(
        LongTaskWorkflow.run,
        args=["count to 6", TaskOptions(after_events=24), None],
        id=queue,
        task_queue=queue,
    )
    await handle.result()
    return queue


async def chat(client: Client, queue: str) -> str:
    """A chat: tasks on one session, with Continue-As-New between messages."""
    handle = await client.start_workflow(
        ChatWorkflow.run, 3, id=queue, task_queue=queue
    )
    for text in ("count to 2", "count to 4", "count to 6"):
        await handle.signal(ChatWorkflow.send, text)
    await handle.result()
    return queue


async def bash(client: Client, queue: str) -> str:
    """A Claude Code Bash call that waits for approval, then runs as its own Activity."""
    handle = await client.start_workflow(
        ShellWorkflow.run,
        args=["run: make test", ShellOptions(tool_approvals=["Bash"])],
        id=queue,
        task_queue=queue,
    )
    tool_use_id = await _first_pending(handle, ShellWorkflow.pending_approvals)
    await handle.execute_update(ShellWorkflow.review, args=[tool_use_id, True])
    await handle.result()
    return queue


async def bash_retried(client: Client, queue: str) -> str:
    """A Bash call whose first step fails before the call could start (it is tried
    again, as a new Activity) and whose second step fails after it (Claude learns
    the call may have run)."""
    handle = await client.start_workflow(
        ShellWorkflow.run,
        args=["run: make test", ShellOptions()],
        id=queue,
        task_queue=queue,
    )
    await handle.result()
    return queue


async def file_steps(client: Client, queue: str) -> str:
    """An Edit (with a durable call in the same message) and a Write as their own
    Activities, each returning Claude Code's own record of its result
    (``ToolOutcome.entry``)."""
    options = ShellOptions(
        builtin_tools=["Edit", "Write"], tool_activities=["Edit", "Write"]
    )
    handle = await client.start_workflow(
        ShellWorkflow.run,
        args=["edit: /work/notes.txt", options],
        id=queue,
        task_queue=queue,
    )
    await handle.result()
    return queue


def pretend_bash() -> Callable[[dict[str, Any]], str]:
    """Bash in tool steps: it prints what it would have run."""
    return lambda args: f"pretend output of {args['command']}"


def pretend_file_tools() -> dict[str, Callable[[dict[str, Any]], Any]]:
    """Edit and Write in tool steps: their result, with a record as Claude Code
    writes it (without the result, which the outcome carries)."""
    made: list[str] = []

    def record() -> dict[str, Any]:
        made.append(f"00000000-0000-4000-8000-{len(made) + 1:012d}")
        return {
            "type": "user",
            "uuid": made[-1],
            "sessionId": "pretend",
            "message": {"role": "user", "content": []},
            "sourceToolAssistantUUID": "pretend",
        }

    def edit(args: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(
            content=f"The file {args['file_path']} has been updated.", entry=record()
        )

    def write(args: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(
            content=f"The file {args['file_path']} has been written.", entry=record()
        )

    return {"Edit": edit, "Write": write}


def bash_that_fails_twice() -> Callable[[dict[str, Any]], str]:
    """Bash in tool steps: not run the first time, interrupted the second."""
    tries: list[str] = []

    def bash(args: dict[str, Any]) -> str:
        tries.append(args["command"])
        if len(tries) == 1:
            raise ApplicationError("not here", type=TOOL_CALL_NOT_RUN)
        raise ApplicationError("the engine went away", type=TOOL_CALL_INTERRUPTED)

    return bash


async def failed_task(client: Client, queue: str) -> str:
    """A task stopped by max_segments, then a task that answers the waiting call."""
    handle = await client.start_workflow(
        TasksWorkflow.run, ["count to 5", "count to 1"], id=queue, task_queue=queue
    )
    await handle.result()
    return queue


Scenario = Callable[[Client, str], Awaitable[str]]
SCENARIOS: dict[str, tuple[Scenario, Callable[[], Any], str]] = {
    "refund-approval": (refund, lambda: refund_policy, "held"),
    "refund-approval-store": (refund, lambda: refund_policy, "store"),
    "refund-rejected": (refund_rejected, lambda: refund_policy, "held"),
    "refund-unknown-order": (refund_unknown_order, lambda: refund_policy, "held"),
    "cancel-during-tool": (cancel_during_tool, lambda: refund_policy, "held"),
    "cancel-waiting-approval": (
        cancel_waiting_for_approval,
        lambda: refund_policy,
        "held",
    ),
    "parallel-approval": (parallel, lambda: parallel_policy, "held"),
    "continue-as-new": (continue_as_new, lambda: count_policy, "held"),
    "live-continue-as-new": (live_continue_as_new, lambda: count_policy, "held"),
    "chat-continue-as-new": (chat, lambda: count_policy, "held"),
    "chat-no-tools": (chat_without_tools, lambda: count_policy, "held"),
    "update-chat": (update_chat, lambda: count_policy, "held"),
    "signal-approval": (signal_approval, lambda: count_policy, "held"),
    "bash-approval": (bash, lambda: shell_policy, "held"),
    "bash-retried": (bash_retried, lambda: shell_policy, "held"),
    "file-steps": (file_steps, lambda: edit_policy, "held"),
    "failed-task": (failed_task, lambda: count_policy, "held"),
    "failed-segment": (failed_segment, lambda: count_policy, "held"),
}
"""Name: (how to run it, the scripted policy, where the conversation lives)."""

ENGINE_TOOLS: dict[str, Callable[[], dict[str, Callable[[dict[str, Any]], Any]]]] = {
    "bash-retried": lambda: {"Bash": bash_that_fails_twice()},
    "file-steps": pretend_file_tools,
}
"""Scenarios whose tool steps do something else than ``pretend_bash``."""


def scrub(history_json: str) -> str:
    """Drop failures' stack traces: they name local paths, and replay ignores them."""

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: ("" if k == "stackTrace" else clean(v)) for k, v in value.items()
            }
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return json.dumps(clean(json.loads(history_json)), indent=2)


async def record(client: Client, name: str, folder: Path) -> list[Path]:
    """Run one scenario and save every run's history; return the files."""
    scenario, policy, mode = SCENARIOS[name]
    runner: Any = ScriptedClaude(
        policy(),
        folder / name if mode == "store" else None,
        engine_tools=ENGINE_TOOLS.get(name, lambda: {"Bash": pretend_bash()})(),
    )
    if name == "failed-segment":
        runner = ExplodingRunner(runner)
    for old in HISTORIES.glob(f"{name}.json"):
        old.unlink()
    for old in HISTORIES.glob(f"{name}-run-*.json"):
        old.unlink()
    queue = f"golden-{name}-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=WORKFLOWS,
        activities=[*COUNTING, *SHOP],
        plugins=[ClaudeAgentPlugin(runner, heartbeat_every=1.0)],
    ):
        workflow_id = await scenario(client, queue)
    runs = [
        execution.run_id
        async for execution in client.list_workflows(f'WorkflowId = "{workflow_id}"')
    ]
    runs.reverse()  # oldest first
    files: list[Path] = []
    for number, run_id in enumerate(runs, start=1):
        history = await client.get_workflow_handle(
            workflow_id, run_id=run_id
        ).fetch_history()
        suffix = f"-run-{number}" if len(runs) > 1 else ""
        path = HISTORIES / f"{name}{suffix}.json"
        path.write_text(scrub(history.to_json()), encoding="utf-8")
        files.append(path)
    return files


async def main(names: list[str]) -> None:
    """Record the scenarios named (every one when ``names`` is empty)."""
    unknown = sorted(set(names) - set(SCENARIOS))
    if unknown:
        raise SystemExit(f"Unknown scenarios: {', '.join(unknown)}")
    HISTORIES.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="golden-"))
    env = await WorkflowEnvironment.start_local(
        dev_server_download_version=DEV_SERVER_DOWNLOAD_VERSION
    )
    try:
        for name in names or list(SCENARIOS):
            os.environ["SHOP_DIR"] = str(work / f"shop-{name}")
            for path in await record(env.client, name, work):
                print(f"recorded {path.name}")
    finally:
        await env.shutdown()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
