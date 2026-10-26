"""Fail closed: if a Claude Code engine ever stops honoring "defer", the agent stops loudly.

Anthropic documents the "defer" decision in the Claude Code hooks docs, and the same
page says it is ignored in some cases. These tests simulate an engine that ignores
it, using the real engine and a local fake Messages API, and check that nothing runs
outside Temporal and that the step fails with a clear, non-retryable error instead of
finishing as if all were well.
"""

from __future__ import annotations

import asyncio
import re
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    DeferredCall,
    FileSessionStore,
    SegmentInput,
    SegmentOutput,
    ToolSpec,
    _runner,
)
from temporalio.client import Client, WorkflowFailureError
from temporalio.worker import Worker
from tests.helpers.fake_messages_api import (
    FakeMessagesAPI,
    engine_env,
    start_with_policy,
)
from tests.helpers.workers import FAIL_FAST
from tests.refund import shop
from tests.refund.activities import ALL
from tests.refund.policy import refund_policy
from tests.refund.workflows import ORDER, RefundAgentWorkflow

BROKEN_HOOK = Path(__file__).parent / "helpers" / "broken_engine_hook.py"
BROKEN_TEAPOT = "Order A-1001 arrived broken, I want my money back."
TOOLS = [
    ToolSpec("look_up_order", "Look up an order.", ORDER),
    ToolSpec("issue_refund", "Refund money.", {"type": "object"}),
    ToolSpec("email_customer", "Email the customer.", {"type": "object"}),
]


def failure_text(err: BaseException) -> str:
    """The error and all of its causes, in one string."""
    parts: list[str] = []
    current: BaseException | None = err
    while current is not None:
        parts.append(str(current))
        current = getattr(current, "cause", None) or current.__cause__
    return " | ".join(parts)


@pytest.fixture
def fake_api() -> Iterator[FakeMessagesAPI]:
    """The refund story played through the real engine."""
    api = start_with_policy(refund_policy)
    yield api
    api.stop()


@pytest.fixture
def broken_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the engine's PreToolUse hook behave like an engine that ignores defer."""
    entry = {"type": "command", "command": sys.executable, "args": [str(BROKEN_HOOK)]}
    monkeypatch.setattr(_runner, "_hook_entry", lambda: entry)


def real_runner(
    api: FakeMessagesAPI, tmp_path: Path, mode: str
) -> ClaudeAgentSdkRunner:
    """A runner on the real engine whose hook misbehaves as ``mode`` says."""
    env = {**engine_env(api, str(tmp_path / "cfg")), "BROKEN_ENGINE": mode}
    return ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "store"), cwd=str(tmp_path), env=env
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "allow",
            "ran durable tool(s) look_up_order inside the engine instead of pausing",
        ),
        ("deny", "did not pause"),
    ],
)
@pytest.mark.usefixtures("broken_engine")
async def test_step_fails_closed_when_the_engine_ignores_defer(
    fake_api: FakeMessagesAPI,
    tmp_path: Path,
    mode: str,
    expected: str,
) -> None:
    runner = real_runner(fake_api, tmp_path, mode)
    out = await runner.run(
        SegmentInput(session_id=str(uuid.uuid4()), prompt=BROKEN_TEAPOT, tools=TOOLS), 1
    )
    assert out.is_error and out.deferred is None and out.result is None
    assert out.error is not None and expected in out.error
    assert re.search(r"Claude Code \d+\.\d+\.\d+", out.error), (
        out.error
    )  # names the version
    assert "nothing ran outside Temporal" in out.error


@pytest.mark.timeout(120)
@pytest.mark.usefixtures("broken_engine", "shop_dir")
async def test_workflow_stops_loudly_and_no_tool_activity_runs(
    client: Client,
    fake_api: FakeMessagesAPI,
    tmp_path: Path,
) -> None:
    queue = f"broken-{uuid.uuid4().hex[:6]}"
    plugin = ClaudeAgentPlugin(
        real_runner(fake_api, tmp_path, "allow"), heartbeat_every=1.0
    )
    async with Worker(
        client,
        task_queue=queue,
        workflows=[RefundAgentWorkflow],
        activities=ALL,
        plugins=[plugin],
        **FAIL_FAST,
    ):
        handle = await client.start_workflow(
            RefundAgentWorkflow.run, BROKEN_TEAPOT, id=queue, task_queue=queue
        )
        with pytest.raises(WorkflowFailureError) as err:
            await asyncio.wait_for(handle.result(), 100)
        calls = await handle.query(RefundAgentWorkflow.tool_calls)
    assert "inside the engine instead of pausing" in failure_text(err.value)
    assert calls == []  # the Workflow never ran a tool Activity
    assert shop.executions("look_up_order") == [] and shop.read("refunds.jsonl") == []


class RepeatingRunner:
    """A runner whose engine keeps asking for the same tool call."""

    async def run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        """Always pause at the same call."""
        del attempt
        return SegmentOutput(
            session_id=inp.session_id,
            deferred=DeferredCall(
                id="toolu_same", name="look_up_order", input={"order_id": "A-1001"}
            ),
            checkpoint=f"cp{inp.segment_index}",
        )


@pytest.mark.usefixtures("shop_dir")
async def test_a_tool_call_never_runs_twice(client: Client) -> None:
    queue = f"repeat-{uuid.uuid4().hex[:6]}"
    plugin = ClaudeAgentPlugin(RepeatingRunner(), heartbeat_every=1.0)
    async with Worker(
        client,
        task_queue=queue,
        workflows=[RefundAgentWorkflow],
        activities=ALL,
        plugins=[plugin],
        **FAIL_FAST,
    ):
        handle = await client.start_workflow(
            RefundAgentWorkflow.run, BROKEN_TEAPOT, id=queue, task_queue=queue
        )
        with pytest.raises(WorkflowFailureError) as err:
            await asyncio.wait_for(handle.result(), 60)
    assert "which already ran" in failure_text(err.value)
    assert (
        len(shop.executions("look_up_order")) == 1
    )  # it ran once and was not run again
