"""The durable agent loop in a Workflow, with a scripted stand-in for Claude (no engine).

The conversation lives in the Workflow (the default) unless a test says ``store``.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.claude_agent_sdk import ClaudeAgentPlugin
from temporalio.claude_agent_sdk.testing import ScriptedClaude
from temporalio.client import Client, WorkflowHistory, WorkflowUpdateFailedError
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment
from temporalio.worker import Replayer, Worker
from tests.conftest import wait_for_approval
from tests.helpers.workers import FAIL_FAST
from tests.refund import shop
from tests.refund.activities import ALL, issue_refund
from tests.refund.policy import refund_policy
from tests.refund.workflows import MANAGER, RefundAgentWorkflow

BROKEN_TEAPOT = "Order A-1001 arrived broken, I want my money back."


async def run_case(
    client: Client,
    tmp: Path,
    *,
    approve: bool | None = True,
    prompt: str = BROKEN_TEAPOT,
    mode: str = "held",
) -> tuple[str, list[dict[str, Any]], float, WorkflowHistory]:
    """Run one refund request end to end and return what happened."""
    queue = f"tq-{uuid.uuid4().hex[:8]}"
    folder = tmp / "fake" if mode == "store" else None
    plugin = ClaudeAgentPlugin(
        ScriptedClaude(refund_policy, folder), heartbeat_every=1.0
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
            RefundAgentWorkflow.run,
            prompt,
            id=f"wf-{uuid.uuid4().hex[:8]}",
            task_queue=queue,
        )
        pending = await wait_for_approval(handle)
        if pending is not None and approve is not None:
            await handle.execute_update(
                RefundAgentWorkflow.review, args=[pending["id"], approve, MANAGER]
            )
        result = await asyncio.wait_for(handle.result(), 60)
        calls = await handle.query(RefundAgentWorkflow.tool_calls)
        cost = await handle.query(RefundAgentWorkflow.cost_usd)
        history = await handle.fetch_history()
    return result, calls, cost, history


@pytest.mark.usefixtures("shop_dir")
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_approved_refund_runs_each_tool_once(
    client: Client, tmp_path: Path, mode: str
) -> None:
    result, calls, cost, _ = await run_case(client, tmp_path, mode=mode)
    assert result.startswith("Done. Refunded 49.99 EUR for order A-1001")
    assert [(c["name"], c["status"]) for c in calls] == [
        ("look_up_order", "done"),
        ("issue_refund", "done"),
        ("email_customer", "done"),
    ]
    assert len(shop.read("refunds.jsonl")) == 1 and len(shop.read("emails.jsonl")) == 1
    assert len(shop.executions("issue_refund")) == 1
    assert cost == pytest.approx(0.04)  # 4 model segments


@pytest.mark.usefixtures("shop_dir")
async def test_rejected_refund_moves_no_money(client: Client, tmp_path: Path) -> None:
    result, calls, _, _ = await run_case(client, tmp_path, approve=False)
    assert "not approved" in result
    assert calls[1]["status"] == "rejected"
    assert shop.read("refunds.jsonl") == [] and shop.executions("issue_refund") == []


@pytest.mark.usefixtures("shop_dir")
async def test_flaky_payment_provider_is_retried(
    client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAIL_REFUND_TIMES", "2")
    result, _, _, _ = await run_case(client, tmp_path)
    assert result.startswith("Done.")
    assert (
        len(shop.executions("issue_refund")) == 3
    )  # 2 simulated timeouts, then success
    assert len(shop.read("refunds.jsonl")) == 1


@pytest.mark.usefixtures("shop_dir")
async def test_tool_error_goes_back_to_claude(client: Client, tmp_path: Path) -> None:
    result, calls, _, _ = await run_case(
        client, tmp_path, prompt="Order Z-9999 never arrived."
    )
    assert "could not find that order" in result
    assert calls[0]["status"] == "failed"


@pytest.mark.usefixtures("shop_dir")
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_history_replays_without_nondeterminism(
    client: Client, tmp_path: Path, mode: str
) -> None:
    _, _, _, history = await run_case(client, tmp_path, mode=mode)
    plugin = ClaudeAgentPlugin(ScriptedClaude(refund_policy))
    replayer = Replayer(workflows=[RefundAgentWorkflow], plugins=[plugin])
    await replayer.replay_workflow(history)


@pytest.mark.usefixtures("shop_dir")
async def test_only_an_allowed_approver_can_approve(
    client: Client, tmp_path: Path
) -> None:
    queue = f"tq-{uuid.uuid4().hex[:8]}"
    del tmp_path
    plugin = ClaudeAgentPlugin(ScriptedClaude(refund_policy), heartbeat_every=1.0)
    async with Worker(
        client,
        task_queue=queue,
        workflows=[RefundAgentWorkflow],
        activities=ALL,
        plugins=[plugin],
        **FAIL_FAST,
    ):
        handle = await client.start_workflow(
            RefundAgentWorkflow.run,
            BROKEN_TEAPOT,
            id=f"wf-{uuid.uuid4().hex[:8]}",
            task_queue=queue,
        )
        pending = await wait_for_approval(handle)
        assert pending is not None
        # The validator refuses it before it reaches the history.
        with pytest.raises(WorkflowUpdateFailedError):
            await handle.execute_update(
                RefundAgentWorkflow.review,
                args=[pending["id"], True, "intruder@example.com"],
            )
        assert shop.read("refunds.jsonl") == []  # no money moved
        await handle.execute_update(
            RefundAgentWorkflow.review, args=[pending["id"], True, MANAGER]
        )
        result = await asyncio.wait_for(handle.result(), 60)
        calls = await handle.query(RefundAgentWorkflow.tool_calls)
    assert result.startswith("Done.")
    assert calls[1]["decided_by"] == MANAGER
    assert len(shop.read("refunds.jsonl")) == 1


@pytest.mark.usefixtures("shop_dir")
async def test_refund_above_the_order_total_is_refused() -> None:
    with pytest.raises(ApplicationError) as err:
        await ActivityEnvironment().run(
            issue_refund, {"order_id": "A-1001", "amount": 500}
        )
    assert err.value.non_retryable and "not allowed" in str(err.value)
    assert shop.read("refunds.jsonl") == []
