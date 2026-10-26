"""How the agent behaves around its edges: defaults, handlers, failures, cancellation, limits."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections import Counter
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from temporalio.api.enums.v1 import EventType
from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    DurableClaudeAgent,
    SegmentInput,
    SegmentOutput,
    _workflow,
    follow_agent,
)
from temporalio.claude_agent_sdk.testing import Policy, ScriptedClaude
from temporalio.client import (
    Client,
    WorkflowFailureError,
    WorkflowHandle,
    WorkflowUpdateFailedError,
)
from temporalio.exceptions import ApplicationError, CancelledError
from temporalio.service import RPCError, RPCStatusCode
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from tests.conftest import LIMIT, SUGGEST_AT, wait_for_approval
from tests.endless.activities import ALL as COUNTING
from tests.endless.policy import count_policy
from tests.endless.workflows import ChatWorkflow, LongTaskWorkflow, TaskOptions
from tests.helpers.workers import FAIL_FAST
from tests.lifecycle.policy import big_input_policy, big_result_policy
from tests.lifecycle.workflows import (
    COUNT,
    BigResultWorkflow,
    LateLiveOutputWorkflow,
    OneShotWorkflow,
    SignalTaskWorkflow,
    TasksWorkflow,
    UpdateChatWorkflow,
)
from tests.refund import shop
from tests.refund.activities import ALL as SHOP
from tests.refund.policy import refund_policy
from tests.refund.workflows import MANAGER, RefundAgentWorkflow
from tests.storage.workflows import fetch_document

pytestmark = pytest.mark.timeout(180)
WORKFLOWS = [
    OneShotWorkflow,
    UpdateChatWorkflow,
    TasksWorkflow,
    SignalTaskWorkflow,
    BigResultWorkflow,
    LateLiveOutputWorkflow,
    LongTaskWorkflow,
    ChatWorkflow,
    RefundAgentWorkflow,
]


@pytest.fixture(params=["held", "store"])
def scripted(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Callable[[Policy], ScriptedClaude]:
    """Scripted Claude with the conversation in the Workflow (the default), or in a
    session store, as Workflows started by the first version keep it."""
    folder = tmp_path / "sessions" if request.param == "store" else None
    return lambda policy: ScriptedClaude(policy, folder)


def scripted_worker(
    client: Client, queue: str, runner: Any, *, fail_fast: bool = True
) -> Worker:
    """A Worker for every Workflow in this file."""
    return Worker(
        client,
        task_queue=queue,
        workflows=WORKFLOWS,
        activities=[*COUNTING, *SHOP, fetch_document],
        plugins=[ClaudeAgentPlugin(runner, heartbeat_every=1.0)],
        **(FAIL_FAST if fail_fast else {}),
    )


def counted() -> list[str]:
    """The step numbers ``count`` really executed, in order."""
    return [e["detail"] for e in shop.executions("count")]


async def activity_types(handle: WorkflowHandle[Any, Any]) -> Counter[str]:
    """How many Activities of each type the run scheduled."""
    kinds: Counter[str] = Counter()
    async for event in handle.fetch_history_events():
        if event.HasField("activity_task_scheduled_event_attributes"):
            name = event.activity_task_scheduled_event_attributes.activity_type.name
            kinds[name] += 1
    return kinds


@pytest.mark.usefixtures("shop_dir")
async def test_quick_start_shape_never_continues_as_new_by_default(
    limited: WorkflowEnvironment,
    scripted: Callable[[Policy], ScriptedClaude],
) -> None:
    """A ``run(self, request)`` Workflow keeps working after the server suggests Continue-As-New."""
    client, queue = limited.client, f"oneshot-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, scripted(count_policy)):
        handle = await client.start_workflow(
            OneShotWorkflow.run, "count to 25", id=queue, task_queue=queue
        )
        result = await asyncio.wait_for(handle.result(), 120)
        length = len((await handle.fetch_history()).events)
        runs = [e async for e in client.list_workflows(f'WorkflowId = "{queue}"')]
    assert result == "counted to 25"
    assert SUGGEST_AT < length < LIMIT  # suggested, and still one run
    assert len(runs) == 1


@pytest.mark.usefixtures("shop_dir")
async def test_update_chat_continues_as_new_between_messages(
    client: Client,
    scripted: Callable[[Policy], ScriptedClaude],
) -> None:
    """``run`` in an Update handler, and ``continue_as_new()`` from the run method."""
    queue = f"upchat-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, scripted(count_policy)):
        await client.start_workflow(
            UpdateChatWorkflow.run, args=[None, False], id=queue, task_queue=queue
        )
        latest = client.get_workflow_handle(queue)  # follows Continue-As-New
        replies = []
        for text in ("count to 2", "count to 4", "count to 6", "count to 8"):
            replies.append(await latest.execute_update(UpdateChatWorkflow.ask, text))
        runs = await latest.query(UpdateChatWorkflow.runs)
        await latest.terminate()
    assert replies == ["counted to 2", "counted to 4", "counted to 6", "counted to 8"]
    assert runs > 1  # it continued as new between messages
    assert counted() == [str(n) for n in range(1, 9)]  # one session, each step once


@pytest.mark.usefixtures("shop_dir")
async def test_checking_continue_as_new_in_a_wait_condition_is_safe(
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Callable[[Policy], ScriptedClaude],
) -> None:
    """``should_continue_as_new()`` in a ``wait_condition`` (as this chat's run method
    does) when the state no longer fits: it says False and the chat goes on, with no
    failed Workflow task (the check that decides this records a patch marker)."""
    monkeypatch.setattr(
        _workflow.DurableClaudeAgent,
        "_handover_problem",
        lambda self: "The agent's state is too large (this test says so).",
    )
    queue = f"waitcheck-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[UpdateChatWorkflow],
        activities=COUNTING,
        plugins=[ClaudeAgentPlugin(scripted(count_policy))],
        **FAIL_FAST,
    ):
        handle = await client.start_workflow(
            UpdateChatWorkflow.run, args=[None, False], id=queue, task_queue=queue
        )
        replies = [
            await handle.execute_update(UpdateChatWorkflow.ask, f"count to {n}")
            for n in (2, 4, 6, 8)
        ]
        runs = await handle.query(UpdateChatWorkflow.runs)
        kinds = [e.event_type async for e in handle.fetch_history_events()]
        await handle.terminate()
    assert replies == ["counted to 2", "counted to 4", "counted to 6", "counted to 8"]
    assert runs == 1  # its state did not fit, so it stayed in this run
    assert EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED not in kinds
    assert EventType.EVENT_TYPE_MARKER_RECORDED in kinds


@pytest.mark.usefixtures("shop_dir")
async def test_auto_continue_as_new_refuses_to_run_in_an_update_handler(
    client: Client,
    scripted: Callable[[Policy], ScriptedClaude],
) -> None:
    """It fails the Update at once instead of deadlocking at Continue-As-New."""
    queue = f"upauto-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, scripted(count_policy)):
        handle = await client.start_workflow(
            UpdateChatWorkflow.run, args=[None, True], id=queue, task_queue=queue
        )
        with pytest.raises(WorkflowUpdateFailedError) as err:
            await asyncio.wait_for(
                handle.execute_update(UpdateChatWorkflow.ask, "count to 1"), 30
            )
        await handle.terminate()
    assert isinstance(err.value.cause, ApplicationError)
    assert "auto_continue_as_new" in err.value.cause.message
    assert counted() == []


@pytest.mark.usefixtures("shop_dir")
async def test_cancel_during_a_tool_cancels_the_workflow(
    client: Client,
    scripted: Callable[[Policy], ScriptedClaude],
) -> None:
    """The cancellation is not reported to Claude as a tool failure; no segment runs after it."""
    os.environ["REFUND_DELAY"] = "3"  # money moves, then the tool is slow to reply
    queue = f"cancel-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, scripted(refund_policy)):
        handle = await client.start_workflow(
            RefundAgentWorkflow.run,
            "Order A-1001 arrived broken, I want my money back.",
            id=queue,
            task_queue=queue,
        )
        pending = await wait_for_approval(handle)
        assert pending is not None
        await handle.execute_update(
            RefundAgentWorkflow.review, args=[pending["id"], True, MANAGER]
        )
        while not shop.read("refunds.jsonl"):
            await asyncio.sleep(0.05)
        await handle.cancel()
        with pytest.raises(WorkflowFailureError) as err:
            await asyncio.wait_for(handle.result(), 60)
    assert isinstance(err.value.cause, CancelledError)
    kinds = await activity_types(handle)
    assert kinds["run_claude_segment"] == 2  # none after the cancel
    assert kinds["email_customer"] == 0


class ExplodingRunner:
    """Scripted Claude, except that the prompt 'explode' breaks its segment for good."""

    def __init__(self, inner: ScriptedClaude) -> None:
        self.inner = inner

    async def run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        """Run the segment, or fail it."""
        if inp.prompt == "explode":
            raise ApplicationError("the engine exploded", non_retryable=True)
        return await self.inner.run(inp, attempt)


@pytest.mark.usefixtures("shop_dir")
async def test_a_failed_task_does_not_poison_the_next_one(
    client: Client,
    scripted: Callable[[Policy], ScriptedClaude],
) -> None:
    """``max_segments`` stops before a tool runs; a failed segment ends its task; the next task works."""
    queue = f"tasks-{uuid.uuid4().hex[:8]}"
    runner = ExplodingRunner(scripted(count_policy))
    async with scripted_worker(client, queue, runner):
        answers = await client.execute_workflow(
            TasksWorkflow.run,
            ["count to 5", "explode", "count to 1"],
            id=queue,
            task_queue=queue,
        )
    assert answers[0].startswith(
        "error: ApplicationError: Stopped after 2 segments, before running"
    )
    assert answers[1].startswith("error: ActivityError:")  # catch FailureError for both
    assert answers[2] == "counted to 1"
    assert counted() == ["1"]  # the stopped call never ran


@pytest.mark.usefixtures("shop_dir")
async def test_a_chat_without_tool_calls_continues_as_new_between_messages(
    client: Client,
    scripted: Callable[[Policy], ScriptedClaude],
) -> None:
    """Turns without tool calls never reach a point between tool calls; the chat hands over."""
    queue = f"notools-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, scripted(count_policy)):
        handle = await client.start_workflow(
            ChatWorkflow.run, 12, id=queue, task_queue=queue
        )
        for _ in range(12):
            await handle.signal(ChatWorkflow.send, "count to 0")
        replies = await asyncio.wait_for(handle.result(), 90)
        progress = await handle.query(ChatWorkflow.progress)
    assert replies == ["counted to 0"] * 12
    assert progress["runs"] > 1 and progress["tool_calls"] == 0


async def test_auto_continue_as_new_refuses_to_run_in_a_signal_handler(
    client: Client,
    scripted: Callable[[Policy], ScriptedClaude],
) -> None:
    queue = f"sigauto-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, scripted(count_policy)):
        handle = await client.start_workflow(
            SignalTaskWorkflow.run, id=queue, task_queue=queue
        )
        await handle.signal(SignalTaskWorkflow.ask, "count to 1")
        with pytest.raises(WorkflowFailureError) as err:
            await asyncio.wait_for(handle.result(), 30)
    assert isinstance(err.value.cause, ApplicationError)
    assert "a Signal handler" in err.value.cause.message


@pytest.mark.usefixtures("shop_dir")
async def test_continue_as_new_carries_a_large_result_with_live_output(
    client: Client, tmp_path: Path
) -> None:
    """A 1.9 MB result waits in the state; the stream carried with it shrinks to fit."""
    queue = f"bigres-{uuid.uuid4().hex[:8]}"
    runner = ScriptedClaude(big_result_policy, tmp_path)
    async with scripted_worker(client, queue, runner):
        handle = await client.start_workflow(
            BigResultWorkflow.run,
            args=["fetch 1900 KB twice", None],
            id=queue,
            task_queue=queue,
        )
        result = await asyncio.wait_for(handle.result(), 90)
        runs = await client.get_workflow_handle(queue).query(BigResultWorkflow.runs)
    assert result == "fetched 2 documents" and runs > 1


async def wait_for_pending(handle: WorkflowHandle[Any, Any]) -> dict[str, Any]:
    """Wait until the long task asks for an approval."""
    for _ in range(600):
        pending = await handle.query(LongTaskWorkflow.pending_approvals)
        if pending:
            return pending[0]
        await asyncio.sleep(0.1)
    raise TimeoutError("no approval request")


async def park_subscribers(client: Client, workflow_id: str, how_many: int) -> None:
    """Open subscriptions that read to the end, wait there, then are abandoned."""

    async def subscribe() -> None:
        async for _ in follow_agent(client, workflow_id):
            pass

    tasks = [asyncio.ensure_future(subscribe()) for _ in range(how_many)]
    await asyncio.sleep(3)  # every poll is now an Update waiting for the next event
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.usefixtures("shop_dir")
async def test_approval_by_signal_works_when_subscribers_use_up_the_update_limit(
    client: Client,
    scripted: Callable[[Policy], ScriptedClaude],
) -> None:
    queue = f"many-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, scripted(count_policy)):
        handle = await client.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 2 and publish", TaskOptions(live=True), None],
            id=queue,
            task_queue=queue,
        )
        pending = await wait_for_pending(handle)
        await park_subscribers(client, queue, 12)
        # Ten Updates in flight is the server's limit: an Update approval cannot land.
        with pytest.raises(RPCError) as refused:
            await handle.execute_update(
                LongTaskWorkflow.review,
                args=[pending["id"], True],
                rpc_timeout=timedelta(seconds=3),
            )
        assert refused.value.status == RPCStatusCode.RESOURCE_EXHAUSTED
        await handle.signal(LongTaskWorkflow.review_by_signal, args=["nope", True])
        await handle.signal(
            LongTaskWorkflow.review_by_signal, args=[pending["id"], True]
        )
        result = await asyncio.wait_for(handle.result(), 60)
    assert result == "counted to 2"
    assert len(shop.executions("publish")) == 1


@pytest.mark.usefixtures("shop_dir")
async def test_continue_as_new_input_stays_small_with_big_tool_inputs(
    client: Client, tmp_path: Path
) -> None:
    """Live output carries at most ``live_output_keep_bytes`` of events to the next run."""
    queue = f"big-{uuid.uuid4().hex[:8]}"
    runner = ScriptedClaude(big_input_policy, tmp_path)
    async with scripted_worker(client, queue, runner):
        handle = await client.start_workflow(
            LongTaskWorkflow.run,
            args=["big 8", TaskOptions(after_events=40, live=True), None],
            id=queue,
            task_queue=queue,
        )
        events = []
        async for event in follow_agent(client, queue):
            events.append(event)
            if event["type"] in ("done", "error"):
                break
        result = await asyncio.wait_for(handle.result(), 90)
        progress = await handle.query(LongTaskWorkflow.progress)
    assert result == "counted to 8" and progress["runs"] > 1
    calls = [e for e in events if e["type"] == "tool_call"]
    assert len(calls) == 8
    assert all(c["truncated"] and len(c["input"]) == 32 * 1024 for c in calls)


async def test_live_output_agent_must_be_created_during_init(
    client: Client,
    scripted: Callable[[Policy], ScriptedClaude],
) -> None:
    queue = f"late-{uuid.uuid4().hex[:8]}"
    # The default Worker options: this checks the failed Workflow task itself.
    async with scripted_worker(client, queue, scripted(count_policy), fail_fast=False):
        handle = await client.start_workflow(
            LateLiveOutputWorkflow.run, "count to 1", id=queue, task_queue=queue
        )
        message = ""
        for _ in range(100):
            async for event in handle.fetch_history_events():
                if event.HasField("workflow_task_failed_event_attributes"):
                    message = (
                        event.workflow_task_failed_event_attributes.failure.message
                    )
            if message:
                break
            await asyncio.sleep(0.1)
        await handle.terminate()
    assert "live_output=True" in message and "initialized" in message


def test_tool_names_must_be_unique() -> None:
    with pytest.raises(ValueError, match="unique: count"):
        DurableClaudeAgent(tools=[COUNT, COUNT])


@pytest.mark.usefixtures("shop_dir")
async def test_histories_with_continue_as_new_and_live_output_replay(
    client: Client,
    scripted: Callable[[Policy], ScriptedClaude],
) -> None:
    queue = f"replay-{uuid.uuid4().hex[:8]}"
    async with scripted_worker(client, queue, scripted(count_policy)):
        handle = await client.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 8", TaskOptions(after_events=40, live=True), None],
            id=queue,
            task_queue=queue,
        )
        async for event in follow_agent(client, queue):
            if event["type"] in ("done", "error"):
                break
        assert await asyncio.wait_for(handle.result(), 60) == "counted to 8"
    histories = []
    async for execution in client.list_workflows(f'WorkflowId = "{queue}"'):
        run = client.get_workflow_handle(queue, run_id=execution.run_id)
        histories.append(await run.fetch_history())
    assert len(histories) > 1
    replayer = Replayer(workflows=WORKFLOWS)
    for history in histories:
        await replayer.replay_workflow(history)
