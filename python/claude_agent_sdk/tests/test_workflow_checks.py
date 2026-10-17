"""What the Workflow refuses, and how a task stops, without an engine.

A segment runner that breaks its contract (no checkpoint, a splice that does not fit
the conversation, a conversation kept somewhere else) fails the task before any other
tool runs. Two agents in one Workflow keep their conversations apart. The calls of
one message finish or stop together when the Workflow is cancelled. A second task
while one runs is refused with a message, and a call to an unknown tool is an error
Claude reads. The Continue-As-New decisions follow their documented rules, for new
runs and for histories that older versions recorded.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
import uuid
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from temporalio.claude_agent_sdk import (
    AgentState,
    ClaudeAgentPlugin,
    DeferredCall,
    DurableClaudeAgent,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    ToolStepInput,
)
from temporalio.claude_agent_sdk import _workflow as wf
from temporalio.claude_agent_sdk.testing import (
    Final,
    HistoryItem,
    ScriptedClaude,
    ToolCall,
)
from temporalio.client import (
    Client,
    WorkflowFailureError,
    WorkflowHandle,
    WorkflowUpdateFailedError,
)
from temporalio.contrib.workflow_streams import PublisherState, WorkflowStreamState
from temporalio.exceptions import ApplicationError, CancelledError
from temporalio.worker import Worker
from tests.checks.activities import ALL as CHECKS
from tests.checks.workflows import BothAtOnceWorkflow, MisuseWorkflow, TwoAgentsWorkflow
from tests.endless.activities import ALL as COUNTING
from tests.endless.policy import count_policy
from tests.helpers.outside_workflow import (
    CarriedStream,
    ContinuedAsNew,
    OutsideWorkflow,
    Topic,
)
from tests.helpers.workers import FAIL_FAST
from tests.lifecycle.workflows import OneShotWorkflow
from tests.refund import shop

pytestmark = [pytest.mark.timeout(120), pytest.mark.usefixtures("shop_dir")]


def worker(client: Client, queue: str, runner: Any) -> Worker:
    return Worker(
        client,
        task_queue=queue,
        workflows=[
            OneShotWorkflow,
            TwoAgentsWorkflow,
            BothAtOnceWorkflow,
            MisuseWorkflow,
        ],
        activities=[*COUNTING, *CHECKS],
        plugins=[ClaudeAgentPlugin(runner, heartbeat_every=1.0)],
        **FAIL_FAST,
    )


def ran(tool: str) -> list[str]:
    """What ``tool`` really did, in order (from the shop ledger)."""
    return [e["detail"] for e in shop.executions(tool)]


async def scheduled(handle: WorkflowHandle[Any, Any]) -> Counter[str]:
    """How many times the run scheduled each Activity ID."""
    ids: Counter[str] = Counter()
    async for event in handle.fetch_history_events():
        if event.HasField("activity_task_scheduled_event_attributes"):
            ids[event.activity_task_scheduled_event_attributes.activity_id] += 1
    return ids


async def eventually(check: Callable[[], bool], timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.05)


# ---- a runner that breaks its contract ----


class BrokenStep:
    """ScriptedClaude, but segment ``at`` comes back changed by ``breaks``."""

    def __init__(
        self, inner: ScriptedClaude, at: int, breaks: Callable[[SegmentOutput], None]
    ) -> None:
        self.inner, self.at, self.breaks = inner, at, breaks

    async def run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        out = await self.inner.run(inp, attempt)
        if inp.segment_index == self.at:
            self.breaks(out)
        return out

    async def run_tool_step(self, step: ToolStepInput, attempt: int) -> ToolOutcome:
        return await self.inner.run_tool_step(step, attempt)


def setting(**fields: Any) -> Callable[[SegmentOutput], None]:
    def breaks(out: SegmentOutput) -> None:
        for name, value in fields.items():
            setattr(out, name, value)

    return breaks


SAME = (
    "Every Worker of a task queue needs the same kind of runner (with or without a "
    "session store)."
)
BROKEN_STEPS: dict[str, tuple[bool, Callable[[SegmentOutput], None], str]] = {
    "no checkpoint": (
        False,
        setting(checkpoint=None),
        "The segment runner returned no checkpoint, so a retry could not continue "
        "this session safely.",
    ),
    "keeps more than the Workflow holds": (
        False,
        setting(transcript_keep=5),
        "The segment runner kept 5 of 1 conversation entries and added 1.",
    ),
    "keeps fewer than none": (
        False,
        setting(transcript_keep=-1),
        "The segment runner kept -1 of 1 conversation entries and added 1.",
    ),
    "keeps and adds nothing": (
        False,
        setting(transcript_keep=0, transcript_add=[]),
        "The segment runner kept 0 of 1 conversation entries and added 0.",
    ),
    "a step from a store runner, the Workflow holds it": (
        False,
        setting(transcript_keep=None),
        "The segment runner keeps conversations in a session store, but this "
        f"Workflow holds this one. {SAME}",
    ),
    "a step for the Workflow, a store keeps it": (
        True,
        setting(transcript_keep=0, transcript_add=[{"uuid": "x"}]),
        "The segment runner returned a conversation for the Workflow to hold, but "
        f"this one is kept in a session store. {SAME}",
    ),
}


@pytest.mark.parametrize("case", list(BROKEN_STEPS))
async def test_a_step_that_breaks_the_runner_contract_stops_the_task(
    client: Client, tmp_path: Path, case: str
) -> None:
    """Before the call it paused at runs: nothing runs after a step it cannot trust."""
    in_store, breaks, message = BROKEN_STEPS[case]
    inner = ScriptedClaude(count_policy, tmp_path / "sessions" if in_store else None)
    queue = f"broken-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, BrokenStep(inner, at=1, breaks=breaks)):
        handle = await client.start_workflow(
            OneShotWorkflow.run, "count to 3", id=queue, task_queue=queue
        )
        with pytest.raises(WorkflowFailureError) as failed:
            await handle.result()
    cause = failed.value.cause
    assert isinstance(cause, ApplicationError) and cause.non_retryable
    assert cause.message == message
    assert ran("count") == ["1"]  # the call of the broken step never ran


# ---- agents and calls side by side ----


async def test_two_agents_in_one_workflow_keep_their_conversations_apart(
    client: Client,
) -> None:
    """Each segment reads its own agent's conversation (the Query serves both)."""
    queue = f"two-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(count_policy)):
        handle = await client.start_workflow(
            TwoAgentsWorkflow.run,
            args=["count to 2", "count to 3"],
            id=queue,
            task_queue=queue,
        )
        assert await handle.result() == ["counted to 2", "counted to 3"]
        prompts = await handle.query(TwoAgentsWorkflow.prompts)
    assert prompts == [["count to 2"], ["count to 3"]]
    assert sorted(ran("count")) == ["1", "1", "2", "2", "3"]


def both_policy(prompt: str, history: list[HistoryItem]) -> list[ToolCall] | Final:
    del prompt
    if not history:
        return [ToolCall("quick", {}), ToolCall("slow", {})]
    return Final("both returned")


async def test_a_cancel_while_two_calls_run_stops_them_once_each(
    client: Client,
) -> None:
    """The finished call stays finished, the running one is cancelled (and says so),
    and the cancel goes on only when both are settled. Nothing runs twice."""
    queue = f"both-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(both_policy)):
        handle = await client.start_workflow(
            BothAtOnceWorkflow.run, "go", id=queue, task_queue=queue
        )
        # Cancel once the Workflow itself saw the quick call finish (not only the
        # ledger), so the cancel never lands in the task that delivers its result.
        deadline = time.monotonic() + 30
        while True:
            calls = await handle.query(BothAtOnceWorkflow.calls)
            statuses = {c["name"]: c["status"] for c in calls}
            if statuses == {"quick": "done", "slow": "started"}:
                break
            assert time.monotonic() < deadline, statuses
            await asyncio.sleep(0.05)
        await eventually(lambda: ran("slow") == ["started"])
        await handle.cancel()
        with pytest.raises(WorkflowFailureError) as failed:
            await handle.result()
        statuses = {
            c["name"]: c["status"] for c in await handle.query(BothAtOnceWorkflow.calls)
        }
        ids = await scheduled(handle)
    assert isinstance(failed.value.cause, CancelledError)
    assert statuses == {"quick": "done", "slow": "cancelled"}
    assert ran("quick") == ["done"] and ran("slow") == ["started", "cancelled"]
    assert ids and set(ids.values()) == {1}  # no Activity scheduled twice


# ---- misuse ----


async def test_a_second_task_while_one_runs_is_refused(client: Client) -> None:
    queue = f"twice-{uuid.uuid4().hex[:8]}"
    runner = ScriptedClaude(count_policy, think_seconds=0.3)
    async with worker(client, queue, runner):
        handle = await client.start_workflow(
            MisuseWorkflow.run, "count to 2", id=queue, task_queue=queue
        )
        await eventually(lambda: ran("count") == ["1"])
        with pytest.raises(WorkflowUpdateFailedError) as refused:
            await handle.execute_update(MisuseWorkflow.ask, "count to 9")
        assert await handle.result() == "counted to 2"
    cause = refused.value.cause
    assert isinstance(cause, ApplicationError) and cause.non_retryable
    assert cause.message == (
        "The agent runs one task at a time; wait for the current task to end."
    )
    assert ran("count") == ["1", "2"]


async def test_a_bug_in_workflow_code_fails_the_test_in_seconds(
    client: Client,
) -> None:
    """``run(None)`` with no unfinished task is a bug in Workflow code: a
    ``ValueError``. By default Temporal fails the Workflow task and tries it again
    until the code is fixed; the tests' Workers (``FAIL_FAST``) fail the Workflow at
    once instead, so this test gets the error in seconds, not at its timeout."""
    queue = f"none-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(count_policy)):
        handle = await client.start_workflow(
            MisuseWorkflow.run, None, id=queue, task_queue=queue
        )
        with pytest.raises(WorkflowFailureError) as failed:
            await handle.result()
    cause = failed.value.cause
    assert isinstance(cause, ApplicationError) and cause.type == "ValueError"
    assert cause.message == "No prompt, and no unfinished task to continue"


def unknown_policy(prompt: str, history: list[HistoryItem]) -> ToolCall | Final:
    del prompt
    if not history:
        return ToolCall("missing", {"x": 1})
    last = history[-1]
    return Final(f"{last.content} ({'error' if last.is_error else 'ok'})")


async def test_a_call_to_an_unknown_tool_is_an_error_for_claude(
    client: Client,
) -> None:
    """For example a tool removed from the Workflow while Claude still knows it."""
    queue = f"unknown-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(unknown_policy)):
        handle = await client.start_workflow(
            MisuseWorkflow.run, "use a tool", id=queue, task_queue=queue
        )
        assert await handle.result() == "Unknown tool: missing (error)"
        calls = await handle.query(MisuseWorkflow.calls)
        ids = await scheduled(handle)
    assert [(c["name"], c["status"]) for c in calls] == [("missing", "unknown tool")]
    assert not [i for i in ids if i.startswith("tool-")]  # no Activity ran for it


# ---- outside a Workflow: the decisions themselves ----


def test_an_agent_needs_room_for_at_least_one_segment() -> None:
    with pytest.raises(ValueError, match="max_segments must be at least 1, or None"):
        DurableClaudeAgent(max_segments=0)
    assert DurableClaudeAgent(max_segments=1).segments == 0
    assert DurableClaudeAgent(state=AgentState(segments=7)).segments == 7


def test_a_decision_counts_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    OutsideWorkflow().install(monkeypatch)
    agent = DurableClaudeAgent(approvers=["ana"])
    agent._waiting["toolu_1"] = DeferredCall(id="toolu_1", name="refund", input={})
    agent._calls["toolu_1"] = {"id": "toolu_1", "status": "waiting for approval"}
    with caplog.at_level(logging.WARNING, logger="outside-workflow"):
        assert not agent.decide("toolu_1", True, "bob")  # not an approver
        assert agent.decide("toolu_1", False, "ana")
        assert not agent.decide("toolu_1", True, "ana")  # already decided
        assert not agent.decide("toolu_9", True, "ana")  # nothing waits
    with pytest.raises(ValueError, match="Tool call toolu_1 was already decided"):
        agent.validate_decision("toolu_1", "ana")
    assert agent._decisions == {"toolu_1": False}
    assert agent._calls["toolu_1"]["decided_by"] == "ana"
    ignored = [r.getMessage() for r in caplog.records]
    assert ignored == [
        "Ignored a decision: 'bob' is not allowed to approve tool calls",
        "Ignored a decision: Tool call toolu_1 was already decided",
        "Ignored a decision: No tool call toolu_9 is waiting for approval",
    ]


async def test_the_calls_of_a_message_all_settle_before_an_error_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = DurableClaudeAgent()
    finished: list[str] = []

    async def run_tool(call: DeferredCall) -> ToolOutcome:
        if call.id == "bad":
            raise RuntimeError("a bug in one call")
        await asyncio.sleep(0.05)
        finished.append(call.id)
        return ToolOutcome(content=call.id)

    monkeypatch.setattr(agent, "_run_tool", run_tool)
    calls = [DeferredCall(id=i, name="t", input={}) for i in ("a", "bad", "c")]
    with pytest.raises(RuntimeError, match="a bug in one call"):
        await agent._run_tools(calls)
    assert sorted(finished) == ["a", "c"]
    good = [calls[0], calls[2]]
    assert [o.content for o in await agent._run_tools(good)] == ["a", "c"]
    assert [o.content for o in await agent._run_tools(good[1:])] == ["c"]


class HandedOver(Exception):
    pass


class Stopped(Exception):
    pass


def can_expected(
    suggested: bool, too_large: bool, full: bool, first: bool, new: bool
) -> str:
    """What the agent does before a step, written from the docs (not the code).

    A run of this version continues as new when the server suggests it or the run is
    nearly full, if the state fits; when it does not fit, the agent keeps going until
    the run is nearly full, then stops the task with a message. A history recorded by
    the first published version (``new`` False) replays its decisions: it continued
    as new when the server suggested it, only before steps after tool calls, and it
    never stopped a task.
    """
    if not new:
        return "continue as new" if suggested and not first else "keep going"
    if too_large:
        return "stop" if full else "keep going"
    return "continue as new" if suggested or full else "keep going"


@pytest.mark.parametrize("new", [True, False], ids=["new run", "first version"])
async def test_continue_as_new_decisions_follow_the_table(
    monkeypatch: pytest.MonkeyPatch, new: bool
) -> None:
    outside = OutsideWorkflow(patched=new).install(monkeypatch)
    for suggested, too_large, full, first in itertools.product((False, True), repeat=4):
        agent = DurableClaudeAgent(auto_continue_as_new=True)
        outside.history.suggested = suggested
        monkeypatch.setattr(
            agent,
            "_history_nearly_full",
            lambda full=full: "nearly full" if full else None,
        )
        monkeypatch.setattr(
            agent,
            "_handover_problem",
            lambda measured=None, too_large=too_large: (
                "too large" if too_large else None
            ),
        )

        async def hand_over() -> None:
            raise HandedOver

        async def fail(message: str) -> None:
            raise Stopped(message)

        monkeypatch.setattr(agent, "_hand_over", hand_over)
        monkeypatch.setattr(agent, "_fail", fail)
        try:
            await agent._continue_as_new_or_stop(first_of_task=first)
            got = "keep going"
        except HandedOver:
            got = "continue as new"
        except Stopped as stop:
            got = "stop"
            assert str(stop) == (
                "nearly full, and the agent cannot continue as new, so the task stops "
                "here instead of the server ending the Workflow. too large"
            )
        want = can_expected(suggested, too_large, full, first, new)
        case = (
            f"suggested {suggested}, too large {too_large}, full {full}, first {first}"
        )
        assert got == want, case


async def test_a_fixed_history_length_decides_continue_as_new(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = OutsideWorkflow().install(monkeypatch)
    agent = DurableClaudeAgent(continue_as_new_after_events=100)
    outside.history.suggested = True  # ignored: the fixed length decides
    outside.history.events = 99
    assert not agent.should_continue_as_new()
    outside.history.events = 100
    assert agent.should_continue_as_new()


def test_tool_step_attempts_leave_room_in_the_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another attempt of a tool step whose call did not run needs room for a step
    under Temporal's limits, within the call's own share of the history; with
    ``auto_continue_as_new``, none once it is time to continue as new (the next
    segment does, and Claude can call again there)."""
    outside = OutsideWorkflow().install(monkeypatch)

    def room(
        events: int, size: int = 0, suggested: bool = False, start: int = 0, **kw: Any
    ) -> bool:
        outside.history.events, outside.history.size = events, size
        outside.history.suggested = suggested
        return DurableClaudeAgent(**kw)._room_for_another_attempt(start)

    assert room(1_000)
    assert room(1_000 + wf._ATTEMPT_EVENTS - 1, start=1_000)
    assert not room(1_000 + wf._ATTEMPT_EVENTS, start=1_000)  # this call's share
    assert not room(wf._HISTORY_EVENTS - wf._ROOM_EVENTS)
    assert not room(1_000, size=wf._HISTORY_BYTES - wf._ROOM_BYTES)
    more: dict[str, Any] = {"continue_as_new_after_events": 100_000}  # allowed more
    assert room(wf._HISTORY_EVENTS, start=wf._HISTORY_EVENTS - 10, **more)
    assert room(1_000, suggested=True)  # this agent does not continue as new
    assert not room(1_000, suggested=True, auto_continue_as_new=True)
    after: dict[str, Any] = {
        "auto_continue_as_new": True,
        "continue_as_new_after_events": 500,
    }
    assert room(499, suggested=True, **after)  # the fixed length decides
    assert not room(500, **after)


async def test_a_state_too_large_to_carry_is_said_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    outside = OutsideWorkflow().install(monkeypatch)
    outside.history.suggested = True
    monkeypatch.setattr(wf, "PAYLOAD_LIMIT_BYTES", 1000)
    agent = DurableClaudeAgent(state=AgentState(conversation=["x" * 2000]))
    with caplog.at_level(logging.WARNING, logger="outside-workflow"):
        assert not agent.should_continue_as_new()
        assert not agent.should_continue_as_new()
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert warnings[0].startswith(
        "The agent's state is 0.00 MB, mostly the conversation"
    )
    assert warnings[0].endswith("The agent keeps going in this run.")


# ---- handing over: measured again when it happens ----


def carrying(
    conversation: list[str], publishers: int
) -> tuple[DurableClaudeAgent, CarriedStream, Topic]:
    agent = DurableClaudeAgent(
        state=AgentState(task_prompt="go", conversation=conversation)
    )
    seen = datetime(2026, 10, 3, tzinfo=timezone.utc)
    stream = CarriedStream(
        WorkflowStreamState(
            publishers={
                f"publisher-{i:04d}": PublisherState(sequence=i, last_seen=seen)
                for i in range(publishers)
            }
        )
    )
    topic = Topic()
    agent._stream = stream  # type: ignore[assignment]
    agent._topic = topic  # type: ignore[assignment]
    return agent, stream, topic


@pytest.mark.parametrize(
    ("size", "publishers", "answers", "outcome"),
    [
        (100, 0, {}, "continue as new"),
        (2000, 0, {}, "fail"),  # the state is too large
        (2000, 0, {wf._CHECKS_PATCH: False, wf._CARRY_PATCH: False}, "continue as new"),
        (700, 40, {}, "fail"),  # fits, but not with the stream's publishers
        (700, 40, {wf._CARRY_PATCH: False}, "continue as new"),  # before that check
    ],
    ids=[
        "fits",
        "too large",
        "first version",
        "stream too large",
        "before carry check",
    ],
)
async def test_handing_over_checks_the_new_runs_input_as_it_will_be(
    monkeypatch: pytest.MonkeyPatch,
    size: int,
    publishers: int,
    answers: dict[str, bool],
    outcome: str,
) -> None:
    """Measured after the handlers finished, with the live output stream in it."""
    OutsideWorkflow(answers=answers).install(monkeypatch)
    monkeypatch.setattr(wf, "PAYLOAD_LIMIT_BYTES", 2048)
    agent, stream, topic = carrying(["x" * size], publishers)
    if outcome == "fail":
        with pytest.raises(ApplicationError) as failed:
            await agent._hand_over()
        assert failed.value.non_retryable
        assert "more than the new run's input can carry" in failed.value.message
        return
    with pytest.raises(ContinuedAsNew) as continued:
        await agent._hand_over()
    prompt, state = continued.value.run_args
    assert prompt == "go" and state.runs == 2 and state.conversation == ["x" * size]
    assert stream.detached
    assert [e["type"] for e in topic.events] == ["continued_as_new"]
    assert topic.events[0]["run"] == 2
    assert state.stream is not None and len(state.stream.publishers) == publishers
