"""Human-in-the-loop: SDK-native interrupt mapped to Query + Update.

``interrupt_on={...}`` makes Deep Agents pause before a guarded tool runs. With a
checkpointer configured, LangGraph does *not* raise out of ``ainvoke`` — it
returns the current state with an ``__interrupt__`` entry describing the pending
approval (verified against deepagents 0.6.12 / langchain 1.x). Because the loop
runs in the workflow, that pause surfaces directly in workflow code. The plugin's
recommended mapping — used here — is: detect the returned ``__interrupt__``,
expose its payload via a Query, and resume with a Workflow Update carrying the
human's decision through ``Command(resume=...)``. No shim exception is invented;
the native LangGraph resume protocol (``{"decisions": [{"type": ...}]}``) is used
as-is.

State lives on the workflow instance (per-execution), which is the idiomatic
Temporal pattern for state shared between the run method and its handlers.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import uuid
from datetime import timedelta
from typing import Any

import pytest

from temporalio.testing import WorkflowEnvironment

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="deepagents requires Python >= 3.11"
)

pytest.importorskip("deepagents")
pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from temporalio import activity, workflow  # noqa: E402
from temporalio.client import WorkflowExecutionStatus  # noqa: E402
from temporalio.worker import Replayer, Worker  # noqa: E402
from tests.helpers.deepagents import count_scheduled_activities  # noqa: E402

# Bind deepagents symbols off the module importorskip returns: a static
# `from deepagents import ...` cannot resolve on Python 3.10 (deepagents
# needs >= 3.11), and with the package absent the type checkers mis-resolve
# the name against this same-named test directory.
create_deep_agent = pytest.importorskip("deepagents").create_deep_agent

with workflow.unsafe.imports_passed_through():
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from temporalio.deepagents import (
        DeepAgentsPlugin,
        ToolApprovals,
        activity_as_tool,
        tool_as_activity,
    )
    from temporalio.deepagents.testing import FakeModel, mock_model_provider


@workflow.defn
class HitlWorkflow:
    def __init__(self) -> None:
        self._interrupt: str | None = None
        self._resume_value: str | None = None
        self._resumed = False

    @workflow.run
    async def run(self, city: str) -> str:
        def book_trip(city: str) -> str:
            """Book a trip to a city (requires human approval)."""
            return f"Booked a trip to {city}."

        trip_tool = tool_as_activity(
            book_trip, start_to_close_timeout=timedelta(seconds=30)
        )
        agent = create_deep_agent(
            model="anthropic:claude-sonnet-4-5",
            tools=[trip_tool],
            interrupt_on={"book_trip": True},
            checkpointer=InMemorySaver(),
        )
        config: RunnableConfig = {
            "configurable": {"thread_id": workflow.info().workflow_id}
        }
        payload: Any = {"messages": [{"role": "user", "content": f"Book {city}."}]}
        result = await agent.ainvoke(payload, config=config)
        # LangGraph returns (not raises) the pending approval under __interrupt__.
        pending = result.get("__interrupt__")
        if pending:
            self._interrupt = str(getattr(pending[0], "value", pending[0]))
            await workflow.wait_condition(lambda: self._resumed)
            result = await agent.ainvoke(
                Command(resume={"decisions": [{"type": self._resume_value}]}),
                config=config,
            )
        return result["messages"][-1].content

    @workflow.query
    def pending_interrupt(self) -> str | None:
        return self._interrupt

    @workflow.update
    async def resume(self, decision: str) -> None:
        self._resume_value = decision
        self._resumed = True


@pytest.mark.asyncio
async def test_interrupt_query_then_resume(env: WorkflowEnvironment) -> None:
    approve = AIMessage(
        content="",
        tool_calls=[{"name": "book_trip", "args": {"city": "Rome"}, "id": "c1"}],
    )
    done = AIMessage(content="Booked a trip to Rome.")
    plugin = DeepAgentsPlugin(model_provider=mock_model_provider([approve, done]))
    async with Worker(
        env.client,
        task_queue="da-hitl",
        workflows=[HitlWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            HitlWorkflow.run,
            "Rome",
            id=f"da-hitl-{uuid.uuid4()}",
            task_queue="da-hitl",
        )

        # Wait for the agent to hit the interrupt (surfaced via the Query), then
        # approve via an Update. A bounded poll with a sleep, so a regression that
        # never raises the interrupt fails fast instead of busy-spinning.
        for _ in range(100):
            if await handle.query(HitlWorkflow.pending_interrupt) is not None:
                break
            await asyncio.sleep(0.1)
        else:
            pytest.fail("workflow never surfaced the HITL interrupt via the query")

        await handle.execute_update(HitlWorkflow.resume, "approve")
        out = await handle.result()

    assert "Rome" in out


# ---------------------------------------------------------------------------
# Per-call approval while parallel sub-agents run
# ---------------------------------------------------------------------------
#
# interrupt_on pauses the graph at the end of the step, so a sub-agent's
# approval request is invisible until every sibling task call finishes.
# ToolApprovals holds only the gated call, in the workflow.


class _Lane:
    """What the test's Activities did, and a switch that holds the sibling."""

    def __init__(self) -> None:
        self.lookup_started = threading.Event()
        self.lookup_released = threading.Event()
        self.published: list[str] = []

    def reset(self) -> None:
        self.lookup_started.clear()
        self.lookup_released.clear()
        self.published.clear()


_LANE = _Lane()


@activity.defn
async def publish_note(note: str) -> str:
    """Publish a note. Irreversible."""
    _LANE.published.append(note)
    return f"published {note}"


@activity.defn
async def slow_lookup(query: str) -> str:
    """Look something up; takes as long as the test wants."""
    _LANE.lookup_started.set()
    while not _LANE.lookup_released.is_set():
        await asyncio.sleep(0.05)
    return f"found {query}"


def _call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id}


class _ScriptedModels:
    """One script per model name; a call past the end of a script fails."""

    def __init__(self, scripts: dict[str, list[Any]]) -> None:
        self.scripts = scripts
        self.calls: dict[str, int] = {}

    def __call__(self, name: str) -> FakeModel:
        i = self.calls.get(name, 0)
        self.calls[name] = i + 1
        return FakeModel([self.scripts[name][i]])


def _two_lanes() -> _ScriptedModels:
    tasks = AIMessage(
        content="",
        tool_calls=[
            _call("task", {"description": "Publish x.", "subagent_type": "pub"}, "t-1"),
            _call("task", {"description": "Look up y.", "subagent_type": "web"}, "t-2"),
        ],
    )
    return _ScriptedModels(
        {
            "fake:main": [tasks, "Both done."],
            "fake:pub": [
                AIMessage(
                    content="",
                    tool_calls=[_call("publish_note", {"note": "x"}, "pub-1")],
                ),
                "Pub done.",
            ],
            "fake:web": [
                AIMessage(
                    content="",
                    tool_calls=[_call("slow_lookup", {"query": "y"}, "web-1")],
                ),
                "Web done.",
            ],
        }
    )


_TIMEOUT = timedelta(seconds=30)


@workflow.defn
class ParallelApprovalWorkflow:
    def __init__(self) -> None:
        self.approvals = ToolApprovals()

    @workflow.run
    async def run(self) -> list[str]:
        agent = create_deep_agent(
            model="fake:main",
            subagents=[
                {
                    "name": "pub",
                    "description": "Publishes.",
                    "system_prompt": "Publish.",
                    "model": "fake:pub",
                    "tools": [
                        activity_as_tool(publish_note, start_to_close_timeout=_TIMEOUT)
                    ],
                    "middleware": [self.approvals.middleware(["publish_note"])],
                },
                {
                    "name": "web",
                    "description": "Looks up.",
                    "system_prompt": "Look up.",
                    "model": "fake:web",
                    "tools": [
                        activity_as_tool(slow_lookup, start_to_close_timeout=_TIMEOUT)
                    ],
                },
            ],
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Publish x and look up y."}]}
        )
        return [str(m.content) for m in result["messages"] if m.type == "tool"]

    @workflow.query
    def pending_approvals(self) -> list[dict[str, Any]]:
        return self.approvals.pending_approvals()

    @workflow.update
    def decide(self, request_id: str, approved: bool) -> None:
        self.approvals.decide(request_id, approved, None if approved else "not now")

    @decide.validator
    def check_decide(self, request_id: str, approved: bool) -> None:
        del approved  # a validator takes the Update's arguments
        self.approvals.validate_decision(request_id)


async def _eventually(check: Any, what: str) -> Any:
    for _ in range(300):
        value = await check()
        if value:
            return value
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_approval_is_pending_while_a_sibling_subagent_runs(
    env: WorkflowEnvironment, approved: bool
) -> None:
    pytest.importorskip("deepagents")
    _LANE.reset()
    models = _two_lanes()
    try:
        async with Worker(
            env.client,
            task_queue="da-parallel-approval",
            workflows=[ParallelApprovalWorkflow],
            activities=[publish_note, slow_lookup],
            plugins=[DeepAgentsPlugin(model_provider=models)],
        ):
            handle = await env.client.start_workflow(
                ParallelApprovalWorkflow.run,
                id=f"da-parallel-approval-{uuid.uuid4()}",
                task_queue="da-parallel-approval",
            )

            async def pending() -> list[dict[str, Any]]:
                return await handle.query(ParallelApprovalWorkflow.pending_approvals)

            async def lookup_started() -> bool:
                return _LANE.lookup_started.is_set()

            # The request is visible while the sibling's lookup is running.
            assert await _eventually(pending, "the pending request") == [
                {"request_id": "pub-1", "tool": "publish_note", "args": {"note": "x"}}
            ]
            await _eventually(lookup_started, "the sibling's lookup")
            with pytest.raises(Exception):
                await handle.execute_update(
                    ParallelApprovalWorkflow.decide, args=["pub-9", True]
                )
            await handle.execute_update(
                ParallelApprovalWorkflow.decide, args=["pub-1", approved]
            )

            # The gated lane finishes while the sibling is still held.
            async def pub_finished() -> bool:
                return models.calls.get("fake:pub", 0) == 2

            await _eventually(pub_finished, "the gated lane to finish")
            assert not _LANE.lookup_released.is_set()
            assert _LANE.published == (["x"] if approved else [])
            _LANE.lookup_released.set()
            assert sorted(await handle.result()) == ["Pub done.", "Web done."]
            history = await handle.fetch_history()
    finally:
        _LANE.lookup_released.set()

    counts = await count_scheduled_activities(handle)
    assert counts["publish_note"] == (1 if approved else 0)
    assert counts["slow_lookup"] == 1
    await Replayer(
        workflows=[ParallelApprovalWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=_two_lanes())],
    ).replay_workflow(history)


@workflow.defn
class TwoGatedCallsWorkflow:
    def __init__(self) -> None:
        self.approvals = ToolApprovals()

    @workflow.run
    async def run(self) -> list[str]:
        agent = create_deep_agent(
            model="fake:main",
            tools=[activity_as_tool(publish_note, start_to_close_timeout=_TIMEOUT)],
            middleware=[self.approvals.middleware(["publish_note"])],
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Publish both."}]}
        )
        return [str(m.content) for m in result["messages"] if m.type == "tool"]

    @workflow.query
    def pending_approvals(self) -> list[dict[str, Any]]:
        return self.approvals.pending_approvals()

    @workflow.update
    def decide(self, request_id: str, approved: bool) -> None:
        self.approvals.decide(request_id, approved)

    @decide.validator
    def check_decide(self, request_id: str, approved: bool) -> None:
        del approved  # a validator takes the Update's arguments
        self.approvals.validate_decision(request_id)


def _two_publishes() -> _ScriptedModels:
    calls = [
        _call("publish_note", {"note": "one"}, "p-1"),
        _call("publish_note", {"note": "two"}, "p-2"),
    ]
    return _ScriptedModels(
        {"fake:main": [AIMessage(content="", tool_calls=calls), "Done."]}
    )


@pytest.mark.asyncio
async def test_two_gated_calls_are_decided_by_their_tool_call_ids(
    env: WorkflowEnvironment,
) -> None:
    """Two gated calls in one message, decided in reverse order."""
    pytest.importorskip("deepagents")
    _LANE.reset()
    async with Worker(
        env.client,
        task_queue="da-two-gated",
        workflows=[TwoGatedCallsWorkflow],
        activities=[publish_note],
        plugins=[DeepAgentsPlugin(model_provider=_two_publishes())],
    ):
        handle = await env.client.start_workflow(
            TwoGatedCallsWorkflow.run,
            id=f"da-two-gated-{uuid.uuid4()}",
            task_queue="da-two-gated",
        )

        async def both_pending() -> list[str]:
            pending = await handle.query(TwoGatedCallsWorkflow.pending_approvals)
            ids = sorted(p["request_id"] for p in pending)
            return ids if len(ids) == 2 else []

        assert await _eventually(both_pending, "both requests") == ["p-1", "p-2"]
        await handle.execute_update(TwoGatedCallsWorkflow.decide, args=["p-2", False])
        await handle.execute_update(TwoGatedCallsWorkflow.decide, args=["p-1", True])
        results = await handle.result()
        history = await handle.fetch_history()
    assert _LANE.published == ["one"]
    assert results == [
        "published one",
        "The call to publish_note was rejected. It did not run.",
    ]
    await Replayer(
        workflows=[TwoGatedCallsWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=_two_publishes())],
    ).replay_workflow(history)


@pytest.mark.asyncio
async def test_cancelling_a_pending_approval_cancels_the_run(
    env: WorkflowEnvironment,
) -> None:
    pytest.importorskip("deepagents")
    async with Worker(
        env.client,
        task_queue="da-gated-cancel",
        workflows=[TwoGatedCallsWorkflow],
        activities=[publish_note],
        plugins=[DeepAgentsPlugin(model_provider=_two_publishes())],
    ):
        handle = await env.client.start_workflow(
            TwoGatedCallsWorkflow.run,
            id=f"da-gated-cancel-{uuid.uuid4()}",
            task_queue="da-gated-cancel",
        )

        async def pending() -> list[dict[str, Any]]:
            return await handle.query(TwoGatedCallsWorkflow.pending_approvals)

        await _eventually(pending, "the pending requests")
        await handle.cancel()
        with pytest.raises(Exception):
            await handle.result()
        assert (await handle.describe()).status == WorkflowExecutionStatus.CANCELED


def test_middleware_refuses_tool_names_that_hold_nothing() -> None:
    approvals = ToolApprovals()
    with pytest.raises(TypeError):
        approvals.middleware("publish_note")
    with pytest.raises(ValueError):
        approvals.middleware([])


_NOTE = {"file_path": "/notes.md", "content": "Draft."}


def _gated_write() -> _ScriptedModels:
    return _ScriptedModels(
        {
            "fake:main": [
                AIMessage(
                    content="",
                    tool_calls=[_call("write_file", _NOTE, "write-1")],
                ),
                "Written.",
            ]
        }
    )


@workflow.defn
class GatedBuiltinWorkflow:
    def __init__(self) -> None:
        self.approvals = ToolApprovals()

    @workflow.run
    async def run(self) -> dict[str, Any]:
        agent = create_deep_agent(
            model="fake:main",
            middleware=[self.approvals.middleware(["write_file"])],
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Write a note."}]}
        )
        return {
            "files": sorted(result.get("files") or {}),
            "tool": [str(m.content) for m in result["messages"] if m.type == "tool"],
        }

    @workflow.query
    def pending_approvals(self) -> list[dict[str, Any]]:
        return self.approvals.pending_approvals()

    @workflow.update
    def decide(self, request_id: str, approved: bool) -> None:
        self.approvals.decide(request_id, approved, None if approved else "not now")

    @decide.validator
    def check_decide(self, request_id: str, approved: bool) -> None:
        del approved  # a validator takes the Update's arguments
        self.approvals.validate_decision(request_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_gated_builtin_tool_keeps_its_state_update(
    env: WorkflowEnvironment, approved: bool
) -> None:
    """A gated built-in tool that returns a Command updates state only if approved.

    Deep Agents' write_file returns a Command that writes the file into state.
    """
    pytest.importorskip("deepagents")
    async with Worker(
        env.client,
        task_queue="da-gated-builtin",
        workflows=[GatedBuiltinWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=_gated_write())],
    ):
        handle = await env.client.start_workflow(
            GatedBuiltinWorkflow.run,
            id=f"da-gated-builtin-{uuid.uuid4()}",
            task_queue="da-gated-builtin",
        )

        async def pending() -> list[dict[str, Any]]:
            return await handle.query(GatedBuiltinWorkflow.pending_approvals)

        assert await _eventually(pending, "the pending request") == [
            {"request_id": "write-1", "tool": "write_file", "args": _NOTE}
        ]
        await handle.execute_update(
            GatedBuiltinWorkflow.decide, args=["write-1", approved]
        )
        out = await handle.result()
        history = await handle.fetch_history()

    if approved:
        # The tool's Command reached the graph: state holds the file.
        assert out["files"] == ["/notes.md"]
    else:
        assert out["files"] == []
        assert len(out["tool"]) == 1 and "rejected" in out["tool"][0]
    await Replayer(
        workflows=[GatedBuiltinWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=_gated_write())],
    ).replay_workflow(history)
