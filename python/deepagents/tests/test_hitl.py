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
import threading
import uuid
from datetime import timedelta
from typing import Any

import pytest

from temporalio.testing import WorkflowEnvironment

pytest.importorskip("deepagents")
pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from temporalio import activity, workflow  # noqa: E402
from temporalio.client import WorkflowExecutionStatus  # noqa: E402
from temporalio.worker import Replayer, Worker  # noqa: E402
from tests.helpers.deepagents import count_scheduled_activities  # noqa: E402

# Resolve framework symbols through the imported module to avoid confusing
# the top-level package with this same-named test directory.
create_deep_agent = pytest.importorskip("deepagents").create_deep_agent

with workflow.unsafe.imports_passed_through():
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langchain.agents.middleware.human_in_the_loop import (
        HITLRequest,
        HITLResponse,
    )
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from temporalio.deepagents import (
        DeepAgentsPlugin,
        activity_as_tool,
        create_temporal_deep_agent,
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
# approval request is invisible until every sibling task call finishes. With
# approval_handler, interrupt_on is still the policy, but each guarded call
# waits for the handler on its own.


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
_APPROVE_OR_REJECT: dict[str, Any] = {"allowed_decisions": ["approve", "reject"]}


class _Approvals:
    """The workflow side of approval_handler: requests for a Query, decisions from an Update."""

    def __init__(self) -> None:
        self.pending: dict[str, HITLRequest] = {}
        self.responses: dict[str, HITLResponse] = {}

    async def review_tool_call(
        self, request_id: str, request: HITLRequest
    ) -> HITLResponse:
        self.pending[request_id] = request
        try:
            await workflow.wait_condition(lambda: request_id in self.responses)
            return self.responses.pop(request_id)
        finally:
            self.pending.pop(request_id, None)

    def validate(self, request_id: str, response: HITLResponse) -> None:
        if request_id not in self.pending or request_id in self.responses:
            raise ValueError(f"No pending tool call {request_id!r}.")
        allowed = self.pending[request_id]["review_configs"][0]["allowed_decisions"]
        if [d["type"] for d in response["decisions"]] not in ([t] for t in allowed):
            raise ValueError(f"Expected one decision out of {allowed}.")


@workflow.defn
class ParallelApprovalWorkflow:
    def __init__(self) -> None:
        self.approvals = _Approvals()

    @workflow.run
    async def run(self, override: dict[str, Any] | None) -> list[str]:
        pub: dict[str, Any] = {
            "name": "pub",
            "description": "Publishes.",
            "system_prompt": "Publish.",
            "model": "fake:pub",
            "tools": [activity_as_tool(publish_note, start_to_close_timeout=_TIMEOUT)],
        }
        if override is not None:
            pub["interrupt_on"] = override
        agent = create_temporal_deep_agent(
            model="fake:main",
            subagents=[
                pub,
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
            # Declared once for the agent; the "pub" sub-agent inherits it.
            interrupt_on={"publish_note": _APPROVE_OR_REJECT},
            approval_handler=self.approvals.review_tool_call,
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Publish x and look up y."}]}
        )
        return [str(m.content) for m in result["messages"] if m.type == "tool"]

    @workflow.query
    def pending_approvals(self) -> dict[str, HITLRequest]:
        return self.approvals.pending

    @workflow.update
    def decide(self, request_id: str, response: HITLResponse) -> None:
        self.approvals.responses[request_id] = response

    @decide.validator
    def check_decide(self, request_id: str, response: HITLResponse) -> None:
        self.approvals.validate(request_id, response)


async def _eventually(check: Any, what: str) -> Any:
    for _ in range(300):
        value = await check()
        if value:
            return value
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _decision(approved: bool) -> HITLResponse:
    if approved:
        return {"decisions": [{"type": "approve"}]}
    return {"decisions": [{"type": "reject", "message": "not now"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_approval_is_pending_while_a_sibling_subagent_runs(
    env: WorkflowEnvironment, approved: bool
) -> None:
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
                None,
                id=f"da-parallel-approval-{uuid.uuid4()}",
                task_queue="da-parallel-approval",
            )

            async def pending() -> dict[str, Any]:
                return await handle.query(ParallelApprovalWorkflow.pending_approvals)

            async def lookup_started() -> bool:
                return _LANE.lookup_started.is_set()

            # The native request, for the one call, is visible while the
            # sibling's lookup is running.
            requests = await _eventually(pending, "the pending request")
            assert list(requests) == ["pub-1"]
            request = requests["pub-1"]
            assert [(a["name"], a["args"]) for a in request["action_requests"]] == [
                ("publish_note", {"note": "x"})
            ]
            assert request["review_configs"] == [
                {
                    "action_name": "publish_note",
                    "allowed_decisions": ["approve", "reject"],
                }
            ]
            await _eventually(lookup_started, "the sibling's lookup")
            # The validator refuses an unknown id and a decision the policy
            # does not allow.
            with pytest.raises(Exception):
                await handle.execute_update(
                    ParallelApprovalWorkflow.decide, args=["pub-9", _decision(True)]
                )
            with pytest.raises(Exception):
                await handle.execute_update(
                    ParallelApprovalWorkflow.decide,
                    args=[
                        "pub-1",
                        {"decisions": [{"type": "respond", "message": "x"}]},
                    ],
                )
            await handle.execute_update(
                ParallelApprovalWorkflow.decide, args=["pub-1", _decision(approved)]
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


@pytest.mark.asyncio
async def test_subagent_interrupt_on_overrides_the_inherited_policy(
    env: WorkflowEnvironment,
) -> None:
    """A sub-agent's own interrupt_on replaces the agent's, as without a handler."""
    _LANE.reset()
    _LANE.lookup_released.set()
    async with Worker(
        env.client,
        task_queue="da-approval-override",
        workflows=[ParallelApprovalWorkflow],
        activities=[publish_note, slow_lookup],
        plugins=[DeepAgentsPlugin(model_provider=_two_lanes())],
    ):
        out = await env.client.execute_workflow(
            ParallelApprovalWorkflow.run,
            {"publish_note": False},
            id=f"da-approval-override-{uuid.uuid4()}",
            task_queue="da-approval-override",
            execution_timeout=_TIMEOUT,
        )
    assert sorted(out) == ["Pub done.", "Web done."]
    assert _LANE.published == ["x"]


@workflow.defn
class GatedCallsWorkflow:
    """One agent with interrupt_on for publish_note, skipping notes marked free."""

    def __init__(self) -> None:
        self.approvals = _Approvals()

    @workflow.run
    async def run(self) -> list[str]:
        agent = create_temporal_deep_agent(
            model="fake:main",
            tools=[activity_as_tool(publish_note, start_to_close_timeout=_TIMEOUT)],
            interrupt_on={
                "publish_note": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "Publishing is irreversible.",
                    "when": lambda req: req.tool_call["args"]["note"] != "free",
                }
            },
            approval_handler=self.approvals.review_tool_call,
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Publish all three."}]}
        )
        return [str(m.content) for m in result["messages"] if m.type == "tool"]

    @workflow.query
    def pending_approvals(self) -> dict[str, HITLRequest]:
        return self.approvals.pending

    @workflow.update
    def decide(self, request_id: str, response: HITLResponse) -> None:
        self.approvals.responses[request_id] = response

    @decide.validator
    def check_decide(self, request_id: str, response: HITLResponse) -> None:
        self.approvals.validate(request_id, response)


def _three_publishes() -> _ScriptedModels:
    calls = [
        _call("publish_note", {"note": "one"}, "p-1"),
        _call("publish_note", {"note": "two"}, "p-2"),
        _call("publish_note", {"note": "free"}, "p-3"),
    ]
    return _ScriptedModels(
        {"fake:main": [AIMessage(content="", tool_calls=calls), "Done."]}
    )


@pytest.mark.asyncio
async def test_guarded_calls_are_decided_one_by_one(
    env: WorkflowEnvironment,
) -> None:
    """Calls in one message are decided by tool call id, in any order; the
    `when` predicate lets a call through without asking, and an edit runs the
    reviewer's arguments."""
    _LANE.reset()
    async with Worker(
        env.client,
        task_queue="da-gated-calls",
        workflows=[GatedCallsWorkflow],
        activities=[publish_note],
        plugins=[DeepAgentsPlugin(model_provider=_three_publishes())],
    ):
        handle = await env.client.start_workflow(
            GatedCallsWorkflow.run,
            id=f"da-gated-calls-{uuid.uuid4()}",
            task_queue="da-gated-calls",
        )

        async def both_pending() -> dict[str, Any]:
            pending = await handle.query(GatedCallsWorkflow.pending_approvals)
            return pending if len(pending) == 2 else {}

        pending = await _eventually(both_pending, "both requests")
        assert sorted(pending) == ["p-1", "p-2"]
        assert (
            pending["p-1"]["action_requests"][0]["description"]
            == "Publishing is irreversible."
        )

        async def free_published() -> bool:
            return _LANE.published == ["free"]

        # The call the predicate skips ran without waiting.
        await _eventually(free_published, "the unguarded call")
        await handle.execute_update(
            GatedCallsWorkflow.decide,
            args=["p-2", {"decisions": [{"type": "reject"}]}],
        )
        await handle.execute_update(
            GatedCallsWorkflow.decide,
            args=[
                "p-1",
                {
                    "decisions": [
                        {
                            "type": "edit",
                            "edited_action": {
                                "name": "publish_note",
                                "args": {"note": "ONE"},
                            },
                        }
                    ]
                },
            ],
        )
        results = await handle.result()
        history = await handle.fetch_history()
    assert _LANE.published == ["free", "ONE"]
    assert len(results) == 3
    assert results[0].endswith("published ONE") and "reviewer" in results[0]
    assert "User rejected the tool call for `publish_note`" in results[1]
    assert results[2] == "published free"
    await Replayer(
        workflows=[GatedCallsWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=_three_publishes())],
    ).replay_workflow(history)


@pytest.mark.asyncio
async def test_cancelling_a_pending_approval_cancels_the_run(
    env: WorkflowEnvironment,
) -> None:
    async with Worker(
        env.client,
        task_queue="da-gated-cancel",
        workflows=[GatedCallsWorkflow],
        activities=[publish_note],
        plugins=[DeepAgentsPlugin(model_provider=_three_publishes())],
    ):
        handle = await env.client.start_workflow(
            GatedCallsWorkflow.run,
            id=f"da-gated-cancel-{uuid.uuid4()}",
            task_queue="da-gated-cancel",
        )

        async def pending() -> dict[str, Any]:
            return await handle.query(GatedCallsWorkflow.pending_approvals)

        await _eventually(pending, "the pending requests")
        await handle.cancel()
        with pytest.raises(Exception):
            await handle.result()
        assert (await handle.describe()).status == WorkflowExecutionStatus.CANCELED


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
        self.approvals = _Approvals()

    @workflow.run
    async def run(self) -> dict[str, Any]:
        agent = create_temporal_deep_agent(
            model="fake:main",
            interrupt_on={"write_file": True},
            approval_handler=self.approvals.review_tool_call,
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Write a note."}]}
        )
        return {
            "files": sorted(result.get("files") or {}),
            "tool": [str(m.content) for m in result["messages"] if m.type == "tool"],
        }

    @workflow.query
    def pending_approvals(self) -> dict[str, HITLRequest]:
        return self.approvals.pending

    @workflow.update
    def decide(self, request_id: str, response: HITLResponse) -> None:
        self.approvals.responses[request_id] = response

    @decide.validator
    def check_decide(self, request_id: str, response: HITLResponse) -> None:
        self.approvals.validate(request_id, response)


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_gated_builtin_tool_keeps_its_state_update(
    env: WorkflowEnvironment, approved: bool
) -> None:
    """A guarded built-in tool that returns a Command updates state only if approved.

    Deep Agents' write_file returns a Command that writes the file into state.
    """
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

        async def pending() -> dict[str, Any]:
            return await handle.query(GatedBuiltinWorkflow.pending_approvals)

        requests = await _eventually(pending, "the pending request")
        assert requests["write-1"]["action_requests"][0]["args"] == _NOTE
        await handle.execute_update(
            GatedBuiltinWorkflow.decide, args=["write-1", _decision(approved)]
        )
        out = await handle.result()
        history = await handle.fetch_history()

    if approved:
        # The tool's Command reached the graph: state holds the file.
        assert out["files"] == ["/notes.md"]
    else:
        assert out["files"] == []
        assert out["tool"] == [
            "User rejected the tool call for `write_file` with reason: not now"
        ]
    await Replayer(
        workflows=[GatedBuiltinWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=_gated_write())],
    ).replay_workflow(history)


async def _never(request_id: str, request: Any) -> Any:
    raise AssertionError(f"not called, got {request_id}: {request}")


def test_approval_handler_refuses_a_hitl_middleware_it_cannot_replace() -> None:
    """A HumanInTheLoopMiddleware passed directly would still interrupt the graph."""
    hitl = HumanInTheLoopMiddleware(interrupt_on={"publish_note": True})
    with pytest.raises(ValueError, match="interrupt_on"):
        create_temporal_deep_agent(
            model="fake:main", middleware=[hitl], approval_handler=_never
        )
    with pytest.raises(ValueError, match="'pub'"):
        create_temporal_deep_agent(
            model="fake:main",
            subagents=[
                {
                    "name": "pub",
                    "description": "Publishes.",
                    "system_prompt": "Publish.",
                    "middleware": [hitl],
                }
            ],
            approval_handler=_never,
        )


def test_agents_without_a_handler_keep_the_native_middleware() -> None:
    """Outside an approval_handler call, Deep Agents builds the native middleware."""
    with_handler = create_temporal_deep_agent(
        model="fake:main", interrupt_on={"write_file": True}, approval_handler=_never
    )
    without = create_temporal_deep_agent(
        model="fake:main", interrupt_on={"write_file": True}
    )
    assert "HumanInTheLoopMiddleware.after_model" not in with_handler.get_graph().nodes
    assert "HumanInTheLoopMiddleware.after_model" in without.get_graph().nodes
