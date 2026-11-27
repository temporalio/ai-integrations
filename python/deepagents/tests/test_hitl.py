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

The second half of this file covers the identity of those interrupts:
``Interrupt.id`` derives from a checkpoint id LangGraph mints in-workflow, so
without the plugin's checkpoint-id patch the same history yields a different
id on every replay. Those tests record real interrupting runs and compare the
ids each execution printed — printing being the only channel observable from
both a live execution and its replay.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import importlib
import re
import threading
import time
import uuid
import weakref
from datetime import timedelta
from typing import Any, cast

import pytest

from temporalio.testing import WorkflowEnvironment

pytest.importorskip("deepagents")
pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from temporalio import workflow  # noqa: E402
from temporalio.api.common.v1 import WorkflowExecution  # noqa: E402
from temporalio.api.enums.v1 import EventType, WorkflowTaskFailedCause  # noqa: E402
from temporalio.api.workflowservice.v1 import (  # noqa: E402
    ResetWorkflowExecutionRequest,
)
from temporalio.worker import (  # noqa: E402
    Interceptor,
    Replayer,
    Worker,
    WorkflowInboundInterceptor,
    WorkflowInterceptorClassInput,
    WorkflowOutboundInterceptor,
)

# Resolve framework symbols through the imported module to avoid confusing
# the top-level package with this same-named test directory.
create_deep_agent = pytest.importorskip("deepagents").create_deep_agent

with workflow.unsafe.imports_passed_through():
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint import base as checkpoint_base
    from langgraph.checkpoint.base import id as checkpoint_id_module
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from temporalio.deepagents import DeepAgentsPlugin, tool_as_activity
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


# -- interrupt ids: replay-stable through the checkpoint-id patch ------------
#
# `minted-id <live|replay> <workflow id> <id>` lines, printed by the
# workflows below from both a live execution and its replay.
_ID_LINE = re.compile(r"minted-id (live|replay) (\S+) (.+)")


def _printed_ids(out: str, mode: str | None, workflow_id: str) -> list[str]:
    """Ids a workflow printed in creation order; duplicates preserved.

    ``mode=None`` returns both live and replay prints, in the order they were
    printed.
    """
    return [
        i
        for m, w, i in _ID_LINE.findall(out)
        if (mode is None or m == mode) and w == workflow_id
    ]


class _ToolCountModel(FakeModel):
    """Scripted by the conversation it is shown, not a shared cursor.

    The reply depends only on how many guarded tool calls already completed,
    so concurrent runs of one worker each script themselves; a
    ``mock_model_provider`` cursor would be advanced by whichever run's
    activity happened to fire.
    """

    def __init__(self) -> None:
        """Satisfy ``FakeModel``'s script argument; ``_reply`` ignores it."""
        super().__init__(responses=["counting"])

    def _reply(self, messages: list[BaseMessage]) -> AIMessage:
        done = sum(1 for m in messages if getattr(m, "type", "") == "tool")
        if done == 0:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "publish_note", "args": {"note": "Rome"}, "id": "c1"}
                ],
            )
        if done == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "publish_note", "args": {"note": "Oslo"}, "id": "c2"}
                ],
            )
        return AIMessage(content="published both")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])


@workflow.defn
class InterruptIdsWorkflow:
    """Two guarded approvals in one run; prints every interrupt id it sees."""

    @workflow.run
    async def run(self) -> dict[str, Any]:
        def publish_note(note: str) -> str:
            """Publish a note (requires human approval)."""
            return f"published: {note}"

        agent = create_deep_agent(
            model="fake:model",
            tools=[
                tool_as_activity(
                    publish_note, start_to_close_timeout=timedelta(seconds=30)
                )
            ],
            interrupt_on={"publish_note": True},
            checkpointer=InMemorySaver(),
        )
        config: RunnableConfig = {
            "configurable": {"thread_id": workflow.info().workflow_id}
        }
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Publish Rome, then Oslo."}]},
            config=config,
        )
        mode = "replay" if workflow.unsafe.is_replaying() else "live"
        run_id = workflow.info().workflow_id
        ids: list[str] = []
        while result.get("__interrupt__"):
            iid = result["__interrupt__"][0].id
            ids.append(iid)
            print(f"minted-id {mode} {run_id} {iid}")
            result = await agent.ainvoke(
                Command(resume={"decisions": [{"type": "approve"}]}), config=config
            )
        return {"ids": ids, "reply": str(result["messages"][-1].content)}


async def _run_ids_workflow(
    env: WorkflowEnvironment, **worker_kwargs: Any
) -> tuple[dict[str, Any], Any, str]:
    """Run ``InterruptIdsWorkflow`` once; return its result, history and id."""
    workflow_id = f"da-interrupt-ids-{uuid.uuid4()}"
    task_queue = f"da-interrupt-ids-{uuid.uuid4()}"
    plugin = DeepAgentsPlugin(model_provider=lambda _: _ToolCountModel())
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[InterruptIdsWorkflow],
        plugins=[plugin],
        **worker_kwargs,
    ):
        handle = await env.client.start_workflow(
            InterruptIdsWorkflow.run, id=workflow_id, task_queue=task_queue
        )
        result = await handle.result()
        return result, await handle.fetch_history(), workflow_id


@pytest.mark.asyncio
async def test_interrupt_ids_stable_across_replay(
    env: WorkflowEnvironment, capsys: pytest.CaptureFixture[str]
) -> None:
    result, history, workflow_id = await _run_ids_workflow(env)

    # Two interrupts in the execution, each with its own id. This much also
    # holds without the patch (ids were never *reused* before, only unstable).
    assert len(result["ids"]) == 2
    assert len(set(result["ids"])) == 2
    assert result["reply"] == "published both"

    live = _printed_ids(capsys.readouterr().out, "live", workflow_id)
    assert live == result["ids"]

    # The replayed execution of the same history computed the same ids, in
    # the same order. This is the assertion that fails without the patch.
    await Replayer(
        workflows=[InterruptIdsWorkflow], plugins=[DeepAgentsPlugin()]
    ).replay_workflow(history)
    replay = _printed_ids(capsys.readouterr().out, "replay", workflow_id)
    assert replay == result["ids"]


@pytest.mark.asyncio
async def test_interrupt_ids_distinct_across_interleaved_runs(
    env: WorkflowEnvironment, capsys: pytest.CaptureFixture[str]
) -> None:
    # Two runs alive at once on one worker: their mints interleave on the
    # event loop, so the per-run tick floor must key by run.
    task_queue = f"da-interrupt-ids-{uuid.uuid4()}"
    plugin = DeepAgentsPlugin(model_provider=lambda _: _ToolCountModel())
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[InterruptIdsWorkflow],
        plugins=[plugin],
    ):
        handles = [
            await env.client.start_workflow(
                InterruptIdsWorkflow.run,
                id=f"da-interrupt-ids-{uuid.uuid4()}",
                task_queue=task_queue,
            )
            for _ in range(2)
        ]
        results = await asyncio.gather(*(h.result() for h in handles))

    # Same logical interrupts in different executions never share an id (48
    # run-derived hash bits apart; holds without the patch too)...
    assert results[0]["ids"] != results[1]["ids"]
    # ...and each run replayed alone re-derives its own ids.
    out = capsys.readouterr().out
    for handle, result in zip(handles, results):
        assert _printed_ids(out, "live", handle.id) == result["ids"]
        await Replayer(
            workflows=[InterruptIdsWorkflow], plugins=[DeepAgentsPlugin()]
        ).replay_workflow(await handle.fetch_history())
        assert (
            _printed_ids(capsys.readouterr().out, "replay", handle.id)
            == (result["ids"])
        )


@pytest.mark.asyncio
async def test_interrupt_ids_stable_across_eviction(
    env: WorkflowEnvironment, capsys: pytest.CaptureFixture[str]
) -> None:
    # max_cached_workflows=0 disables caching: every workflow task builds a
    # fresh instance and re-executes the run from its start, exactly like a
    # workflow evicted between tasks. Each re-execution re-prints the ids it
    # re-derives — usually flagged "replay", since re-deriving existing
    # history *is* replay — so every print from the run, both flags together,
    # must match the ids the completing execution returned. This is exactly
    # the property eviction puts at risk: an id the run already surfaced (or
    # recorded in a Query answer or an Update payload) drifting when a later
    # task re-derives it.
    result, history, workflow_id = await _run_ids_workflow(env, max_cached_workflows=0)
    assert result["reply"] == "published both"

    printed = _printed_ids(capsys.readouterr().out, None, workflow_id)
    assert set(printed) == set(result["ids"])
    assert list(dict.fromkeys(printed)) == result["ids"]

    await Replayer(
        workflows=[InterruptIdsWorkflow], plugins=[DeepAgentsPlugin()]
    ).replay_workflow(history)
    replay = _printed_ids(capsys.readouterr().out, "replay", workflow_id)
    assert replay == result["ids"]


@workflow.defn
class CheckpointModuleIdsWorkflow:
    """Mint ids through the other patched checkpoint modules directly.

    ``InterruptIdsWorkflow`` exercises the mint inside the agent loop
    (``langgraph.pregel._checkpoint``); this one calls the defining module
    and the saver-side holder (``langgraph.checkpoint.base``), whose own
    ``uuid6`` bindings the patch must also swap. The saver-side *helpers*
    (``empty_checkpoint`` / ``create_checkpoint``) cannot run in-workflow at
    all — their ``datetime.now()`` call is blocked by the sandbox's time
    restrictions — so the binding is exercised directly.
    """

    @workflow.run
    async def run(self) -> list[str]:
        mode = "replay" if workflow.unsafe.is_replaying() else "live"
        run_id = workflow.info().workflow_id
        ids = [
            str(checkpoint_id_module.uuid6()),
            # Imported-not-re-exported per pyright, but the runtime binding is
            # exactly the seam this patch swaps.
            str(checkpoint_base.uuid6(clock_seq=-2)),  # pyright: ignore[reportPrivateImportUsage]
        ]
        for iid in ids:
            print(f"minted-id {mode} {run_id} {iid}")
        return ids


@pytest.mark.asyncio
async def test_saver_side_checkpoint_ids_stable_across_replay(
    env: WorkflowEnvironment, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow_id = f"da-checkpoint-ids-{uuid.uuid4()}"
    task_queue = f"da-checkpoint-ids-{uuid.uuid4()}"
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[CheckpointModuleIdsWorkflow],
        plugins=[DeepAgentsPlugin()],
    ):
        handle = await env.client.start_workflow(
            CheckpointModuleIdsWorkflow.run, id=workflow_id, task_queue=task_queue
        )
        result = await handle.result()
        history = await handle.fetch_history()

    assert len(set(result)) == 2
    assert _printed_ids(capsys.readouterr().out, "live", workflow_id) == result

    # Ordering next to unpatched ids: the tick field carries the workflow's
    # real (recorded) time in uuid6's own epoch, so a minted id lands in the
    # same time range as a wall-clock uuid6 minted around the same moment —
    # a small or shifted tick here is exactly what would make patched ids
    # sort away from unpatched ones. Five minutes of slack, in 100-ns ticks.
    wall = checkpoint_id_module.uuid6().time
    minted = checkpoint_id_module.UUID(result[0]).time
    assert abs(wall - minted) < 5 * 60 * 10**7

    await Replayer(
        workflows=[CheckpointModuleIdsWorkflow], plugins=[DeepAgentsPlugin()]
    ).replay_workflow(history)
    replay = _printed_ids(capsys.readouterr().out, "replay", workflow_id)
    assert replay == result


def test_uuid6_packing_orders_across_version_nibble() -> None:
    # Unit guard for the one property the string ids rely on: for a fixed
    # node / clock_seq, ids must sort as strings exactly by tick — including
    # where the tick's low 12 bits carry into the high field — and must keep
    # uuid6's field layout. Imported lazily so the behavioral tests above
    # keep failing on unpatched builds for their own reason.
    # Mint through LangGraph's bundled UUID class, the same one the source
    # mints through: the stdlib constructor rejects version=6 before 3.14.
    from langgraph.checkpoint.base.id import UUID as langgraph_uuid

    from temporalio.deepagents import _checkpoint_ids

    node, clock_seq = 0x0123456789AB, 0x2AC
    ticks = [
        1,
        2,
        0xFFE,
        0xFFF,
        0x1000,
        0x1001,
        (1 << 24) - 1,
        1 << 24,
        1 << 59,
        (1 << 60) - 1,
    ]
    values = [_checkpoint_ids._uuid6_int(t, node, clock_seq) for t in ticks]

    ids = [str(langgraph_uuid(int=v, version=6)) for v in values]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    for tick, value in zip(ticks, values):
        assert ((value >> 80) << 12) | ((value >> 64) & 0xFFF) == tick
        assert value & 0xFFFFFFFFFFFF == node
        assert (value >> 48) & 0x3FFF == clock_seq & 0x3FFF


@pytest.mark.asyncio
async def test_checkpoint_patch_inert_outside_workflows_and_restored(
    env: WorkflowEnvironment,
) -> None:
    # The patch is worker-lifetime and process-global, but outside workflow
    # code the original wall-clock uuid6 must answer (client-side LangGraph
    # in the same process), and worker shutdown must restore the module
    # attribute exactly. async-with starts the worker on a task, so the
    # patch appearing / disappearing is polled, not assumed instantaneous.
    original = checkpoint_id_module.uuid6

    async def uuid6_is_patched() -> bool:
        return checkpoint_id_module.uuid6 is not original

    async with Worker(
        env.client,
        task_queue=f"da-idle-{uuid.uuid4()}",
        workflows=[InterruptIdsWorkflow],
        plugins=[DeepAgentsPlugin()],
    ):
        for _ in range(100):
            if await uuid6_is_patched():
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("worker never installed the checkpoint-id patch")
        # Outside a workflow the binding delegates to the real uuid6: its
        # time field is wall-clock in uuid6's own epoch, within minutes of a
        # fresh original-minted id — the same comparison the saver-side test
        # makes, and one that fails for a counter, an offset-scaled, or a
        # workflow-time tick. Five minutes of slack, in 100-ns ticks.
        wall = original().time
        assert abs(checkpoint_id_module.uuid6().time - wall) < 5 * 60 * 10**7

    for _ in range(100):
        if await uuid6_is_patched():
            await asyncio.sleep(0.05)
        else:
            break
    else:
        pytest.fail("worker shutdown never restored the original uuid6")


def test_checkpoint_uuid6_imported_while_patched_survives_uninstall() -> None:
    # A module that runs `from langgraph.checkpoint.base.id import uuid6`
    # while a worker is up keeps the patched function after the last worker
    # shuts down; restoring the module attributes cannot reach that
    # reference. It must keep delegating to the real uuid6, not to a cleared
    # global (TypeError: 'NoneType' object is not callable).
    from temporalio.deepagents import _checkpoint_ids

    original = checkpoint_id_module.uuid6
    _checkpoint_ids.install_checkpoint_id_patch()
    try:
        imported = checkpoint_id_module.uuid6
        assert imported is not original
    finally:
        _checkpoint_ids.uninstall_checkpoint_id_patch()
    assert checkpoint_id_module.uuid6 is original

    # Wall-clock time in uuid6's epoch, within five minutes (100-ns ticks).
    wall = original().time
    assert abs(imported().time - wall) < 5 * 60 * 10**7


@workflow.defn(sandboxed=False)
class PlainLangGraphIdWorkflow:
    """A workflow on a Worker without DeepAgentsPlugin that mints a checkpoint id."""

    @workflow.run
    async def run(self) -> str:
        # Looked up at call time, so it is the patched binding while a
        # DeepAgentsPlugin worker is running in this process.
        return str(checkpoint_id_module.uuid6(clock_seq=0))


@pytest.mark.asyncio
async def test_checkpoint_patch_leaves_plugin_free_workers_alone(
    env: WorkflowEnvironment,
) -> None:
    # The patch is process-wide, but per-instance state exists only where the
    # plugin's CheckpointIdInterceptor creates it. A Worker running LangGraph
    # without the plugin has no such interceptor, so its runs must keep real
    # uuid6, completed and evicted runs alike (max_cached_workflows=0 evicts
    # after every task).
    from temporalio.deepagents import _checkpoint_ids

    original = checkpoint_id_module.uuid6
    async with Worker(
        env.client,
        task_queue=f"da-idle-{uuid.uuid4()}",
        workflows=[InterruptIdsWorkflow],
        plugins=[DeepAgentsPlugin()],
    ):
        for _ in range(100):
            if checkpoint_id_module.uuid6 is not original:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("worker never installed the checkpoint-id patch")

        plain_queue = f"plain-langgraph-{uuid.uuid4()}"
        run_ids = []
        async with Worker(
            env.client,
            task_queue=plain_queue,
            workflows=[PlainLangGraphIdWorkflow],
            max_cached_workflows=0,
        ):
            for _ in range(3):
                handle = await env.client.start_workflow(
                    PlainLangGraphIdWorkflow.run,
                    id=f"plain-langgraph-{uuid.uuid4()}",
                    task_queue=plain_queue,
                )
                minted = await handle.result()
                run_ids.append(handle.result_run_id)
                # Real uuid6: wall-clock time, and not the run-derived node
                # the replay-stable drop-in would have used.
                wall = original().time
                assert (
                    abs(checkpoint_id_module.UUID(minted).time - wall) < 5 * 60 * 10**7
                )
                run_node, _ = _checkpoint_ids._run_id_bits(handle.result_run_id or "")
                assert checkpoint_id_module.UUID(minted).node != run_node


@workflow.defn(sandboxed=False)
class TwoMintsAroundSignalWorkflow:
    """Mints a checkpoint id, waits for a signal, mints again."""

    def __init__(self) -> None:
        self._go = False

    @workflow.signal
    def go(self) -> None:
        self._go = True

    @workflow.run
    async def run(self) -> list[str]:
        first = str(checkpoint_id_module.uuid6(clock_seq=0))
        await workflow.wait_condition(lambda: self._go)
        return [first, str(checkpoint_id_module.uuid6(clock_seq=0))]


@pytest.mark.asyncio
async def test_checkpoint_state_is_per_instance_not_per_run(
    env: WorkflowEnvironment,
) -> None:
    # Two instances of one run can be alive in one process (a handoff between
    # Workers after a workflow-task timeout, or a Replayer). When the other
    # instance is evicted, the surviving one must keep its tick floor and
    # keep minting replay-stable ids; with state keyed by run id the eviction
    # deleted its entry and later ids fell back to wall-clock/random.
    from temporalio.deepagents import _checkpoint_ids

    task_queue = f"da-two-instances-{uuid.uuid4()}"
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[TwoMintsAroundSignalWorkflow],
        plugins=[DeepAgentsPlugin()],
    ):
        handle = await env.client.start_workflow(
            TwoMintsAroundSignalWorkflow.run,
            id=f"da-two-instances-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        await asyncio.sleep(0.5)
        history = await handle.fetch_history()
        run_node, _ = _checkpoint_ids._run_id_bits(handle.result_run_id or "")

        # A second instance of the same run starts and is then evicted.
        await Replayer(
            workflows=[TwoMintsAroundSignalWorkflow], plugins=[DeepAgentsPlugin()]
        ).replay_workflow(history)

        await handle.signal(TwoMintsAroundSignalWorkflow.go)
        first, second = await handle.result()

    assert checkpoint_id_module.UUID(first).node == run_node
    assert checkpoint_id_module.UUID(second).node == run_node
    assert first < second


@pytest.mark.asyncio
async def test_failed_checkpoint_install_unwinds_earlier_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A moved or renamed LangGraph seam must fail worker startup loudly
    # (install_checkpoint_id_patch raising AttributeError) *and* leave
    # nothing behind: the seams installed before the failing one are unwound
    # on the way out, not leaked for the process lifetime. Patch-internal
    # test: it drives _run_context directly and reads the private modules,
    # so it errors at import on a source tree without the new module.
    deepagents = importlib.import_module("deepagents")
    BackendProtocol = importlib.import_module(
        "deepagents.backends.protocol"
    ).BackendProtocol
    from temporalio.deepagents import _checkpoint_ids, _model, _tools

    def moved_seam() -> None:
        raise AttributeError("simulated moved LangGraph checkpoint module")

    monkeypatch.setattr(_checkpoint_ids, "install_checkpoint_id_patch", moved_seam)

    original_create = deepagents.create_deep_agent
    original_resolve = importlib.import_module("deepagents.graph").resolve_model
    original_als = BackendProtocol.__dict__["als"]

    with pytest.raises(AttributeError, match="simulated moved"):
        async with DeepAgentsPlugin()._run_context():
            pytest.fail("run_context must not yield when a seam fails loudly")

    # The bindings installed before the failing seam are back to the exact
    # original objects, and each seam's undo state is empty again, so a later
    # install starts clean instead of stacking on leaked patches.
    assert deepagents.create_deep_agent is original_create
    assert importlib.import_module("deepagents.graph").resolve_model is original_resolve
    assert BackendProtocol.__dict__["als"] is original_als
    assert _model._original_create_deep_agent is None
    assert _model._original_resolve_model is None
    assert not _tools._original_backend_async_defaults
    assert _checkpoint_ids._original_uuid6 is None


@workflow.defn(sandboxed=False)
class ApproveByCheckpointIdWorkflow:
    """Records a decision keyed on a minted id: a timer only if the id matches."""

    def __init__(self) -> None:
        self._minted: str | None = None
        self._approved: str | None = None
        self._done = False

    @workflow.signal
    def approve(self, checkpoint_id: str) -> None:
        self._approved = checkpoint_id

    @workflow.signal
    def finish(self) -> None:
        self._done = True

    @workflow.query
    def minted(self) -> str | None:
        return self._minted

    @workflow.run
    async def run(self) -> list[str]:
        self._minted = str(checkpoint_id_module.uuid6(clock_seq=0))
        await workflow.wait_condition(lambda: self._approved is not None)
        if self._approved == self._minted:
            await workflow.sleep(timedelta(milliseconds=10))
        await workflow.wait_condition(lambda: self._done)
        return [self._minted, self._approved or ""]


async def _event_types(handle: Any) -> list[int]:
    return [e.event_type for e in (await handle.fetch_history()).events]


@pytest.mark.asyncio
async def test_checkpoint_ids_survive_reset(env: WorkflowEnvironment) -> None:
    # A reset copies the history prefix into a new run with a new run id and
    # replays it. An id minted in that prefix, and any decision already
    # recorded against it (here: a timer started only because the signalled
    # id matched), must be re-derived as it was; hashing the new run id made
    # the replayed branch skip the TimerStarted in the copied history and
    # fail the reset run with a nondeterminism error.
    from temporalio.deepagents import _checkpoint_ids

    workflow_id = f"da-reset-ids-{uuid.uuid4()}"
    task_queue = f"da-reset-ids-{uuid.uuid4()}"
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[ApproveByCheckpointIdWorkflow],
        plugins=[DeepAgentsPlugin()],
    ):
        handle = await env.client.start_workflow(
            ApproveByCheckpointIdWorkflow.run, id=workflow_id, task_queue=task_queue
        )
        for _ in range(100):
            minted = await handle.query(ApproveByCheckpointIdWorkflow.minted)
            if minted is not None:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("workflow never minted a checkpoint id")
        await handle.signal(ApproveByCheckpointIdWorkflow.approve, minted)

        # Wait for the task that handles the fired timer to complete; reset
        # to it, so the copied prefix holds the matching-id branch's timer.
        completed = EventType.EVENT_TYPE_WORKFLOW_TASK_COMPLETED
        for _ in range(100):
            types = await _event_types(handle)
            if EventType.EVENT_TYPE_TIMER_FIRED in types and types[-1] == completed:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("the approved branch never started and fired its timer")
        history = await handle.fetch_history()
        reset_to = max(e.event_id for e in history.events if e.event_type == completed)
        original_run_id = handle.result_run_id or ""

        reset = await env.client.workflow_service.reset_workflow_execution(
            ResetWorkflowExecutionRequest(
                namespace=env.client.namespace,
                workflow_execution=WorkflowExecution(
                    workflow_id=workflow_id, run_id=original_run_id
                ),
                reason="checkpoint ids survive reset",
                workflow_task_finish_event_id=reset_to,
                request_id=str(uuid.uuid4()),
            )
        )
        assert reset.run_id != original_run_id
        reset_handle = env.client.get_workflow_handle(workflow_id, run_id=reset.run_id)
        await reset_handle.signal(ApproveByCheckpointIdWorkflow.finish)
        for _ in range(200):
            types = await _event_types(reset_handle)
            # The reset itself records one failed task (cause ResetWorkflow)
            # at the reset point; any other is the replayed prefix diverging.
            assert not [
                e.workflow_task_failed_event_attributes
                for e in (await reset_handle.fetch_history()).events
                if e.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED
                and e.workflow_task_failed_event_attributes.cause
                != WorkflowTaskFailedCause.WORKFLOW_TASK_FAILED_CAUSE_RESET_WORKFLOW
            ]
            if EventType.EVENT_TYPE_WORKFLOW_EXECUTION_COMPLETED in types:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("the reset run never completed")
        result = await reset_handle.result()

    assert result == [minted, minted]
    # The namespace is the original run's, not the reset run's.
    original_node, _ = _checkpoint_ids._run_id_bits(original_run_id)
    assert checkpoint_id_module.UUID(minted).node == original_node


@workflow.defn(sandboxed=False)
class ValidatorMintWorkflow:
    """An Update whose validator and handler both mint a checkpoint id."""

    def __init__(self) -> None:
        self._done = False

    @workflow.update
    def mint(self) -> str:
        minted = str(checkpoint_id_module.uuid6(clock_seq=0))
        mode = "replay" if workflow.unsafe.is_replaying() else "live"
        print(f"minted-id {mode} {workflow.info().workflow_id} {minted}")
        self._done = True
        return minted

    @mint.validator
    def validate_mint(self) -> None:
        # Temporal skips validators on replay, so this mint must not move
        # the floor the handler's id is derived from.
        checkpoint_id_module.uuid6(clock_seq=0)

    @workflow.run
    async def run(self) -> None:
        await workflow.wait_condition(lambda: self._done)


@pytest.mark.asyncio
async def test_update_validator_mint_does_not_shift_replayed_ids(
    env: WorkflowEnvironment, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow_id = f"da-validator-ids-{uuid.uuid4()}"
    task_queue = f"da-validator-ids-{uuid.uuid4()}"
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[ValidatorMintWorkflow],
        plugins=[DeepAgentsPlugin()],
    ):
        handle = await env.client.start_workflow(
            ValidatorMintWorkflow.run, id=workflow_id, task_queue=task_queue
        )
        minted = await handle.execute_update(ValidatorMintWorkflow.mint)
        await handle.result()
        history = await handle.fetch_history()

    assert _printed_ids(capsys.readouterr().out, "live", workflow_id) == [minted]
    await Replayer(
        workflows=[ValidatorMintWorkflow], plugins=[DeepAgentsPlugin()]
    ).replay_workflow(history)
    assert _printed_ids(capsys.readouterr().out, "replay", workflow_id) == [minted]


def test_checkpoint_state_lives_and_dies_with_the_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The tick floor is created with the instance, before the SDK decodes
    # the start arguments, and a start payload that fails to decode evicts
    # the instance without ever entering the run method. So the state must
    # be owned by the instance's inbound interceptor, not by a module-level
    # map that only run-method cleanup empties: dropping the interceptor
    # must free it, whatever the run method did or did not do.
    from temporalio.deepagents import _checkpoint_ids

    alive: weakref.WeakSet[Any] = weakref.WeakSet()

    class TrackedState(_checkpoint_ids._InstanceState):
        def __init__(self) -> None:
            super().__init__()
            alive.add(self)

    monkeypatch.setattr(_checkpoint_ids, "_InstanceState", TrackedState)
    interceptor = _checkpoint_ids.CheckpointIdInterceptor()
    inbound_class = interceptor.workflow_interceptor_class(cast(Any, None))
    inbound = inbound_class(cast(Any, object()))
    assert len(alive) == 1
    del inbound
    gc.collect()
    assert not alive


class _ReplacedInfoInterceptor(Interceptor):
    """Returns a fresh ``Info`` copy from every ``workflow.info()`` call."""

    def workflow_interceptor_class(
        self, input: WorkflowInterceptorClassInput
    ) -> type[WorkflowInboundInterceptor]:
        class Outbound(WorkflowOutboundInterceptor):
            def info(self) -> workflow.Info:
                return dataclasses.replace(super().info())

        class Inbound(WorkflowInboundInterceptor):
            def init(self, outbound: WorkflowOutboundInterceptor) -> None:
                super().init(Outbound(outbound))

        return Inbound


@pytest.mark.asyncio
async def test_checkpoint_ids_with_interceptor_replacing_info(
    env: WorkflowEnvironment,
) -> None:
    # The instance state must not be found through the identity of the
    # ``workflow.info()`` result: an outbound interceptor may return a new
    # object on every call, which made the patch fall back to wall-clock,
    # random-node uuid6 for the whole run.
    from temporalio.deepagents import _checkpoint_ids

    task_queue = f"da-replaced-info-{uuid.uuid4()}"
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[PlainLangGraphIdWorkflow],
        plugins=[DeepAgentsPlugin()],
        interceptors=[_ReplacedInfoInterceptor()],
    ):
        handle = await env.client.start_workflow(
            PlainLangGraphIdWorkflow.run,
            id=f"da-replaced-info-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        minted = await handle.result()

    run_node, _ = _checkpoint_ids._run_id_bits(handle.result_run_id or "")
    assert checkpoint_id_module.UUID(minted).node == run_node


def test_concurrent_checkpoint_installs_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two workers starting on separate threads: the second install must wait
    # for the first instead of also seeing "unpatched", so both are counted
    # and the saved original is LangGraph's function, not the first wrapper.
    # The first install is held inside its LangGraph import while the second
    # starts.
    from temporalio.deepagents import _checkpoint_ids

    original = checkpoint_id_module.uuid6
    real_import = importlib.import_module
    first_importing = threading.Event()

    def slow_import(name: str, package: str | None = None) -> Any:
        if name == "langgraph.checkpoint.base.id" and not first_importing.is_set():
            first_importing.set()
            time.sleep(0.3)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", slow_import)
    first = threading.Thread(target=_checkpoint_ids.install_checkpoint_id_patch)
    second = threading.Thread(target=_checkpoint_ids.install_checkpoint_id_patch)
    try:
        first.start()
        assert first_importing.wait(5)
        second.start()
        first.join(5)
        second.join(5)
        monkeypatch.undo()

        assert _checkpoint_ids._original_uuid6 is original
        _checkpoint_ids.uninstall_checkpoint_id_patch()
        # The other install is still active.
        assert checkpoint_id_module.uuid6 is not original
        _checkpoint_ids.uninstall_checkpoint_id_patch()
        assert checkpoint_id_module.uuid6 is original
    finally:
        while _checkpoint_ids._original_uuid6 is not None:
            _checkpoint_ids.uninstall_checkpoint_id_patch()
        # Never leave a wrapper behind for later tests, even when the
        # assertions above failed with the original lost.
        for name in _checkpoint_ids._PATCHED_MODULE_NAMES:
            setattr(importlib.import_module(name), "uuid6", original)
