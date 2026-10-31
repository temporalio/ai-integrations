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
import importlib
import re
import sys
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

from temporalio import workflow  # noqa: E402
from temporalio.worker import Replayer, Worker  # noqa: E402

# Bind deepagents symbols off the module importorskip returns: a static
# `from deepagents import ...` cannot resolve on Python 3.10 (deepagents
# needs >= 3.11), and with the package absent the type checkers mis-resolve
# the name against this same-named test directory.
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
