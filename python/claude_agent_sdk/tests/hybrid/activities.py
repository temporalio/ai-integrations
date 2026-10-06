"""Long-running CLI Activity and a separately scheduled, idempotent test tool."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from temporalio import activity
from temporalio.client import Client
from temporalio.exceptions import ApplicationError
from tests.hybrid.engine import Burst, PrototypeBlocked
from tests.hybrid.models import (
    Attempt,
    BurstInput,
    BurstResult,
    Call,
    Checkpoint,
    Entry,
    Reply,
    TurnCheckpoint,
)
from tests.hybrid.store import TranscriptStore
from tests.hybrid.workflows import HybridWorkflow


class HybridActivities:
    native_tools = False

    def __init__(
        self, client: Client, root: Path, env: dict[str, str], store: TranscriptStore
    ) -> None:
        self.client, self.root, self.env, self.store = client, root, env, store
        self.bursts: list[Burst] = []
        self.subagents = False
        self.recovery = False
        self.started: dict[str, float] = {}
        self.finished: dict[str, float] = {}
        self.tool_calls: list[str] = []
        self.fail_after_effect = False
        self.after_checkpoint: asyncio.Event | None = None
        self.after_suspension: asyncio.Event | None = None
        self.before_request: asyncio.Event | None = None
        self.before_delivery: asyncio.Event | None = None

    def create_burst(
        self,
        inp: BurstInput,
        attempt: Attempt,
        execute: Callable[[Call], Coroutine[Any, Any, Reply]],
        resume: bool,
        entries: dict[str, Entry],
    ) -> Burst:
        return Burst(
            self.root,
            self.env,
            self.store,
            inp.session_id,
            attempt,
            execute,
            resume=resume,
            subagents=self.subagents,
            recovery=self.recovery,
            recovery_entries=entries,
        )

    @activity.defn(name="hybrid_tool")
    async def tool(self, call: Call) -> Reply:
        self.tool_calls.append(call.id)
        self.started[call.id] = asyncio.get_running_loop().time()
        await asyncio.sleep(float(call.arguments.get("delay", 0)))
        # Stable across CLI retries, Workflow replay and Continue-As-New.
        workflow_id = activity.info().workflow_id
        assert workflow_id is not None
        key = workflow_id + ":tool-" + call.id
        text = await asyncio.to_thread(
            self.store.effect, key, {"n": call.arguments["n"]}
        )
        if self.fail_after_effect:
            self.fail_after_effect = False
            raise RuntimeError("injected failure after external effect")
        self.finished[call.id] = asyncio.get_running_loop().time()
        return Reply(text)

    @activity.defn(name="hybrid_burst")
    async def burst(self, inp: BurstInput) -> BurstResult:
        async def beat() -> None:
            while True:
                activity.heartbeat({"attempt": activity.info().attempt})
                await asyncio.sleep(0.2)

        heartbeat = asyncio.create_task(beat())
        try:
            return await self.run_burst(inp)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def run_burst(self, inp: BurstInput) -> BurstResult:
        info = activity.info()
        assert info.workflow_id is not None
        handle = self.client.get_workflow_handle(
            info.workflow_id, run_id=info.workflow_run_id
        )
        attempt = Attempt(
            inp.burst,
            inp.generation + info.attempt,
            f"{info.workflow_run_id}:{info.activity_id}:{inp.generation + info.attempt}",
        )
        before = await handle.query(HybridWorkflow.snapshot)
        if (
            before.pending is not None
            and before.pending.attempt.number > inp.generation
        ):
            # The CLI stopped and the Update committed, but the Activity's
            # completion may have been lost. Return its durable receipt.
            return BurstResult(pending=before.pending)
        snap = await handle.execute_update(HybridWorkflow.register, attempt)
        if snap.checkpoints and snap.checkpoints[-1].attempt.burst == inp.burst:
            return BurstResult(snap.checkpoints[-1].answers)
        # Completed-turn recovery can resume. Pending-call recovery has no proven protocol.
        if not self.recovery and any(
            e.call.attempt.burst == inp.burst for e in snap.ledger.values()
        ):
            raise ApplicationError(
                "blocked: Activity lost with uncheckpointed native MCP calls; "
                "published engine cannot yet be trusted to restore pending identities",
                non_retryable=True,
            )

        async def execute(call: Call) -> Reply:
            # Native hooks borrow a live executor after this request schedules
            # their Activity. A replacement joins the same accepted Update,
            # including when the original Activity completion was lost.
            return await handle.execute_update(
                HybridWorkflow.request,
                call,
                id="native-" + call.id if self.native_tools else None,
            )

        resume = inp.checkpoint is not None or inp.recovering
        if self.recovery and info.attempt > 1:
            # Storage may contain a requested call even if the old Worker died
            # before its request Update reached Temporal.
            # A missing recovery transcript must fail rather than silently
            # starting another model turn with replacement tool IDs.
            resume = True

        burst = self.create_burst(
            inp,
            attempt,
            execute,
            resume,
            {
                tid: entry
                for tid, entry in snap.ledger.items()
                if entry.call.attempt.burst == inp.burst
            },
        )
        burst.before_request = self.before_request
        burst.before_delivery = self.before_delivery
        self.bursts.append(burst)

        async def run() -> BurstResult:
            await burst.open()
            with (self.root / "cli-pids.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        {
                            "pid": burst.pid,
                            "worker": os.getpid(),
                            "supervised": burst.supervised,
                        }
                    )
                    + "\n"
                )
            answers = []
            for offset, prompt in enumerate(inp.prompts):
                index = inp.burst + offset
                completed = snap.turns.get(index)
                if completed is not None:
                    answers.append(completed.answer)
                    continue
                answer = (await burst.query(prompt)).result or ""
                uid, _ = await burst.checkpoint()
                await handle.execute_update(
                    HybridWorkflow.finish_turn,
                    TurnCheckpoint(attempt, index, uid, answer),
                )
                answers.append(answer)
            uid, delivered = await burst.checkpoint()
            await handle.execute_update(
                HybridWorkflow.acknowledge,
                Checkpoint(attempt, inp.session_id, uid, delivered, burst.pid, answers),
            )
            if self.after_checkpoint is not None:
                await self.after_checkpoint.wait()
            return BurstResult(answers)

        async def watch_suspension() -> BurstResult:
            while True:
                current = await handle.query(HybridWorkflow.snapshot)
                accepted = {
                    tid
                    for tid, entry in current.ledger.items()
                    if entry.call.attempt.burst == inp.burst
                }
                if (
                    current.suspend_requested == attempt
                    and burst.calls
                    and accepted == set(burst.calls)
                    and await burst.suspend_pending()
                ):
                    assert burst.suspended is not None
                    await handle.execute_update(
                        HybridWorkflow.acknowledge_suspension, burst.suspended
                    )
                    if self.after_suspension is not None:
                        await self.after_suspension.wait()
                    return BurstResult(pending=burst.suspended)
                await asyncio.sleep(0.05)

        running = asyncio.create_task(run())
        suspension = asyncio.create_task(watch_suspension()) if self.recovery else None
        try:
            if suspension is None:
                return await running
            await asyncio.wait(
                {running, suspension}, return_when=asyncio.FIRST_COMPLETED
            )
            if suspension.done():
                return await suspension
            # Killing the CLI ends the reader before its stopped receipt is
            # acknowledged. Let that acknowledgment finish before returning.
            if burst.delivery.is_set():
                return await running
            return await suspension
        except PrototypeBlocked as exc:
            raise ApplicationError(str(exc), non_retryable=True) from exc
        finally:
            running.cancel()
            if suspension is not None:
                suspension.cancel()
                await asyncio.gather(suspension, return_exceptions=True)
            await asyncio.gather(running, return_exceptions=True)
            await burst.close()
