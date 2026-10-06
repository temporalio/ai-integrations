"""Native Read/Edit execution gated by Activities, with an explicit recovery limit."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Coroutine
from typing import Any

from claude_agent_sdk import HookMatcher, ResultMessage, ToolResultBlock

from temporalio import activity
from temporalio.claude_agent_sdk._managed import SupervisedClient
from temporalio.exceptions import ApplicationError
from tests.hybrid.activities import HybridActivities
from tests.hybrid.engine import Burst, PrototypeBlocked
from tests.hybrid.models import Attempt, BurstInput, BurstResult, Call, Entry, Reply
from tests.hybrid.native_store import NativeStore
from tests.hybrid.workflows import HybridWorkflow


class NativeBurst(Burst):
    supervised = True

    def __init__(self, *args: Any, native_store: NativeStore, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.native_store = native_store
        self.sdk.options.tools = sorted(
            t for t in native_store.tools if not t.startswith("mcp__")
        )
        self.sdk.options.allowed_tools = sorted(native_store.tools)
        self.sdk.options.mcp_servers = native_store.mcp_servers
        self.sdk.options.permission_mode = "acceptEdits"
        if hasattr(self.sdk.options, "parallel_tool_recovery"):
            # Read/Edit share mutable workspace state; restore them in order.
            setattr(self.sdk.options, "parallel_tool_recovery", False)
        if native_store.phase == "ordinary-probe":
            # Observe the real engine's default resume behavior, without the
            # pre-start recovery callback or any invented transcript markers.
            setattr(self.sdk.options, "recover_pending_tool", None)
        self.sdk.options.hooks = {
            "PreToolUse": [
                HookMatcher(matcher=".*", hooks=[self.pre_tool], timeout=120)
            ],
            "PostToolUse": [
                HookMatcher(matcher=".*", hooks=[self.post_tool], timeout=120)
            ],
            "PostToolUseFailure": [
                HookMatcher(matcher=".*", hooks=[self.post_tool], timeout=120)
            ],
        }
        self.sdk = SupervisedClient(self.sdk.options, native_store.lock_path)
        self.native_gate = asyncio.Lock()
        self.gated: set[str] = set()

    async def open(self) -> NativeBurst:
        await asyncio.to_thread(self.native_store.checkout, self.attempt)
        # A result can reach the transcript before its Activity completion
        # reaches Temporal. Join that original Activity even when the SDK sees
        # no pending call and therefore does not invoke its recovery callback.
        for tid, call in self.calls.items():
            if tid not in self.replies and self.native_store.execution(tid):
                row = self.native_store.execution(tid)
                if row is not None and row["result"] is not None:
                    self.replies[tid] = await self.execute(call)
        await super().open()
        return self

    async def recover_tool(self, pending: Any) -> ToolResultBlock:
        if pending.key != self.key or pending.name not in self.native_store.tools:
            raise PrototypeBlocked("unexpected native recovery call")
        uid, subpath = await self.store.wait_call(
            self.key, pending.id, pending.name, pending.input, self.storage_timeout
        )
        if subpath or uid != pending.transcript_uuid:
            raise PrototypeBlocked("native recovery storage proof differs")
        row = self.native_store.execution(pending.id)
        if row is None or row["result"] is None:
            raise PrototypeBlocked(
                "native recovery blocked: no committed execution result for "
                + pending.name
                + "; its effect may have executed and requires reconciliation"
            )
        call = Call(pending.id, pending.name, dict(pending.input), self.attempt, uid)
        self.calls[call.id] = call
        reply = await self.execute(call)
        self.replies[call.id] = reply
        block = json.loads(reply.text)
        if block != row["result"] or block["tool_use_id"] != call.id:
            raise PrototypeBlocked(
                "native Activity outcome differs from its original result"
            )
        return ToolResultBlock(call.id, block.get("content"), block.get("is_error"))

    async def suspend_pending(self) -> bool:
        # These hooks do not park delivery like the MCP callback bridge. Keep
        # their executor alive; native pending suspension is not part of this probe.
        return False

    async def pre_tool(self, data: Any, tid: str | None, context: Any) -> Any:
        del context
        try:
            if not tid or tid != data.get("tool_use_id"):
                raise PrototypeBlocked("missing or inconsistent native hook ID")
            if self.native_store.phase == "ordinary-probe":
                self.native_store.mark(tid, "ordinary-hook")
                raise PrototypeBlocked(
                    "ordinary resume tried to dispatch a native call"
                )
            # Read and Edit share a workspace. Keep the previous tool's durable
            # completion ahead of the next tool's execution permission.
            if self.callbacks:
                await asyncio.gather(*self.callbacks)
            await self.native_gate.acquire()
            self.gated.add(tid)
            deadline = time.monotonic() + self.storage_timeout
            while tid not in self.observed:
                if time.monotonic() > deadline:
                    raise PrototypeBlocked("native hook has no observed assistant call")
                await asyncio.sleep(0.02)
            name, arguments = data["tool_name"], data["tool_input"]
            if self.observed[tid] != (name, arguments):
                raise PrototypeBlocked("native hook disagrees with assistant call")
            uid, subpath = await self.store.wait_call(
                self.key, tid, name, arguments, self.storage_timeout
            )
            call = Call(tid, name, dict(arguments), self.attempt, uid, subpath)
            self.calls[tid] = call
            await asyncio.to_thread(self.native_store.prepare, call)

            async def finish() -> Reply:
                reply = await self.execute(call)
                self.replies[tid] = reply
                return reply

            task = asyncio.create_task(finish())
            self.callbacks.add(task)

            def completed(done: asyncio.Task[Reply]) -> None:
                self.callbacks.discard(done)
                self.release_gate(tid)
                if not done.cancelled() and (error := done.exception()) is not None:
                    self.failure = error

            task.add_done_callback(completed)
            while True:
                if task.done():
                    raise PrototypeBlocked(
                        "native Activity completed before execution permission"
                    )
                row = self.native_store.execution(tid)
                if row is not None and row["phase"] == "permitted":
                    break
                await asyncio.sleep(0.02)
            if self.native_store.phase == name + "-before-execution":
                self.native_store.mark(tid, "held-before-execution")
                await self.native_store.held.wait()
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                }
            }
        except asyncio.CancelledError:
            if tid:
                self.release_gate(tid)
            raise
        except Exception as exc:
            if tid:
                self.release_gate(tid)
            self.failure = exc
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": str(exc),
                }
            }

    def release_gate(self, tid: str) -> None:
        if tid in self.gated:
            self.gated.remove(tid)
            self.native_gate.release()

    async def post_tool(self, data: Any, tid: str | None, context: Any) -> Any:
        del context
        try:
            if not tid or tid != data.get("tool_use_id") or tid not in self.calls:
                raise PrototypeBlocked("unknown native completion ID")
            call = self.calls[tid]
            if (data["tool_name"], data["tool_input"]) != (call.name, call.arguments):
                raise PrototypeBlocked("native completion disagrees with original call")
            self.native_store.mark(tid, "executed")
            if self.native_store.phase == data["tool_name"] + "-after-write":
                await self.native_store.held.wait()
            await asyncio.to_thread(self.native_store.stage, tid)
            # Let the engine render its original tool_result. The eager mirror
            # captures that exact block with this snapshot before acknowledging
            # the tool Activity. We never manufacture a built-in's output.
            return {}
        except Exception as exc:
            self.failure = exc
            return {"continue": False, "stopReason": str(exc)}

    async def query(self, prompt: Any) -> ResultMessage:
        if prompt == "edit" and self.native_store.phase == "between-turns":
            with self.native_store.connect() as db:
                db.execute(
                    "INSERT INTO native_events(id,phase,owner) VALUES ('','between-turns',?)",
                    (self.native_store.owner,),
                )
            await self.native_store.held.wait()
        await self.sdk.query(prompt)
        while True:
            message = await self.results.get()
            if isinstance(message, BaseException):
                raise message
            if self.failure:
                raise self.failure
            if message.is_error:
                raise PrototypeBlocked(str(message.errors or message.subtype))
            if not message.origin or message.origin.get("kind") == "human":
                if self.callbacks:
                    await asyncio.gather(*self.callbacks)
                return message

    async def checkpoint(self) -> tuple[str, list[str]]:
        deadline = time.monotonic() + self.storage_timeout
        while True:
            entries = (await self.store.transcripts(self.key))[""]
            delivered = set()
            for entry in entries:
                content = entry.get("message", {}).get("content", [])
                if not isinstance(content, list):
                    continue
                for block in content:
                    tid = block.get("tool_use_id")
                    if block.get("type") != "tool_result" or tid not in self.calls:
                        continue
                    reply = self.replies.get(tid)
                    if reply is None:
                        raise PrototypeBlocked("native result has no Activity outcome")
                    original = json.loads(reply.text)
                    if (block.get("content"), bool(block.get("is_error"))) != (
                        original.get("content"),
                        bool(original.get("is_error")),
                    ):
                        raise PrototypeBlocked("native result content changed")
                    delivered.add(tid)
            final = [e for e in entries if e.get("type") == "assistant"]
            if final and set(self.calls).issubset(delivered):
                return str(final[-1]["uuid"]), sorted(self.calls)
            if time.monotonic() > deadline:
                raise PrototypeBlocked("native checkpoint has not reached storage")
            await asyncio.sleep(0.02)


class NativeActivities(HybridActivities):
    native_tools = True

    async def run_burst(self, inp: BurstInput) -> BurstResult:
        assert isinstance(self.store, NativeStore)
        info = activity.info()
        assert info.workflow_id is not None
        if info.attempt > 1:
            handle = self.client.get_workflow_handle(
                info.workflow_id, run_id=info.workflow_run_id
            )
            snapshot = await handle.query(HybridWorkflow.snapshot)
            for tid, entry in snapshot.ledger.items():
                row = self.store.execution(tid)
                if row is not None and row["result"] is not None:
                    # Complete the original accepted Update before registering
                    # the replacement attempt. A cache query alone does not
                    # prove its completion was committed to Workflow history.
                    # It also avoids the UpdateMachine replay failure seen on
                    # Temporal 1.33/1.34 when both completions share a retried WFT.
                    await handle.execute_update(
                        HybridWorkflow.request, entry.call, id="native-" + tid
                    )
        return await super().run_burst(inp)

    def create_burst(
        self,
        inp: BurstInput,
        attempt: Attempt,
        execute: Callable[[Call], Coroutine[Any, Any, Reply]],
        resume: bool,
        entries: dict[str, Entry],
    ) -> NativeBurst:
        assert isinstance(self.store, NativeStore)
        return NativeBurst(
            self.store.workspace,
            self.env,
            self.store,
            inp.session_id,
            attempt,
            execute,
            resume=resume,
            recovery=self.recovery,
            recovery_entries=entries,
            native_store=self.store,
        )

    @activity.defn(name="hybrid_tool")
    async def tool(self, call: Call) -> Reply:
        assert isinstance(self.store, NativeStore)

        async def beat() -> None:
            while True:
                activity.heartbeat(call.id)
                await asyncio.sleep(0.2)

        heartbeat = asyncio.create_task(beat())
        try:
            row = self.store.execution(call.id)
            if row is None:
                raise ApplicationError(
                    "missing native execution request", non_retryable=True
                )
            if row["result"] is None:
                try:
                    await asyncio.to_thread(self.store.permit, call)
                except RuntimeError as exc:
                    raise ApplicationError(str(exc), non_retryable=True) from exc
            while True:
                row = self.store.execution(call.id)
                if row is not None and row["result"] is not None:
                    if (
                        self.store.phase
                        == call.name + "-after-commit-before-completion"
                    ):
                        await self.store.held.wait()
                    return Reply(
                        json.dumps(row["result"], sort_keys=True),
                        bool(row["result"].get("is_error")),
                    )
                await asyncio.sleep(0.02)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
