"""One live ClaudeSDKClient per burst, with a fail-closed native MCP bridge."""

from __future__ import annotations

import asyncio
import importlib
import os
import time
from collections.abc import Callable, Coroutine
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    MirrorErrorMessage,
    ResultMessage,
    SessionKey,
    SessionStoreEntry,
    StreamEvent,
    ToolResultBlock,
    ToolUseBlock,
    project_key_for_directory,
)
from mcp.server import Server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
    ToolAnnotations,
)

from tests.hybrid.models import Attempt, Call, Entry, PendingCheckpoint, Reply
from tests.hybrid.store import TranscriptStore

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "n": {"type": "integer"},
        "approval": {"type": "boolean"},
        "delay": {"type": "number", "minimum": 0},
    },
    "required": ["n"],
    "additionalProperties": False,
}


class PrototypeBlocked(RuntimeError):
    pass


def native_id(meta: Any) -> str:
    if hasattr(meta, "model_dump"):
        meta = meta.model_dump(by_alias=True)
    tid = (meta or {}).get("claudecode/toolUseId")
    if not isinstance(tid, str) or not tid:
        raise PrototypeBlocked("missing native MCP tool-use ID")
    return tid


class Burst:
    supervised = False

    def __init__(
        self,
        root: Path,
        env: dict[str, str],
        store: TranscriptStore,
        session_id: str,
        attempt: Attempt,
        execute: Callable[[Call], Coroutine[Any, Any, Reply]],
        *,
        resume: bool = False,
        subagents: bool = False,
        storage_timeout: float = 5,
        recovery: bool = False,
        recovery_entries: dict[str, Entry] | None = None,
    ) -> None:
        self.root, self.store, self.session_id = root, store, session_id
        self.attempt, self.execute = attempt, execute
        self.storage_timeout = storage_timeout
        self.recovery = recovery
        self.suspended: PendingCheckpoint | None = None
        self.suspension_failure: Exception | None = None
        self.suspending = asyncio.Lock()
        self.delivery = asyncio.Event()
        self.delivery.set()
        self.batch_ids: set[str] = set()
        self.batch_complete = False
        self.batch_stop: str | None = None
        self.key: SessionKey = {
            "project_key": project_key_for_directory(str(root)),
            "session_id": session_id,
        }
        self.observed: dict[str, tuple[str, dict[str, Any]]] = {}
        self.calls: dict[str, Call] = {}
        self.replies: dict[str, Reply] = {}
        self.callbacks: set[asyncio.Task[Reply]] = set()
        self.messages: list[Any] = []
        self.results: asyncio.Queue[ResultMessage | BaseException] = asyncio.Queue()
        self.active_children: set[str] = set()
        self.failure: BaseException | None = None
        self.closed = False
        self.pid = 0
        self.version = ""
        self.round_latencies: list[float] = []
        self.callback_started = asyncio.Event()
        self.before_request: asyncio.Event | None = None
        self.before_delivery: asyncio.Event | None = None
        self.received_reply = asyncio.Event()
        self.server = Server(
            "durable", on_list_tools=self.list_tools, on_call_tool=self.call_tool
        )
        recovery_options: dict[str, Any] = {}
        if recovery:
            if subagents:
                raise PrototypeBlocked("main-agent recovery excludes subagents")
            if not hasattr(ClaudeAgentOptions, "recover_pending_tool"):
                raise PrototypeBlocked(
                    "main-agent recovery requires the local SDK wheel"
                )
            recovery_options["recover_pending_tool"] = self.recover_tool
            # The only durable fixture is explicitly read-only and eligible
            # for concurrency. Other tools need their own scheduling policy.
            recovery_options["parallel_tool_recovery"] = True
            for tid, entry in (recovery_entries or {}).items():
                if entry.call.subpath:
                    raise PrototypeBlocked("child call in main-agent recovery ledger")
                self.calls[tid] = replace(entry.call, attempt=attempt)
                if entry.outcome is not None:
                    self.replies[tid] = entry.outcome
        self.sdk = ClaudeSDKClient(
            options=ClaudeAgentOptions(
                cwd=str(root),
                cli_path=os.environ.get("HYBRID_CLI_PATH"),
                env=env,
                tools=["Agent", "TaskOutput"] if subagents else [],
                allowed_tools=[
                    "mcp__durable__echo",
                    *(["Agent", "TaskOutput"] if subagents else []),
                ],
                mcp_servers={
                    "durable": {
                        "type": "sdk",
                        "name": "durable",
                        "instance": self.server,
                    }
                },
                setting_sources=[],
                strict_mcp_config=True,
                session_store=cast(Any, store),
                session_store_flush="eager",
                include_partial_messages=recovery,
                session_id=None if resume else session_id,
                resume=session_id if resume else None,
                forward_subagent_text=True,
                **recovery_options,
            )
        )
        self.reader: asyncio.Task[None] | None = None

    async def recover_tool(self, pending: Any) -> ToolResultBlock:
        if pending.key != self.key or not pending.name.startswith("mcp__durable__"):
            raise PrototypeBlocked("unexpected session or tool in recovery")
        name = pending.name.removeprefix("mcp__durable__")
        uid, subpath = await self.store.wait_call(
            self.key, pending.id, pending.name, pending.input, self.storage_timeout
        )
        if subpath or uid != pending.transcript_uuid:
            raise PrototypeBlocked("recovery ID disagrees with stored main-agent call")
        call = Call(pending.id, name, dict(pending.input), self.attempt, uid)
        self.calls[call.id] = call
        reply = await self.execute(call)
        self.replies[call.id] = reply
        return ToolResultBlock(call.id, reply.text, reply.is_error)

    async def list_tools(self, ctx: Any, params: Any) -> ListToolsResult:
        del ctx, params
        # Echo is read-only. Do not advertise mutating tools as read-only to force concurrency.
        return ListToolsResult(
            tools=[
                Tool(
                    name="echo",
                    description="Echo a number.",
                    input_schema=SCHEMA,
                    annotations=ToolAnnotations(
                        read_only_hint=True,
                        destructive_hint=False,
                        idempotent_hint=True,
                        open_world_hint=False,
                    ),
                )
            ]
        )

    async def call_tool(self, ctx: Any, params: Any) -> CallToolResult:
        try:
            tid = native_id(ctx.meta)
            arguments = dict(params.arguments or {})
            deadline = time.monotonic() + self.storage_timeout
            while tid not in self.observed:
                if time.monotonic() > deadline:
                    raise PrototypeBlocked(
                        f"no observed assistant call for native ID {tid}"
                    )
                await asyncio.sleep(0.01)
            if self.observed[tid] != ("mcp__durable__" + params.name, arguments):
                raise PrototypeBlocked(
                    "native ID disagrees with observed assistant call"
                )
            uid, subpath = await self.store.wait_call(
                self.key,
                tid,
                "mcp__durable__" + params.name,
                arguments,
                self.storage_timeout,
            )
            call = Call(tid, params.name, arguments, self.attempt, uid, subpath)
            self.calls[tid] = call
            self.callback_started.set()
            if self.before_request is not None:
                await self.before_request.wait()
            task = asyncio.create_task(self.execute(call))
            self.callbacks.add(task)
            try:
                reply = await task
                self.replies[tid] = reply
                self.received_reply.set()
                if self.before_delivery is not None:
                    await self.before_delivery.wait()
                await self.delivery.wait()
            finally:
                self.callbacks.discard(task)
            return CallToolResult(
                content=[TextContent(type="text", text=reply.text)],
                is_error=reply.is_error,
            )
        except Exception as exc:
            self.failure = exc
            return CallToolResult(
                content=[TextContent(type="text", text=str(exc))], is_error=True
            )

    async def open(self) -> Burst:
        await self.sdk.connect()
        # The test-only suspension probe also owns this subprocess's hard stop.
        transport: Any = self.sdk._transport
        self.pid = transport._process.pid
        self.reader = asyncio.create_task(self.read())
        return self

    async def read(self) -> None:
        try:
            async for msg in self.sdk.receive_messages():
                self.messages.append(msg)
                if isinstance(msg, StreamEvent) and msg.parent_tool_use_id is None:
                    event = msg.event
                    if event.get("type") == "message_start":
                        self.batch_ids.clear()
                        self.batch_complete = False
                        self.batch_stop = None
                    elif event.get("type") == "content_block_start":
                        block = event.get("content_block", {})
                        if block.get("type") == "tool_use":
                            self.batch_ids.add(block["id"])
                    elif event.get("type") == "message_delta":
                        self.batch_stop = event.get("delta", {}).get("stop_reason")
                    elif event.get("type") == "message_stop":
                        self.batch_complete = self.batch_stop == "tool_use"
                if isinstance(msg, AssistantMessage):
                    for b in msg.content:
                        if isinstance(b, ToolUseBlock):
                            self.observed[b.id] = (b.name, dict(b.input))
                if isinstance(msg, MirrorErrorMessage):
                    self.failure = PrototypeBlocked(
                        msg.error or "transcript mirror failed"
                    )
                data = getattr(msg, "data", {})
                if data.get("subtype") == "init":
                    self.version = str(data.get("claude_code_version", ""))
                subtype = getattr(msg, "subtype", data.get("subtype"))
                task_id = getattr(msg, "task_id", data.get("task_id"))
                if subtype == "task_started" and task_id:
                    self.active_children.add(task_id)
                elif subtype == "task_notification" and task_id:
                    self.active_children.discard(task_id)
                if isinstance(msg, ResultMessage):
                    await self.results.put(msg)
        except Exception as exc:
            await self.results.put(exc)

    async def query(self, prompt: Any) -> ResultMessage:
        started = time.monotonic()
        await self.sdk.query(prompt)
        human_result: ResultMessage | None = None
        while True:
            msg = await self.results.get()
            if isinstance(msg, BaseException):
                raise msg
            if self.failure:
                raise self.failure
            if msg.is_error:
                raise PrototypeBlocked(str(msg.errors or msg.subtype))
            if not msg.origin or msg.origin.get("kind") == "human":
                human_result = msg
            if human_result and not self.active_children and not self.callbacks:
                self.round_latencies.append(time.monotonic() - started)
                return human_result

    async def checkpoint(self) -> tuple[str, list[str]]:
        if self.callbacks or self.active_children:
            raise PrototypeBlocked("pending calls/children must keep the CLI alive")
        transcripts = await self.store.transcripts(self.key)
        delivered: set[str] = set()
        for entries in transcripts.values():
            for entry in entries:
                content = entry.get("message", {}).get("content", [])
                if isinstance(content, list):
                    for block in content:
                        tid = block.get("tool_use_id")
                        if block.get("type") != "tool_result" or tid not in self.calls:
                            continue
                        reply = self.replies.get(tid)
                        value = block.get("content", "")
                        if isinstance(value, list):
                            value = "".join(b.get("text", "") for b in value)
                        if (
                            reply is None
                            or value != reply.text
                            or bool(block.get("is_error")) != reply.is_error
                        ):
                            raise PrototypeBlocked(
                                "stored tool result differs from completed outcome"
                            )
                        delivered.add(tid)
        if not set(self.calls).issubset(delivered):
            raise PrototypeBlocked("completed results have not reached session storage")
        final = [e for e in transcripts[""] if e.get("type") == "assistant"]
        if not final:
            raise PrototypeBlocked("completed assistant turn is not stored")
        return str(final[-1]["uuid"]), sorted(self.calls)

    async def suspend_pending(self) -> bool:
        if self.suspended is not None:
            return True
        if not self.recovery or self.active_children or self.closed:
            return False
        # Optional sibling-SDK capability. Published SDKs never reach this
        # path, and their environments must still be able to lint the probe.
        pending_tool_uses = importlib.import_module(
            "claude_agent_sdk._internal.main_agent_recovery"
        ).pending_tool_uses

        async with self.suspending:
            if self.suspended is not None:
                return True
            if not self.batch_complete or not self.batch_ids.issubset(self.calls):
                return False
            # Park result delivery before the first await. Temporal work can
            # finish, but no new callback outcome can reach the live CLI.
            self.delivery.clear()
            stopped = False
            try:
                transcripts = await self.store.transcripts(self.key)
                if any(path for path in transcripts if path):
                    return False
                entries = transcripts[""]
                stored = {
                    block.get("id"): (entry.get("uuid"), block)
                    for entry in entries
                    if entry.get("type") == "assistant"
                    for block in entry.get("message", {}).get("content", [])
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                }
                if not set(self.calls).issubset(stored):
                    return False
                for tid, recorded in self.calls.items():
                    uid, block = stored[tid]
                    if (uid, block.get("name"), block.get("input")) != (
                        recorded.transcript_uuid,
                        "mcp__durable__" + recorded.name,
                        recorded.arguments,
                    ):
                        return False
                pending, leaf = pending_tool_uses(
                    self.key, cast(list[SessionStoreEntry], entries)
                )
                ids = {call.id for call in pending}
                if not ids or not ids.issubset(self.batch_ids):
                    return False
                if not self.batch_ids.issubset(self.observed):
                    return False
                for call in pending:
                    recovery_call = self.calls.get(call.id)
                    if recovery_call is None or (
                        call.name,
                        call.input,
                        call.transcript_uuid,
                    ) != (
                        "mcp__durable__" + recovery_call.name,
                        recovery_call.arguments,
                        recovery_call.transcript_uuid,
                    ):
                        return False
                delivered: set[str] = set()
                for entry in entries:
                    content = entry.get("message", {}).get("content", [])
                    if entry.get("type") != "user" or not isinstance(content, list):
                        continue
                    for block in content:
                        if (
                            not isinstance(block, dict)
                            or block.get("type") != "tool_result"
                        ):
                            continue
                        result_id = block.get("tool_use_id")
                        if (
                            not isinstance(result_id, str)
                            or result_id not in self.calls
                        ):
                            continue
                        reply = self.replies.get(result_id)
                        value = block.get("content", "")
                        if isinstance(value, list):
                            value = "".join(b.get("text", "") for b in value)
                        if reply is None or (value, bool(block.get("is_error"))) != (
                            reply.text,
                            reply.is_error,
                        ):
                            return False
                        delivered.add(result_id)
                if ids | delivered != set(self.calls) or ids & delivered:
                    return False
                assert leaf is not None
                # SIGKILL/TerminateProcess before cancelling any callback or
                # closing stdin. EOF/SIGTERM lets the engine synthesize errors.
                transport: Any = self.sdk._transport
                process = transport._process
                process.kill()
                stopped = True
                await asyncio.wait_for(process.wait(), self.storage_timeout)
                await self.close()
                restored = await self.store.load(self.key)
                if restored != entries:
                    raise PrototypeBlocked(
                        "transcript changed while stopping pending CLI"
                    )
                self.suspended = PendingCheckpoint(
                    self.attempt,
                    self.session_id,
                    str(leaf["uuid"]),
                    sorted(ids),
                    sorted(delivered),
                    self.pid,
                )
                return True
            except Exception as exc:
                if stopped:
                    raise
                self.suspension_failure = exc
                return False
            finally:
                if not stopped:
                    self.delivery.set()

    async def close(self) -> None:
        self.closed = True
        for task in self.callbacks:
            task.cancel()
        await asyncio.gather(*self.callbacks, return_exceptions=True)
        await self.sdk.disconnect()
        if self.reader is not None:
            self.reader.cancel()
            with suppress(asyncio.CancelledError):
                await self.reader
