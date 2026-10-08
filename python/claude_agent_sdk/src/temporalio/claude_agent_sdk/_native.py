"""Live native child delivery backed by Workflow-owned Activity outcomes.

The command hook never executes a requested effect. Shell calls run in a bounded
tool Activity; MCP calls return through an SDK server with the original tool-use
ID. A lost live callback is deliberately not reconstructed by calling the model.
"""

from __future__ import annotations

import asyncio
import dataclasses
import fnmatch
import hashlib
import hmac
import json
import secrets
import shlex
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.server import Server
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.memory import create_client_server_memory_streams
from mcp.types import CallToolResult, ListToolsResult, Tool

from claude_agent_sdk import ToolUseBlock, project_key_for_directory
from temporalio import activity
from temporalio.exceptions import ApplicationError

from ._models import DeferredCall, NativeRequest, SegmentInput, ToolOutcome

UPDATE = "__claude_agent_native_tool"
PREFIX = "mcp__durable__"


@asynccontextmanager
async def mcp_connection(
    config: dict[str, Any], cwd: str | None = None
) -> AsyncIterator[Any]:
    """Connect to one configured MCP server without changing its tool contracts."""
    kind = config.get("type", "stdio")
    if kind == "sdk":
        server = config["instance"]
        async with create_client_server_memory_streams() as (client_io, server_io):
            async with anyio.create_task_group() as group:
                group.start_soon(
                    server.run, *server_io, server.create_initialization_options()
                )
                async with ClientSession(*client_io) as session:
                    await session.initialize()
                    yield session
                group.cancel_scope.cancel()
        return
    if kind == "stdio":
        transport = stdio_client(
            StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env"),
                cwd=cwd,
            ),
        )
    elif kind == "sse":
        transport = sse_client(config["url"], headers=config.get("headers"))
    elif kind == "http":
        async with create_mcp_http_client(headers=config.get("headers")) as http:
            async with streamable_http_client(
                config["url"], http_client=http
            ) as streams:
                async with ClientSession(
                    cast(Any, streams[0]), cast(Any, streams[1])
                ) as session:
                    await session.initialize()
                    yield session
        return
    else:
        raise ApplicationError(f"Unsupported MCP transport: {kind}", non_retryable=True)
    async with transport as streams:
        async with ClientSession(
            cast(Any, streams[0]), cast(Any, streams[1])
        ) as session:
            await session.initialize()
            yield session


class ChildStore:
    """Track child keys, including stores without optional key enumeration."""

    def __init__(self, inner: Any, subpaths: list[str]) -> None:
        """Wrap a store and its previously committed child keys."""
        self.inner = inner
        self.subpaths = set(subpaths)

    async def append(self, key: Any, entries: Any) -> None:
        """Record every child key before forwarding its transcript append."""
        if key.get("subpath"):
            self.subpaths.add(key["subpath"])
        await self.inner.append(key, entries)

    async def load(self, key: Any) -> Any:
        """Read a transcript from the underlying store."""
        return await self.inner.load(key)

    async def list_subkeys(self, key: Any) -> list[str]:
        """Restore child transcripts on every Worker, including after handover."""
        method = getattr(self.inner, "list_subkeys", None)
        if method is not None:
            self.subpaths.update(await method(key))
        return sorted(self.subpaths)

    def __getattr__(self, name: str) -> Any:
        """Forward optional session-store operations to the underlying store."""
        return getattr(self.inner, name)


class NativeBridge:
    """One live engine attempt, with strict original-ID verification and fencing."""

    def __init__(
        self,
        inp: SegmentInput,
        attempt: int,
        store: ChildStore,
        cwd: str | None,
        session_id: str,
        folder: str,
    ) -> None:
        """Create a bridge; register it before starting its engine."""
        self.inp, self.store, self.folder = inp, store, folder
        self.request = NativeRequest(
            inp.conversation.agent if inp.conversation else "",
            inp.segment_index,
            attempt,
            secrets.token_hex(32),
        )
        self.key = {
            "project_key": project_key_for_directory(cwd),
            "session_id": session_id,
        }
        self.observed: dict[str, tuple[str, dict[str, Any]]] = {}
        self.delegations: set[str] = set()
        self.receipts: dict[str, tuple[str, DeferredCall]] = {}
        self.outcomes: dict[str, ToolOutcome] = {}
        self.children: set[str] = set()
        self.active_children: set[str] = set()
        self.failures: list[str] = []
        self.ran_inside: list[str] = []
        self.closed = False
        self.protected = False
        self.server: ThreadingHTTPServer | None = None
        self.loop = asyncio.get_running_loop()
        self.key_secret = secrets.token_hex(32)
        self.callbacks: set[asyncio.Task[Any]] = set()
        info = activity.info()
        self.handle = activity.client().get_workflow_handle(
            info.workflow_id or "", run_id=info.workflow_run_id
        )

    async def update(
        self, call: DeferredCall | None = None, child: str = ""
    ) -> ToolOutcome | None:
        """Send a fenced request to this Activity's own Workflow run."""
        try:
            return await self.handle.execute_update(
                UPDATE,
                dataclasses.replace(self.request, call=call, child=child),
                result_type=ToolOutcome if call is not None else None,
            )
        except Exception as err:
            raise ApplicationError(
                f"Native child bridge refused the request: {err}",
                type="ClaudeNativeDeliveryLost",
                non_retryable=True,
            ) from err

    def observe(self, message: Any) -> None:
        """Retain the native assistant's exact call identity, before tool dispatch."""
        for block in getattr(message, "content", []):
            if isinstance(block, ToolUseBlock):
                value = (block.name, dict(block.input))
                if block.id in self.observed and self.observed[block.id] != value:
                    self.failures.append("Conflicting assistant tool-use ID")
                self.observed[block.id] = value
                if block.name == "Agent" and not getattr(
                    message, "parent_tool_use_id", None
                ):
                    self.delegations.add(block.id)

    def managed(self, name: str) -> bool:
        """Whether this tool must run as its own Activity."""
        return name.startswith(PREFIX) or any(
            fnmatch.fnmatchcase(name, pattern) for pattern in self.inp.tool_activities
        )

    async def proof(
        self, tid: str, child: str, name: str, arguments: dict[str, Any]
    ) -> None:
        """Match hook/MCP identity to both the assistant and the child transcript."""
        if (
            self.closed
            or not self.protected
            or child not in self.children
            or not Path(self.folder, "worker.lock").is_file()
        ):
            raise RuntimeError("Unprotected, stopped, or unidentified native child")
        path = f"subagents/agent-{child}"
        for _ in range(500):
            observed = self.observed.get(tid)
            if observed is not None and observed != (name, arguments):
                raise RuntimeError("Native hook differs from the assistant call")
            entries = await self.store.load({**self.key, "subpath": path}) or []
            matches = [
                block
                for entry in entries
                for block in self.blocks(entry)
                if block.get("type") == "tool_use" and block.get("id") == tid
            ]
            if matches:
                if len(matches) != 1 or (
                    matches[0].get("name"),
                    matches[0].get("input"),
                ) != (name, arguments):
                    raise RuntimeError("Native hook differs from the stored child call")
                if observed is not None:
                    return
            if self.closed:
                break
            await asyncio.sleep(0.01)
        raise RuntimeError(
            "Native child call did not reach the assistant and session store"
        )

    @staticmethod
    def blocks(entry: dict[str, Any]) -> list[dict[str, Any]]:
        """Read content blocks without interpreting tool-controlled text as identity."""
        content = entry.get("message", {}).get("content", [])
        return content if isinstance(content, list) else []

    async def execute(self, tid: str) -> ToolOutcome:
        """Reuse the Workflow's one accepted execution for this original ID."""
        if self.closed or Path(self.folder, "stop").exists():
            raise RuntimeError("Native segment stopped before tool acceptance")
        child, call = self.receipts[tid]
        outcome = await self.update(call, child)
        if outcome is None:
            raise RuntimeError("Native tool Update returned no outcome")
        self.outcomes[tid] = outcome
        return outcome

    async def hook(self, event: dict[str, Any]) -> dict[str, Any]:
        """Serve child lifecycle and tool hooks; refuse any unverified request."""
        task = asyncio.current_task()
        if task is not None:
            self.callbacks.add(task)
        try:
            kind, child = event.get("hook_event_name"), event.get("agent_id", "")
            if self.closed:
                raise RuntimeError("Native segment stopped")
            if kind == "SubagentStart":
                if not child:
                    raise RuntimeError("Missing child identity")
                self.children.add(child)
                self.active_children.add(child)
                return {}
            if kind == "SubagentStop":
                self.active_children.discard(child)
                return {}
            tid, name, arguments = (
                event.get("tool_use_id", ""),
                event.get("tool_name", ""),
                event.get("tool_input", {}),
            )
            if not child and kind == "PreToolUse" and self.managed(name):
                # Assistant observation may race the command hook. Waiting for its
                # own original ID also makes the whole parallel batch visible.
                for _ in range(500):
                    if tid in self.observed:
                        break
                    await asyncio.sleep(0.01)
                if tid not in self.observed:
                    raise RuntimeError("Parent call was not observed")
                while not self.closed:
                    entries = await self.store.load(self.key) or []
                    answered = {
                        b.get("tool_use_id")
                        for e in entries
                        for b in self.blocks(e)
                        if b.get("type") == "tool_result"
                    }
                    started = {
                        b.get("tool_use_id")
                        for e in entries
                        for b in self.blocks(e)
                        if b.get("type") == "tool_result" and b.get("is_error")
                    }
                    for path in self.store.subpaths.copy():
                        child_entries = (
                            await self.store.load({**self.key, "subpath": path}) or []
                        )
                        started.update(
                            e.get("toolUseId")
                            for e in child_entries
                            if e.get("type") == "agent_metadata"
                        )
                    if (
                        not self.active_children
                        and self.delegations <= answered & started
                    ):
                        return {}
                    await asyncio.sleep(0.01)
                raise RuntimeError("Stopped while draining native children")
            if not child or not self.managed(name):
                return {}
            if kind == "PreToolUse":
                await self.proof(tid, child, name, arguments)
                call = DeferredCall(
                    tid,
                    name.removeprefix(PREFIX),
                    arguments,
                    "durable" if name.startswith(PREFIX) else "engine",
                )
                previous = self.receipts.setdefault(tid, (child, call))
                if previous != (child, call):
                    raise RuntimeError("Conflicting child call receipt")
                result: dict[str, Any] = {
                    "hookEventName": kind,
                    "permissionDecision": "allow",
                }
                if name in ("Bash", "PowerShell"):
                    outcome = await self.execute(tid)
                    result["updatedInput"] = {
                        "command": self.renderer(tid, name, outcome),
                        "run_in_background": False,
                        "timeout": 30000,
                    }
                return result
            if name in ("Bash", "PowerShell") and kind == "PostToolUse":
                outcome = self.outcomes[tid]
                if outcome.is_error:
                    raise RuntimeError("An error result rendered as success")
                raw = outcome.native_output or {
                    "stdout": str(outcome.content or ""),
                    "stderr": "",
                    "interrupted": False,
                    "isImage": False,
                }
                return {"hookEventName": kind, "updatedToolOutput": raw}
            return {}
        except Exception as err:
            self.failures.append(str(err))
            return {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Temporal native delivery stopped: {err}",
            }
        finally:
            if task is not None:
                self.callbacks.discard(task)

    def renderer(self, tid: str, name: str, outcome: ToolOutcome) -> str:
        """Render recorded bytes and status; never run the child's original command."""
        raw = outcome.native_output
        if raw is None:
            raw = (
                {"stdout": "", "stderr": str(outcome.content or ""), "exitCode": 1}
                if outcome.is_error
                else {"stdout": "", "stderr": "", "exitCode": 0}
            )
        if not isinstance(raw.get("stdout", ""), str) or not isinstance(
            raw.get("stderr", ""), str
        ):
            raise RuntimeError("Unsupported native shell output shape")
        payload = json.dumps({"output": raw, "is_error": outcome.is_error}).encode()
        digest = hashlib.sha256(payload).hexdigest()
        file = Path(
            self.folder, f"result-{hashlib.sha256(tid.encode()).hexdigest()}.json"
        )
        file.write_bytes(payload)
        argv = [
            sys.executable,
            "-I",
            "-S",
            str(Path(__file__).with_name("_render.py")),
            str(file),
            digest,
        ]
        if name == "PowerShell":
            return (
                "& "
                + " ".join("'" + arg.replace("'", "''") + "'" for arg in argv)
                + "; exit $LASTEXITCODE"
            )
        return shlex.join(argv)

    def start(self) -> dict[str, str]:
        """Start an authenticated loopback hook bridge on a private per-attempt key."""
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                if not hmac.compare_digest(
                    self.headers.get("Authorization", ""), bridge.key_secret
                ):
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 2 * 1024 * 1024:
                        raise ValueError("Invalid native hook size")
                    event = json.loads(self.rfile.read(length))
                    future = asyncio.run_coroutine_threadsafe(
                        bridge.hook(event), bridge.loop
                    )
                    data = json.dumps(future.result()).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except Exception:
                    self.send_error(500)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return {
            "TCA_CHILD_URL": f"http://127.0.0.1:{self.server.server_port}",
            "TCA_CHILD_KEY": self.key_secret,
            "TCA_REQUIRE_LOCK": "1",
        }

    async def verify(self) -> dict[str, list[dict[str, Any]]]:
        """Require every recorded outcome to have reached its original child ID."""
        if self.failures or self.callbacks or self.active_children:
            raise RuntimeError(
                "Native child delivery did not drain: " + "; ".join(self.failures)
            )
        transcripts = {
            path: await self.store.load({**self.key, "subpath": path}) or []
            for path in sorted(self.store.subpaths)
        }
        for tid, (child, call) in self.receipts.items():
            outcome = self.outcomes.get(tid)
            results = [
                block
                for entry in transcripts.get(f"subagents/agent-{child}", [])
                for block in self.blocks(entry)
                if block.get("type") == "tool_result"
                and block.get("tool_use_id") == tid
            ]
            if (
                outcome is None
                or len(results) != 1
                or bool(results[0].get("is_error")) != outcome.is_error
            ):
                raise RuntimeError(f"Native result missing or mismatched for {tid}")
            expected = (
                outcome.blocks
                if outcome.blocks is not None
                else (
                    outcome.content
                    if isinstance(outcome.content, str)
                    else json.dumps(outcome.content)
                )
            )
            content = results[0].get("content")
            if (
                outcome.is_error
                and outcome.native_output is None
                and call.name in ("Bash", "PowerShell")
            ):
                expected = f"Exit code 1\n{expected}"
            candidates = [expected, [{"type": "text", "text": expected}]]
            if (
                outcome.is_error
                and call.name not in ("Bash", "PowerShell")
                and isinstance(expected, str)
            ):
                candidates.extend(
                    [
                        f"Error: {expected}",
                        [{"type": "text", "text": f"Error: {expected}"}],
                    ]
                )
            if content not in candidates:
                raise RuntimeError(f"Native result content differs for {tid}")
        return transcripts

    async def close(self) -> None:
        """Fence callbacks before engine cleanup and stop the loopback server."""
        self.closed = True
        for task in list(self.callbacks):
            task.cancel()
        if self.callbacks:
            await asyncio.gather(*self.callbacks, return_exceptions=True)
        if self.server is not None:
            await asyncio.to_thread(self.server.shutdown)
            self.server.server_close()
            self.server = None

    def proxy(
        self, name: str, tools: list[Tool], session: Any = None
    ) -> dict[str, Any]:
        """Expose unchanged MCP schemas while routing managed child effects durably."""
        bridge = self

        async def list_tools(ctx: Any, params: Any) -> ListToolsResult:
            del ctx, params
            return ListToolsResult(tools=tools)

        async def call_tool(ctx: Any, params: Any) -> CallToolResult:
            task = asyncio.current_task()
            if task is not None:
                bridge.callbacks.add(task)
            try:
                meta = (
                    ctx.meta.model_dump(by_alias=True)
                    if hasattr(ctx.meta, "model_dump")
                    else ctx.meta or {}
                )
                tid = str(meta.get("claudecode/toolUseId") or "")
                receipt = bridge.receipts.get(tid)
                full = f"mcp__{name}__{params.name}"
                if receipt is not None:
                    _, call = receipt
                    if call.name != (
                        params.name if name == "durable" else full
                    ) or call.input != (params.arguments or {}):
                        bridge.failures.append(
                            "MCP callback differs from its native receipt"
                        )
                        raise RuntimeError(bridge.failures[-1])
                    outcome = await bridge.execute(tid)
                    blocks: list[dict[str, Any]] = (
                        outcome.blocks
                        if outcome.blocks is not None
                        else [
                            {
                                "type": "text",
                                "text": outcome.content
                                if isinstance(outcome.content, str)
                                else json.dumps(outcome.content),
                            }
                        ]
                    )
                    blocks = [
                        {
                            "type": "image",
                            "data": b["source"]["data"],
                            "mimeType": b["source"]["media_type"],
                        }
                        if b.get("type") == "image"
                        and b.get("source", {}).get("type") == "base64"
                        else b
                        for b in blocks
                    ]
                    return CallToolResult.model_validate(
                        {"content": blocks, "isError": outcome.is_error}
                    )
                if bridge.managed(full) or session is None:
                    bridge.ran_inside.append(full)
                    bridge.failures.append(
                        "MCP callback has no verified native child receipt"
                    )
                    raise RuntimeError(bridge.failures[-1])
                return await session.call_tool(
                    params.name, params.arguments, meta=ctx.meta
                )
            finally:
                if task is not None:
                    bridge.callbacks.discard(task)

        async def resources(ctx: Any, params: Any) -> Any:
            del ctx
            return await session.list_resources(params=params)

        async def templates(ctx: Any, params: Any) -> Any:
            del ctx
            return await session.list_resource_templates(params=params)

        async def read(ctx: Any, params: Any) -> Any:
            del ctx
            return await session.read_resource(params.uri)

        async def prompts(ctx: Any, params: Any) -> Any:
            del ctx
            return await session.list_prompts(params=params)

        async def prompt(ctx: Any, params: Any) -> Any:
            del ctx
            return await session.get_prompt(params.name, params.arguments)

        forwarding: dict[str, Any] = {}
        capabilities = session.server_capabilities if session is not None else None
        if capabilities is not None and capabilities.resources is not None:
            forwarding.update(
                {
                    "on_list_resources": resources,
                    "on_list_resource_templates": templates,
                    "on_read_resource": read,
                }
            )
        if capabilities is not None and capabilities.prompts is not None:
            forwarding.update(
                {
                    "on_list_prompts": prompts,
                    "on_get_prompt": prompt,
                }
            )
        if session is not None:
            forwarding["instructions"] = session.instructions

        return {
            "type": "sdk",
            "name": name,
            "instance": Server(
                name, on_list_tools=list_tools, on_call_tool=call_tool, **forwarding
            ),
        }
