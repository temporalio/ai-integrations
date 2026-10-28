"""Turn-scoped tool bridge.

Claude calls workflow tools through an in-process MCP server. Each call is one signal to the
workflow, and one broker per turn long-polls the results through a single update.

One poll, not one update per call: Temporal caps in-flight updates per workflow. Signals are
not capped, so requests fan in freely while results come back through one serialized long-poll
held only while a call waits.

Not ``create_sdk_mcp_server``: the handler must carry Claude's ``tool_use_id``. Claude Code
sends that id as ``_meta["claudecode/toolUseId"]`` on ``tools/call``, and the SDK helper hands
its handlers only the arguments. The SDK documents passing any ``mcp.server.Server`` as an
``McpSdkServerConfig``.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from typing import Any, Self

from temporalio.claude_agent_sdk._models import (
    BRIDGE_MCP_SERVER,
    BRIDGE_POLL_UPDATE,
    BRIDGE_REQUEST_SIGNAL,
    ClaudeBridgeBatch,
    ClaudeBridgePoll,
    ClaudeToolDeclaration,
    ClaudeToolRequest,
    ClaudeToolResult,
    tool_call_key,
)
from temporalio.claude_agent_sdk._sdk import load_mcp
from temporalio.client import WorkflowHandle

logger = logging.getLogger(__name__)

TOOL_USE_ID_META = "claudecode/toolUseId"
"""The ``tools/call`` ``_meta`` key under which Claude Code sends the call's ``tool_use_id``."""

_DELIVERY_RECOVERY_SECONDS = 1.0


def _error(tool_use_id: str, message: str) -> ClaudeToolResult:
    return ClaudeToolResult(tool_use_id=tool_use_id, content=message, is_error=True)


class ToolBridgeClient:
    """Sends tool requests to the agent workflow and dispatches its results to waiting calls.

    Use as an async context manager around the Claude run. Every call registered here is
    answered with the workflow's result or an ordinary turn-close error. Uncertain delivery
    wakes Activity supervision instead: the SDK must not turn it into a fabricated tool result.
    """

    def __init__(self, handle: WorkflowHandle[Any, Any], token: str) -> None:
        """Bind the workflow handle and bridge token. See the class docstring."""
        self._handle = handle
        self._token = token
        self._next: Counter[str] = Counter()
        self._unanswered: dict[str, deque[int]] = {}
        self._recorded: dict[str, int] = {}
        self._pending: dict[str, asyncio.Future[ClaudeToolResult]] = {}
        self._waiting = asyncio.Event()
        self._closed: str | None = None
        self._broker: asyncio.Task[None] | None = None
        self.failure: str | None = None
        self._failed = asyncio.Event()

    async def __aenter__(self) -> Self:
        """Start polling the workflow for tool results."""
        self._broker = asyncio.create_task(self._poll())
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the bridge and cancel the poll."""
        self.close("the Claude turn ended")
        if self._broker is not None:
            self._broker.cancel()
            await asyncio.gather(self._broker, return_exceptions=True)

    def number(
        self,
        base: Mapping[str, int],
        stored: Sequence[tuple[str, str, bool]],
        recorded: Mapping[str, tuple[str, int]] | None = None,
        answered: frozenset[str] | None = None,
    ) -> None:
        """Set where ordinals start, before the CLI starts: ``base`` (by call key) after the
        message's earlier runs, then ``stored``, this run's calls in the store's transcript in
        order (id, key, whether it has a result). A call without one is what a resumed CLI
        re-issues, so the next call with its key takes its ordinal.

        With the transcript (``answered``: every id it answers), ``recorded``, the workflow's
        record by id, also counts: new calls number after every recorded slot, and a recorded
        slot no stored result answers (under any of its ids) is what a re-issue takes. A
        transcript window that starts after an earlier attempt's calls cannot hide them then.
        """
        self._next = Counter(base)
        recorded = recorded if answered is not None and recorded else {}
        self._recorded = {
            call_id: ordinal for call_id, (_, ordinal) in recorded.items()
        }
        slots: dict[tuple[str, int], set[str]] = {}
        for call_id, slot in recorded.items():
            slots.setdefault(slot, set()).add(call_id)
        for (key, ordinal), ids in sorted(slots.items()):
            self._next[key] = max(self._next[key], ordinal + 1)
            if not ids & (answered or frozenset()):
                self._unanswered.setdefault(key, deque()).append(ordinal)
        for call_id, key, done in stored:
            if call_id in recorded:
                continue
            if not done:
                self._unanswered.setdefault(key, deque()).append(self._next[key])
            self._next[key] += 1

    @property
    def closed(self) -> bool:
        """Whether the bridge stopped accepting calls."""
        return self._closed is not None

    @property
    def pending(self) -> int:
        """Calls still waiting on a result."""
        return sum(not f.done() for f in self._pending.values())

    async def call(
        self, tool_use_id: str, name: str, arguments: dict[str, Any]
    ) -> ClaudeToolResult:
        """Run one tool call through the workflow and return its model-visible result."""
        if self.failure is not None:
            return await asyncio.Future[ClaudeToolResult]()
        if self._closed is not None:
            return _error(tool_use_id, f"Tool bridge closed: {self._closed}")
        if tool_use_id in self._pending:
            return _error(tool_use_id, f"Duplicate tool call {tool_use_id!r}")
        future: asyncio.Future[ClaudeToolResult] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[tool_use_id] = future
        self._waiting.set()
        key = tool_call_key(name, arguments)
        slots = self._unanswered.get(key)
        if (ordinal := self._recorded.get(tool_use_id)) is not None:
            if slots and ordinal in slots:
                slots.remove(ordinal)
        elif slots:
            ordinal = slots.popleft()
        else:
            ordinal = self._next[key]
            self._next[key] += 1
        try:
            try:
                await self._handle.signal(
                    BRIDGE_REQUEST_SIGNAL,
                    ClaudeToolRequest(
                        token=self._token,
                        tool_use_id=tool_use_id,
                        name=name,
                        arguments=arguments,
                        ordinal=ordinal,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - delivery may have succeeded
                try:
                    async with asyncio.timeout(_DELIVERY_RECOVERY_SECONDS):
                        return await asyncio.shield(future)
                except TimeoutError:
                    self._fail(f"tool request delivery could not be confirmed: {exc}")
                    # Raising here becomes an MCP error. Supervision stops the CLI instead.
                    return await asyncio.Future[ClaudeToolResult]()
            return await future
        finally:
            self._pending.pop(tool_use_id, None)

    def close(self, reason: str) -> None:
        """Stop accepting calls and answer every waiting one with an error naming ``reason``."""
        if self._closed is None:
            self._closed = reason
        if self.failure is not None:
            return
        for tool_use_id, future in self._pending.items():
            if not future.done():
                future.set_result(
                    _error(tool_use_id, f"Tool bridge closed: {self._closed}")
                )

    def _fail(self, reason: str) -> None:
        if self.failure is None:
            self.failure = reason
            self._closed = reason
            self._failed.set()

    async def wait_failure(self) -> None:
        """Wait until the Activity must stop a bridge whose delivery is uncertain."""
        await self._failed.wait()

    async def _poll(self) -> None:
        after = 0
        while self._closed is None:
            await self._waiting.wait()
            try:
                batch = await self._handle.execute_update(
                    BRIDGE_POLL_UPDATE,
                    ClaudeBridgePoll(token=self._token, after=after),
                    result_type=ClaudeBridgeBatch,
                )
            except Exception as exc:  # noqa: BLE001 - supervision stops unanswered calls
                logger.warning("Claude tool bridge poll failed: %r", exc)
                self._fail(f"the agent workflow rejected the tool bridge ({exc})")
                return
            after += len(batch.results)
            for result in batch.results:
                future = self._pending.get(result.tool_use_id)
                if future is not None and not future.done():
                    future.set_result(result)
            if batch.closed:
                self.close("the agent workflow closed the tool bridge")
                return
            if not self.pending:
                self._waiting.clear()


def mcp_server(
    tools: list[ClaudeToolDeclaration], bridge: ToolBridgeClient
) -> dict[str, Any]:
    """An ``McpSdkServerConfig`` listing ``tools`` whose calls all go through ``bridge``."""
    server_module, types = load_mcp()
    listed = types.ListToolsResult(
        tools=[
            types.Tool(
                name=t.name, description=t.description, inputSchema=t.input_schema
            )
            for t in tools
        ]
    )

    async def on_list_tools(ctx: Any, params: Any) -> Any:  # type: ignore
        return listed

    async def on_call_tool(ctx: Any, params: Any) -> Any:  # type: ignore
        meta = params.meta if isinstance(params.meta, dict) else {}
        tool_use_id = meta.get(TOOL_USE_ID_META)
        if not isinstance(tool_use_id, str) or not tool_use_id:
            result = _error("", f"Tool call is missing _meta[{TOOL_USE_ID_META!r}]")
        else:
            result = await bridge.call(
                tool_use_id, params.name, dict(params.arguments or {})
            )
        return types.CallToolResult.model_validate(
            {
                "content": [{"type": "text", "text": result.content}],
                "isError": result.is_error,
            }
        )

    server = server_module.Server(
        BRIDGE_MCP_SERVER,
        version="1.0.0",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
    return {"type": "sdk", "name": BRIDGE_MCP_SERVER, "instance": server}
