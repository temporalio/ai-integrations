"""Unit tests for how the MCP activities evict their pooled connection after a failed call.

Runs the real ``{server}-call-tool`` / ``{server}-list-tools`` activities in an
``ActivityEnvironment`` against a fake MCP session, so no Temporal server or
workflow is needed.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, cast

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent
from strands.tools.mcp import MCPClient

from temporalio.strands_agents import StrandsPlugin
from temporalio.strands_agents import _temporal_mcp_client as mcp_client
from temporalio.testing import ActivityEnvironment


@dataclass
class _FakeMCP:
    """Fake MCP server shared by every connection the factory opens.

    Connections are numbered ``client-1``, ``client-2``, ... in the order they
    are opened. Like the real MCP session, closing a connection fails the
    requests still in flight on it with "Connection closed".
    """

    call_tool: Callable[[str, str], Awaitable[str]]
    list_tools: Callable[[str], Awaitable[None]] | None = None
    opened: list[str] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)
    # Seconds a connection takes to finish closing, like a slow transport teardown.
    close_delay: float = 0

    def client_factory(self) -> MCPClient:
        client_id = f"client-{len(self.opened) + 1}"
        self.opened.append(client_id)

        @asynccontextmanager
        async def transport() -> AsyncGenerator[tuple[str, None], None]:
            yield client_id, None

        # The fake transport yields a connection id where MCP streams belong.
        return MCPClient(cast(Any, transport))


class _FakeSession:
    """Stands in for ``ClientSession``; the "read stream" is the connection id."""

    def __init__(
        self,
        server: _FakeMCP,
        read_stream: str,
        _write_stream: None,
        **_kwargs: Any,
    ) -> None:
        self.server = server
        self._id = read_stream
        self._closed = False
        self._inflight: set[asyncio.Future[Any]] = set()

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._closed = True
        for future in self._inflight:
            if not future.done():
                future.set_exception(RuntimeError("Connection closed"))
        if self.server.close_delay:
            await asyncio.sleep(self.server.close_delay)
        self.server.closed.append(self._id)

    async def initialize(self) -> None:
        return None

    async def _request(self, handler: Awaitable[Any]) -> Any:
        """Send a request to the fake server and wait for its response.

        The server handles the request in its own task and answers through a
        response future, so closing the session can reject a pending request
        without touching the handler.
        """
        if self._closed:
            raise RuntimeError("Connection closed")
        response: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._inflight.add(response)
        task = asyncio.ensure_future(handler)
        task.add_done_callback(partial(self._respond, response))
        try:
            return await response
        finally:
            self._inflight.discard(response)
            task.cancel()

    @staticmethod
    def _respond(response: asyncio.Future[Any], task: asyncio.Task[Any]) -> None:
        if response.done():
            # Rejected or abandoned already; just consume the handler's outcome.
            if not task.cancelled():
                task.exception()
        elif task.cancelled():
            response.cancel()
        elif (error := task.exception()) is not None:
            response.set_exception(error)
        else:
            response.set_result(task.result())

    async def call_tool(self, name: str, _arguments: dict[str, Any]) -> CallToolResult:
        if name == "invalid":
            # Rejected before any request is sent, like client-side argument validation.
            raise ValueError("invalid arguments")
        handled_by = await self._request(self.server.call_tool(self._id, name))
        return CallToolResult(content=[TextContent(type="text", text=handled_by)])

    async def list_tools(self, **_kwargs: Any) -> ListToolsResult:
        assert self.server.list_tools is not None
        await self._request(self.server.list_tools(self._id))
        return ListToolsResult(tools=[])


async def _cleanup(*names: str) -> None:
    for name in names:
        await mcp_client._evict_connection(name)  # pyright: ignore[reportPrivateUsage]


def _fake(
    monkeypatch: pytest.MonkeyPatch,
    call_tool: Callable[[str, str], Awaitable[str]],
    list_tools: Callable[[str], Awaitable[None]] | None = None,
) -> _FakeMCP:
    fake = _FakeMCP(call_tool, list_tools)
    # Replace the MCP ``ClientSession`` the connection owner task opens.
    monkeypatch.setattr(mcp_client, "ClientSession", partial(_FakeSession, fake))
    return fake


async def _until(condition: Callable[[], bool]) -> None:
    """Wait for background connection close tasks to make ``condition`` true."""
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")


async def _settle() -> None:
    """Give background close tasks every chance to run before asserting nothing closed."""
    for _ in range(20):
        await asyncio.sleep(0)


def _text(result: Any) -> str:
    assert result["status"] == "success", result
    return result["content"][0]["text"]


def _args(tool: str) -> Any:
    return mcp_client._CallToolArgs(  # pyright: ignore[reportPrivateUsage]
        tool_name=tool, arguments={}, tool_use_id=f"use-{tool}"
    )


async def test_failed_call_does_not_close_connection_under_in_flight_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = "sibling_server"
    slow_started = asyncio.Event()
    release_slow = asyncio.Event()

    async def call_tool(client_id: str, tool: str) -> str:
        if tool == "slow":
            slow_started.set()
            await release_slow.wait()
        elif tool == "fail":
            await slow_started.wait()
            raise RuntimeError("tool failed")
        return client_id

    fake = _fake(monkeypatch, call_tool)
    call = mcp_client.build_call_tool_activity(server, fake.client_factory)
    try:
        slow = asyncio.ensure_future(ActivityEnvironment().run(call, _args("slow")))
        failed = await ActivityEnvironment().run(call, _args("fail"))
        assert failed["status"] == "error"
        await _settle()
        assert fake.closed == [], "connection must stay open while the slow call runs"
        assert fake.opened == ["client-1"], "both calls should share one connection"

        # The failed connection is not reused: a call made after the failure
        # opens a fresh one.
        assert _text(await ActivityEnvironment().run(call, _args("ok"))) == "client-2"
        assert fake.opened == ["client-1", "client-2"]
        assert fake.closed == [], "broken connection stays open until it drains"

        release_slow.set()
        assert _text(await slow) == "client-1", "the slow call must still succeed"
        await _until(lambda: fake.closed == ["client-1"])
    finally:
        release_slow.set()
        await _cleanup(server)
    assert fake.closed == ["client-1", "client-2"]


async def test_failed_list_tools_does_not_close_connection_under_in_flight_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = "list_fail_server"
    slow_started = asyncio.Event()
    release_slow = asyncio.Event()

    async def call_tool(client_id: str, _tool: str) -> str:
        slow_started.set()
        await release_slow.wait()
        return client_id

    async def list_tools(_client_id: str) -> None:
        await slow_started.wait()
        raise RuntimeError("list failed")

    fake = _fake(monkeypatch, call_tool, list_tools)
    call = mcp_client.build_call_tool_activity(server, fake.client_factory)
    list_activity = mcp_client.build_list_tools_activity(server, fake.client_factory)
    try:
        slow = asyncio.ensure_future(ActivityEnvironment().run(call, _args("slow")))
        with pytest.raises(RuntimeError, match="list failed"):
            await ActivityEnvironment().run(list_activity)
        await _settle()
        assert fake.closed == [], "connection must stay open while the call runs"

        release_slow.set()
        assert _text(await slow) == "client-1"
        await _until(lambda: fake.closed == ["client-1"])
        assert fake.opened == ["client-1"]
    finally:
        release_slow.set()
        await _cleanup(server)


async def test_failed_call_with_nothing_in_flight_closes_connection_and_reconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = "solo_server"

    async def call_tool(client_id: str, tool: str) -> str:
        if tool == "fail":
            raise RuntimeError("tool failed")
        return client_id

    fake = _fake(monkeypatch, call_tool)
    call = mcp_client.build_call_tool_activity(server, fake.client_factory)
    try:
        failed = await ActivityEnvironment().run(call, _args("fail"))
        assert failed["status"] == "error"
        await _until(lambda: fake.closed == ["client-1"])

        assert _text(await ActivityEnvironment().run(call, _args("ok"))) == "client-2"
        assert fake.opened == ["client-1", "client-2"]
    finally:
        await _cleanup(server)
    assert fake.closed == ["client-1", "client-2"]


async def test_dying_old_connection_does_not_unpublish_its_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = "replaced_server"

    async def call_tool(client_id: str, _tool: str) -> str:
        return client_id

    fake = _fake(monkeypatch, call_tool)
    call = mcp_client.build_call_tool_activity(server, fake.client_factory)
    try:
        assert _text(await ActivityEnvironment().run(call, _args("ok"))) == "client-1"
        old = mcp_client._CONNECTIONS.pop(server)  # pyright: ignore[reportPrivateUsage]
        assert _text(await ActivityEnvironment().run(call, _args("ok"))) == "client-2"
        replacement = mcp_client._CONNECTIONS[server]  # pyright: ignore[reportPrivateUsage]
        assert replacement is not old

        # The superseded connection's owner task dies late (as when its
        # transport drops while calls drain). It must not drop the record that
        # replaced it.
        old._owner.cancel()  # pyright: ignore[reportPrivateUsage]
        await old.aclose()
        assert mcp_client._CONNECTIONS.get(server) is replacement  # pyright: ignore[reportPrivateUsage]
    finally:
        await _cleanup(server)


async def test_queued_idle_eviction_does_not_close_replacement_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = "queued_idle_server"

    async def call_tool(client_id: str, _tool: str) -> str:
        return client_id

    fake = _fake(monkeypatch, call_tool)
    call = mcp_client.build_call_tool_activity(server, fake.client_factory)
    try:
        assert _text(await ActivityEnvironment().run(call, _args("ok"))) == "client-1"
        old = mcp_client._CONNECTIONS[server]  # pyright: ignore[reportPrivateUsage]

        # The idle timer's eviction task is queued behind a call that fails
        # (unpublishing and releasing the old record) and a call that then opens
        # a replacement. When it finally runs, the old record looks idle but is
        # no longer the cached connection, so it must leave the replacement alone.
        failing = asyncio.ensure_future(
            ActivityEnvironment().run(call, _args("invalid"))
        )
        healthy = asyncio.ensure_future(ActivityEnvironment().run(call, _args("ok")))
        await asyncio.sleep(0)
        old._on_idle()  # pyright: ignore[reportPrivateUsage]

        assert (await failing)["status"] == "error"
        assert _text(await healthy) == "client-2", "the replacement must not be closed"
        assert fake.opened == ["client-1", "client-2"]
        await _until(lambda: fake.closed == ["client-1"])
    finally:
        await _cleanup(server)
    assert fake.closed == ["client-1", "client-2"]


async def test_plugin_shutdown_waits_for_broken_connection_to_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = "shutdown_server"

    async def call_tool(client_id: str, tool: str) -> str:
        if tool == "fail":
            raise RuntimeError("tool failed")
        return client_id

    fake = _fake(monkeypatch, call_tool)
    fake.close_delay = 0.1
    call = mcp_client.build_call_tool_activity(server, fake.client_factory)
    plugin = StrandsPlugin(models={}, mcp_clients={server: fake.client_factory})
    assert plugin.run_context is not None
    try:
        async with plugin.run_context():
            failed = await ActivityEnvironment().run(call, _args("fail"))
            assert failed["status"] == "error"
        assert fake.closed == ["client-1"], (
            "shutdown returned before the close finished"
        )
    finally:
        await _cleanup(server)
        await _until(lambda: fake.closed == ["client-1"])


async def test_closing_session_fails_in_flight_request() -> None:
    fake = _FakeMCP(call_tool=lambda client_id, _tool: asyncio.sleep(60, client_id))
    async with _FakeSession(fake, "client-1", None) as session:
        pending = asyncio.ensure_future(session.call_tool("slow", {}))
        await asyncio.sleep(0)
        assert not pending.done()
    with pytest.raises(RuntimeError, match="Connection closed"):
        await asyncio.wait_for(pending, timeout=5)
