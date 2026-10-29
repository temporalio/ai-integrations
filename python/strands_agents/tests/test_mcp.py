import asyncio
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from mcp import StdioServerParameters, stdio_client
from strands import tool
from strands.tools.mcp.mcp_client import MCPClient

from temporalio import workflow
from temporalio.client import Client
from temporalio.strands_agents import (
    StrandsPlugin,
    TemporalAgent,
    TemporalMCPClient,
    _temporal_mcp_client,
)
from temporalio.strands_agents._temporal_mcp_tool import TemporalMCPTool
from temporalio.worker import Replayer, Worker
from tests.common import get_activities
from tests.mock_model import MockModel


def _echo_client_factory() -> MCPClient:
    return MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=[str(Path(__file__).parent / "echo_mcp_server.py")],
            )
        )
    )


@workflow.defn
class MCPWorkflow:
    def __init__(self) -> None:
        echo = TemporalMCPClient(
            server="echo",
            cache_tools=True,
            start_to_close_timeout=timedelta(seconds=30),
        )
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=30),
            tools=[echo],
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await self.agent.invoke_async(prompt)
        return str(result)


async def test_mcp(client: Client):
    task_queue = f"test_mcp-{uuid4()}"
    plugin = StrandsPlugin(
        models={
            "mock": lambda: MockModel(
                [
                    {"name": "echo", "input": {"message": "hello"}},
                    "Done!",
                ]
            )
        },
        mcp_clients={
            "echo": lambda: MCPClient(
                lambda: stdio_client(
                    StdioServerParameters(
                        command=sys.executable,
                        args=[str(Path(__file__).parent / "echo_mcp_server.py")],
                    )
                )
            ),
        },
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[MCPWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            MCPWorkflow.run,
            "echo hello",
            id=f"test_mcp_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == "Done!\n"

    history = await handle.fetch_history()
    assert get_activities(history) == [
        "strands_get_model_config",
        "echo-list-tools",
        "invoke_model",
        "echo-call-tool",
        "invoke_model",
    ]

    await Replayer(
        workflows=[MCPWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


@workflow.defn
class MCPReuseWorkflow:
    def __init__(self) -> None:
        echo = TemporalMCPClient(
            server="echo_cached",
            cache_tools=True,
            start_to_close_timeout=timedelta(seconds=30),
        )
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=30),
            tools=[echo],
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await self.agent.invoke_async(prompt)
        return str(result)


async def test_mcp_reuses_connection(client: Client):
    """Successive MCP tool calls reuse one cached worker-side connection."""
    task_queue = f"test_mcp_reuses_connection-{uuid4()}"
    # Count how often the worker opens a connection. One lazily-opened
    # connection serves the list-tools discovery and both tool calls (1);
    # reconnecting per call would make it more.
    factory_calls = [0]

    def counting_factory() -> MCPClient:
        factory_calls[0] += 1
        return _echo_client_factory()

    plugin = StrandsPlugin(
        models={
            "mock": lambda: MockModel(
                [
                    {"name": "echo", "input": {"message": "one"}},
                    {"name": "echo", "input": {"message": "two"}},
                    "Done!",
                ]
            )
        },
        mcp_clients={"echo_cached": counting_factory},
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[MCPReuseWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            MCPReuseWorkflow.run,
            "echo twice",
            id=f"test_mcp_reuses_connection_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == "Done!\n"

    # The worker context has exited, so its run_context finally evicted the
    # cached connection.
    assert "echo_cached" not in _temporal_mcp_client._CONNECTIONS
    assert factory_calls[0] == 1

    history = await handle.fetch_history()
    assert get_activities(history) == [
        "strands_get_model_config",
        "echo_cached-list-tools",
        "invoke_model",
        "echo_cached-call-tool",
        "invoke_model",
        "echo_cached-call-tool",
        "invoke_model",
    ]

    await Replayer(
        workflows=[MCPReuseWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


@workflow.defn
class MCPIdleWorkflow:
    def __init__(self) -> None:
        echo = TemporalMCPClient(
            server="echo_idle",
            cache_tools=True,
            start_to_close_timeout=timedelta(seconds=30),
        )
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=30),
            tools=[echo],
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await self.agent.invoke_async(prompt)
        return str(result)


async def test_mcp_connection_idle_timeout(client: Client):
    """A short idle timeout evicts the cached connection while the worker runs."""
    task_queue = f"test_mcp_connection_idle_timeout-{uuid4()}"
    factory_calls = [0]

    def counting_factory() -> MCPClient:
        factory_calls[0] += 1
        return _echo_client_factory()

    plugin = StrandsPlugin(
        models={
            "mock": lambda: MockModel(
                [
                    {"name": "echo", "input": {"message": "hello"}},
                    "Done!",
                ]
            )
        },
        mcp_clients={"echo_idle": counting_factory},
        # Short window so the cached call connection is evicted mid-run rather
        # than only at worker shutdown. The idle timer only arms once the call
        # releases the connection, so this can't tear it down mid-call.
        mcp_connection_idle_timeout=timedelta(milliseconds=100),
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[MCPIdleWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            MCPIdleWorkflow.run,
            "echo hello",
            id=f"test_mcp_connection_idle_timeout_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == "Done!\n"

        # A connection was opened lazily (on the first list-tools/call-tool).
        # How many times depends on whether the short idle timer fires between
        # activities, so this only asserts that at least one was opened; the
        # eviction-while-alive behavior is what the polling loop below checks.
        assert factory_calls[0] >= 1

        # Still inside the worker context: the short idle timer evicts the
        # cached call connection on its own. Asserting eviction here -- with the
        # worker alive -- proves it came from the idle timer, not shutdown.
        for _ in range(100):
            if "echo_idle" not in _temporal_mcp_client._CONNECTIONS:
                break
            await asyncio.sleep(0.1)
        assert "echo_idle" not in _temporal_mcp_client._CONNECTIONS


@workflow.defn
class MCPNoCacheWorkflow:
    def __init__(self) -> None:
        echo = TemporalMCPClient(
            server="echo_nocache",
            cache_tools=False,
            start_to_close_timeout=timedelta(seconds=30),
        )
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=30),
            tools=[echo],
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await self.agent.invoke_async(prompt)
        return str(result)


async def test_mcp_lists_tools_each_turn_when_uncached(client: Client):
    """With cache_tools=False the tool list is re-fetched on every model call."""
    task_queue = f"test_mcp_lists_tools_each_turn_when_uncached-{uuid4()}"
    plugin = StrandsPlugin(
        models={
            "mock": lambda: MockModel(
                [
                    {"name": "echo", "input": {"message": "one"}},
                    {"name": "echo", "input": {"message": "two"}},
                    "Done!",
                ]
            )
        },
        mcp_clients={"echo_nocache": _echo_client_factory},
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[MCPNoCacheWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            MCPNoCacheWorkflow.run,
            "echo twice",
            id=f"test_mcp_lists_tools_each_turn_when_uncached_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == "Done!\n"

    history = await handle.fetch_history()
    activities = get_activities(history)
    # One list-tools per model call -- the tools are re-listed every turn rather
    # than once for the workflow.
    assert activities.count("echo_nocache-list-tools") == activities.count(
        "invoke_model"
    )
    assert activities.count("echo_nocache-list-tools") == 3

    await Replayer(
        workflows=[MCPNoCacheWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


@workflow.defn
class MCPTwoServersWorkflow:
    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=30),
            tools=[
                TemporalMCPClient(
                    server="echo_a", start_to_close_timeout=timedelta(seconds=30)
                ),
                TemporalMCPClient(
                    server="echo_b", start_to_close_timeout=timedelta(seconds=30)
                ),
            ],
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        try:
            result = await self.agent.invoke_async(prompt)
        except ValueError as err:
            return f"refused: {err}"
        return str(result)


async def test_mcp_duplicate_tool_names_are_refused(client: Client):
    """Two servers that list the same tool name on one agent raise, as in Strands."""
    task_queue = f"test_mcp_duplicate_tool_names_are_refused-{uuid4()}"
    plugin = StrandsPlugin(
        models={"mock": lambda: MockModel(["Done!"])},
        mcp_clients={"echo_a": _echo_client_factory, "echo_b": _echo_client_factory},
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[MCPTwoServersWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            MCPTwoServersWorkflow.run,
            "echo hi",
            id=f"test_mcp_duplicate_tool_names_are_refused_{uuid4()}",
            task_queue=task_queue,
        )
        assert (await handle.result()).startswith(
            "refused: Tool name 'echo' already exists."
        )

    await Replayer(
        workflows=[MCPTwoServersWorkflow],
        plugins=[plugin],
    ).replay_workflow(await handle.fetch_history())


def _mcp_tool(server: str, name: str) -> TemporalMCPTool:
    info = _temporal_mcp_client._MCPToolInfo(
        name=name, description="", input_schema={}, agent_name=name
    )
    return TemporalMCPTool(server, info, {})


async def _run_hook(hook: Any, event: Any) -> None:
    """Run a BeforeModelCallEvent hook that TemporalAgent registered."""
    await hook(event)


def _listing(provider: TemporalMCPClient, *names: str) -> Any:
    """Stand in for {server}-list-tools; each call makes new tools, as _refresh() does."""

    async def refresh() -> None:
        provider._tools = [_mcp_tool(provider.server, name) for name in names]
        provider._fetched = True

    return refresh


@pytest.mark.parametrize("cache_tools", [False, True])
async def test_mcp_duplicate_refused_again_after_caught(
    monkeypatch: pytest.MonkeyPatch, cache_tools: bool
):
    """Catching the duplicate-name error does not let the next turn take the name."""
    a = TemporalMCPClient("a", cache_tools=cache_tools)
    c = TemporalMCPClient("c", cache_tools=cache_tools)
    agent = TemporalAgent(model="m", tools=[a, c])
    event = SimpleNamespace(agent=agent)
    hook_a = agent._make_mcp_refresh_hook(a)
    hook_c = agent._make_mcp_refresh_hook(c)
    monkeypatch.setattr(a, "_refresh", _listing(a, "echo"))
    monkeypatch.setattr(c, "_refresh", _listing(c, "echo"))

    for _turn in range(2):
        await _run_hook(hook_a, event)
        with pytest.raises(ValueError, match="Tool name 'echo' already exists"):
            await _run_hook(hook_c, event)
        assert agent.tool_registry.registry["echo"]._server == "a"  # type: ignore[attr-defined]


async def test_mcp_shared_client_in_two_agents(monkeypatch: pytest.MonkeyPatch):
    """Agents that share one client each keep its tools across the other's refresh."""
    a = TemporalMCPClient("a")
    first = TemporalAgent(model="m", tools=[a])
    second = TemporalAgent(model="m", tools=[a])
    hook_first = first._make_mcp_refresh_hook(a)
    hook_second = second._make_mcp_refresh_hook(a)
    monkeypatch.setattr(a, "_refresh", _listing(a, "echo"))

    await _run_hook(hook_first, SimpleNamespace(agent=first))
    await _run_hook(hook_second, SimpleNamespace(agent=second))
    await _run_hook(hook_first, SimpleNamespace(agent=first))
    for agent in (first, second):
        assert agent.tool_registry.registry["echo"]._server == "a"  # type: ignore[attr-defined]

    # An agent built after the client listed starts with load_tools()'s tools.
    third = TemporalAgent(model="m", tools=[a])
    await _run_hook(third._make_mcp_refresh_hook(a), SimpleNamespace(agent=third))
    assert third.tool_registry.registry["echo"]._server == "a"  # type: ignore[attr-defined]


@pytest.mark.parametrize("cache_tools", [False, True])
async def test_mcp_refused_listing_registers_nothing(
    monkeypatch: pytest.MonkeyPatch, cache_tools: bool
):
    """A name Strands refuses mid-listing undoes the whole listing, so it can recover."""
    h = TemporalMCPClient("h")
    n = TemporalMCPClient("n", cache_tools=cache_tools)
    agent = TemporalAgent(model="m", tools=[h, n])
    event = SimpleNamespace(agent=agent)
    hook_h = agent._make_mcp_refresh_hook(h)
    hook_n = agent._make_mcp_refresh_hook(n)
    monkeypatch.setattr(h, "_refresh", _listing(h, "foo-bar"))
    monkeypatch.setattr(n, "_refresh", _listing(n, "safe", "foo_bar"))

    await _run_hook(hook_h, event)
    for _turn in range(2):
        with pytest.raises(ValueError, match="'foo_bar' already exists as 'foo-bar'"):
            await _run_hook(hook_n, event)
        assert sorted(agent.tool_registry.registry) == ["foo-bar"]

    # The server drops foo_bar; safe registers and stays n's on the next turns.
    monkeypatch.setattr(n, "_refresh", _listing(n, "safe"))
    for _turn in range(2):
        await _run_hook(hook_h, event)
        await _run_hook(hook_n, event)
        assert sorted(agent.tool_registry.registry) == ["foo-bar", "safe"]
        assert agent.tool_registry.registry["safe"]._server == "n"  # type: ignore[attr-defined]


@tool(name="echo")
def _local_echo(message: str) -> str:
    """Return LOCAL:<message>."""
    return f"LOCAL:{message}"


async def test_mcp_shared_cached_client_keeps_refusing(monkeypatch: pytest.MonkeyPatch):
    """A refusal stays with the agent that refused; the other keeps what it accepted."""
    shared = TemporalMCPClient("s", cache_tools=True)
    first = TemporalAgent(model="m", tools=[_local_echo, shared])
    second = TemporalAgent(model="m", tools=[shared])
    hook_first = first._make_mcp_refresh_hook(shared)
    hook_second = second._make_mcp_refresh_hook(shared)
    catalog = ["echo"]
    listings: list[str] = []
    held = asyncio.Event()

    async def refresh() -> None:
        listings.append("s")
        names = list(catalog)
        if len(listings) == 1:
            await held.wait()  # the first agent's listing completes last
        await _listing(shared, *names)()

    monkeypatch.setattr(shared, "_refresh", refresh)

    async def first_turn() -> None:
        with pytest.raises(ValueError, match="Tool name 'echo' already exists"):
            await _run_hook(hook_first, SimpleNamespace(agent=first))

    pending = asyncio.create_task(first_turn())
    await asyncio.sleep(0)
    await _run_hook(hook_second, SimpleNamespace(agent=second))
    held.set()
    await pending
    accepted = second.tool_registry.registry["echo"]
    assert accepted._server == "s"  # type: ignore[attr-defined]

    # The server now lists `other` instead. The second agent keeps its cached
    # `echo` without listing again; the first lists again and still refuses.
    catalog[:] = ["echo", "other"]
    await _run_hook(hook_second, SimpleNamespace(agent=second))
    await first_turn()
    assert second.tool_registry.registry["echo"] is accepted
    assert first.tool_registry.registry["echo"] is _local_echo
    assert listings == ["s", "s", "s"]

    # Once the server stops listing `echo`, the first agent takes `other`.
    catalog[:] = ["other"]
    await _run_hook(hook_first, SimpleNamespace(agent=first))
    await _run_hook(hook_second, SimpleNamespace(agent=second))
    assert sorted(first.tool_registry.registry) == ["echo", "other"]
    assert first.tool_registry.registry["echo"] is _local_echo
    assert sorted(second.tool_registry.registry) == ["echo"]
    assert second.tool_registry.registry["echo"] is accepted
    assert listings == ["s", "s", "s", "s"]


@pytest.mark.parametrize("order", ["ab", "ba"])
async def test_mcp_name_moves_between_servers(
    monkeypatch: pytest.MonkeyPatch, order: str
):
    """A name one server drops and another adds in the same turn moves, in either order."""
    a = TemporalMCPClient("a")
    b = TemporalMCPClient("b")
    providers = [a, b] if order == "ab" else [b, a]
    agent = TemporalAgent(model="m", tools=providers)
    event = SimpleNamespace(agent=agent)
    hooks = [agent._make_mcp_refresh_hook(p) for p in providers]

    async def turn(a_names: tuple[str, ...], b_names: tuple[str, ...]) -> None:
        monkeypatch.setattr(a, "_refresh", _listing(a, *a_names))
        monkeypatch.setattr(b, "_refresh", _listing(b, *b_names))
        for hook in hooks:
            await _run_hook(hook, event)

    def servers() -> dict[str, str]:
        return {n: t._server for n, t in agent.tool_registry.registry.items()}  # type: ignore[attr-defined]

    await turn(("safe",), ("echo",))
    for _turn in range(2):
        await turn(("safe", "echo"), ("other",))
        assert servers() == {"safe": "a", "echo": "a", "other": "b"}
    # Two servers swap names in one turn.
    await turn(("safe", "other"), ("echo",))
    assert servers() == {"safe": "a", "other": "a", "echo": "b"}

    # A name both servers list stays refused on every turn, and nothing changes.
    for _turn in range(2):
        with pytest.raises(ValueError, match="Tool name 'echo' already exists"):
            await turn(("safe", "other", "echo"), ("echo",))
        assert servers() == {"safe": "a", "other": "a", "echo": "b"}
    await turn(("safe", "other"), ("echo",))
    assert servers() == {"safe": "a", "other": "a", "echo": "b"}


async def test_mcp_failed_listing_lists_again(monkeypatch: pytest.MonkeyPatch):
    """A failed listing of a later server leaves the earlier listing unaccepted."""
    a = TemporalMCPClient("a", cache_tools=True)
    b = TemporalMCPClient("b")
    agent = TemporalAgent(model="m", tools=[a, b])
    event = SimpleNamespace(agent=agent)
    hook_a = agent._make_mcp_refresh_hook(a)
    hook_b = agent._make_mcp_refresh_hook(b)
    monkeypatch.setattr(a, "_refresh", _listing(a, "echo"))
    monkeypatch.setattr(b, "_refresh", _listing(b, "echo"))
    await _run_hook(hook_b, event)  # b holds echo; a has not listed yet

    async def unavailable() -> None:
        raise RuntimeError("b-list-tools failed")

    monkeypatch.setattr(b, "_refresh", unavailable)
    with pytest.raises(RuntimeError, match="b-list-tools failed"):
        await _run_hook(hook_a, event)
    assert agent.tool_registry.registry["echo"]._server == "b"  # type: ignore[attr-defined]

    # b now drops echo; a lists again (cache_tools=True) and takes it.
    monkeypatch.setattr(b, "_refresh", _listing(b, "other"))
    await _run_hook(hook_a, event)
    await _run_hook(hook_b, event)
    assert sorted(agent.tool_registry.registry) == ["echo", "other"]
    assert agent.tool_registry.registry["echo"]._server == "a"  # type: ignore[attr-defined]
