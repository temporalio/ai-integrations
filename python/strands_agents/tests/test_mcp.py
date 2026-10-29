import asyncio
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from mcp import StdioServerParameters, stdio_client
from strands import tool
from strands.hooks import BeforeModelCallEvent
from strands.tools.mcp.mcp_client import MCPClient

from temporalio import workflow
from temporalio.client import Client, WorkflowHistory
from temporalio.strands_agents import (
    StrandsPlugin,
    TemporalAgent,
    TemporalMCPClient,
    TemporalSandbox,
    _temporal_mcp_client,
)
from temporalio.strands_agents._temporal_mcp_tool import TemporalMCPTool
from temporalio.worker import Replayer, Worker
from temporalio.workflow import NondeterminismError
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
            "refused: Tool name 'echo' already exists. "
            "MCP servers 'echo_a' and 'echo_b' both list it."
        )

    history = await handle.fetch_history()
    assert get_activities(history) == [
        "strands_get_model_config",
        "echo_a-list-tools",
        "echo_b-list-tools",
    ]
    await Replayer(
        workflows=[MCPTwoServersWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


def _legacy_history(name: str) -> WorkflowHistory:
    """A history of MCPTwoServersWorkflow recorded before duplicates were refused.

    Recorded on main at 63d7aec against a local dev server; only the worker
    identity and task queue name were normalized. ``distinct_names``: echo_b's
    MCPClient has ``prefix="b"``, and the model calls ``echo``, then ``b_echo``.
    ``duplicate_names``: no prefixes, and the model calls ``echo``, which ran on
    echo_b.
    """
    path = Path(__file__).parent / "histories" / f"legacy_mcp_{name}.json"
    return WorkflowHistory.from_json(name, path.read_text())


async def test_mcp_replays_history_with_distinct_names():
    """Runs whose tool names never clashed schedule the same commands as before."""
    await Replayer(
        workflows=[MCPTwoServersWorkflow], plugins=[StrandsPlugin(models={})]
    ).replay_workflow(_legacy_history("distinct_names"))


async def test_mcp_rejects_history_with_duplicate_names():
    """A run that met a duplicate name now refuses it, so its history no longer replays."""
    # The run now completes where it scheduled the model call.
    with pytest.raises(
        NondeterminismError,
        match="Complete workflow machine does not handle this event.*ActivityTaskScheduled",
    ):
        await Replayer(
            workflows=[MCPTwoServersWorkflow], plugins=[StrandsPlugin(models={})]
        ).replay_workflow(_legacy_history("duplicate_names"))


def _mcp_tool(server: str, name: str) -> TemporalMCPTool:
    info = _temporal_mcp_client._MCPToolInfo(
        name=name, description="", input_schema={}, agent_name=name
    )
    return TemporalMCPTool(server, info, {})


async def _before_model_call(agent: TemporalAgent) -> None:
    """Run the agent's hooks as Strands does before each model call."""
    await agent.hooks.invoke_callbacks_async(BeforeModelCallEvent(agent=agent))


def _listing(
    provider: TemporalMCPClient, *names: str, log: list[str] | None = None
) -> Any:
    """Stand in for {server}-list-tools; each call makes new tools, as _refresh() does."""

    async def refresh() -> None:
        if log is not None:
            log.append(provider.server)
        provider._tools = [_mcp_tool(provider.server, name) for name in names]
        provider._fetched = True

    return refresh


def _sources(agent: TemporalAgent, *providers: TemporalMCPClient) -> dict[str, str]:
    """For each tool of the agent, the server whose latest listing it comes from."""
    latest = {id(tool): p.server for p in providers for tool in p._tools}
    return {
        name: latest.get(id(tool), "other")
        for name, tool in agent.tool_registry.registry.items()
    }


@tool(name="echo")
def _local_echo(message: str) -> str:
    """Return LOCAL:<message>."""
    return f"LOCAL:{message}"


@pytest.mark.parametrize("cache_tools", [False, True])
async def test_mcp_duplicate_refused_again_after_caught(
    monkeypatch: pytest.MonkeyPatch, cache_tools: bool
):
    """Catching the duplicate-name error does not let the next turn take the name."""
    a = TemporalMCPClient("a", cache_tools=cache_tools)
    c = TemporalMCPClient("c", cache_tools=cache_tools)
    agent = TemporalAgent(model="m", tools=[a, c])
    log: list[str] = []
    monkeypatch.setattr(a, "_refresh", _listing(a, "echo", log=log))
    monkeypatch.setattr(c, "_refresh", _listing(c, "echo", log=log))

    for _turn in range(2):
        with pytest.raises(
            ValueError,
            match="Tool name 'echo' already exists. MCP servers 'a' and 'c' both list it.",
        ):
            await _before_model_call(agent)
        assert agent.tool_registry.registry == {}
    # A cached listing is checked again, not listed again.
    assert log == (["a", "c"] if cache_tools else ["a", "c", "a", "c"])


async def test_mcp_server_listing_a_name_twice_is_refused(
    monkeypatch: pytest.MonkeyPatch,
):
    """Two tools with one name from one server are refused too."""
    a = TemporalMCPClient("a")
    agent = TemporalAgent(model="m", tools=[a])
    monkeypatch.setattr(a, "_refresh", _listing(a, "search", "search"))

    with pytest.raises(
        ValueError,
        match="Tool name 'search' already exists. MCP server 'a' lists it twice.",
    ):
        await _before_model_call(agent)


async def test_mcp_same_client_twice_on_one_agent(monkeypatch: pytest.MonkeyPatch):
    """A client passed twice is listed twice per turn, as before, and is not a duplicate."""
    a = TemporalMCPClient("a")
    agent = TemporalAgent(model="m", tools=[a, a])
    log: list[str] = []
    monkeypatch.setattr(a, "_refresh", _listing(a, "echo", log=log))

    for _turn in range(2):
        await _before_model_call(agent)
        assert _sources(agent, a) == {"echo": "a"}
    assert log == ["a"] * 4


async def test_mcp_shared_client_in_two_agents(monkeypatch: pytest.MonkeyPatch):
    """Agents that share one client each keep its tools across the other's refresh."""
    a = TemporalMCPClient("a")
    first = TemporalAgent(model="m", tools=[a])
    second = TemporalAgent(model="m", tools=[a])
    monkeypatch.setattr(a, "_refresh", _listing(a, "echo"))

    await _before_model_call(first)
    await _before_model_call(second)
    accepted = second.tool_registry.registry["echo"]
    await _before_model_call(first)
    assert _sources(first, a) == {"echo": "a"}
    assert second.tool_registry.registry == {"echo": accepted}

    # An agent built after the client listed starts with load_tools()'s tools.
    third = TemporalAgent(model="m", tools=[a])
    await _before_model_call(third)
    assert _sources(third, a) == {"echo": "a"}


async def test_mcp_shared_cached_client(monkeypatch: pytest.MonkeyPatch):
    """Agents that share a cached client use its one listing; a refusal stays with its agent."""
    shared = TemporalMCPClient("s", cache_tools=True)
    first = TemporalAgent(model="m", tools=[_local_echo, shared])
    second = TemporalAgent(model="m", tools=[shared])
    log: list[str] = []
    monkeypatch.setattr(shared, "_refresh", _listing(shared, "echo", "other", log=log))

    for _turn in range(2):
        with pytest.raises(
            ValueError, match="MCP server 's' lists it, and another tool of the agent"
        ):
            await _before_model_call(first)
        await _before_model_call(second)
        assert first.tool_registry.registry == {"echo": _local_echo}
        assert _sources(second, shared) == {"echo": "s", "other": "s"}
    assert log == ["s"]


@pytest.mark.parametrize("cache_tools", [False, True])
async def test_mcp_refused_listing_registers_nothing(
    monkeypatch: pytest.MonkeyPatch, cache_tools: bool
):
    """A name that differs from another only by '-' or '_' refuses the whole turn."""
    h = TemporalMCPClient("h")
    n = TemporalMCPClient("n", cache_tools=cache_tools)
    agent = TemporalAgent(model="m", tools=[h, n])
    monkeypatch.setattr(h, "_refresh", _listing(h, "foo-bar"))
    monkeypatch.setattr(n, "_refresh", _listing(n, "safe", "foo_bar"))

    for _turn in range(2):
        with pytest.raises(
            ValueError,
            match="Tool name 'foo_bar' already exists as 'foo-bar'. "
            "MCP servers 'h' and 'n' both list it.",
        ):
            await _before_model_call(agent)
        assert agent.tool_registry.registry == {}

    # h drops foo-bar; n's tools register and stay n's on the next turns.
    monkeypatch.setattr(h, "_refresh", _listing(h, "other"))
    for _turn in range(2):
        await _before_model_call(agent)
        assert _sources(agent, h, n) == {"other": "h", "safe": "n", "foo_bar": "n"}


@pytest.mark.parametrize("order", ["ab", "ba"])
async def test_mcp_name_moves_between_servers(
    monkeypatch: pytest.MonkeyPatch, order: str
):
    """A name one server drops and another adds in the same turn moves, in either order."""
    a = TemporalMCPClient("a")
    b = TemporalMCPClient("b")
    agent = TemporalAgent(model="m", tools=[a, b] if order == "ab" else [b, a])

    async def turn(a_names: tuple[str, ...], b_names: tuple[str, ...]) -> None:
        monkeypatch.setattr(a, "_refresh", _listing(a, *a_names))
        monkeypatch.setattr(b, "_refresh", _listing(b, *b_names))
        await _before_model_call(agent)

    await turn(("safe",), ("echo",))
    for _turn in range(2):
        await turn(("safe", "echo"), ("other",))
        assert _sources(agent, a, b) == {"safe": "a", "echo": "a", "other": "b"}
    # Two servers swap names in one turn.
    await turn(("safe", "other"), ("echo",))
    assert _sources(agent, a, b) == {"safe": "a", "other": "a", "echo": "b"}

    # A name both servers list stays refused on every turn, and nothing changes.
    before = dict(agent.tool_registry.registry)
    for _turn in range(2):
        with pytest.raises(ValueError, match="Tool name 'echo' already exists"):
            await turn(("safe", "other", "echo"), ("echo",))
        assert agent.tool_registry.registry == before
    await turn(("safe", "other"), ("echo",))
    assert _sources(agent, a, b) == {"safe": "a", "other": "a", "echo": "b"}


async def test_mcp_failed_listing_changes_nothing(monkeypatch: pytest.MonkeyPatch):
    """A failed listing leaves the tools as they were; the next turn lists what is due."""
    a = TemporalMCPClient("a", cache_tools=True)
    b = TemporalMCPClient("b")
    agent = TemporalAgent(model="m", tools=[a, b])
    log: list[str] = []
    monkeypatch.setattr(a, "_refresh", _listing(a, "one", log=log))

    async def unavailable() -> None:
        log.append("b")
        raise RuntimeError("b-list-tools failed")

    monkeypatch.setattr(b, "_refresh", unavailable)
    with pytest.raises(RuntimeError, match="b-list-tools failed"):
        await _before_model_call(agent)
    assert agent.tool_registry.registry == {}

    monkeypatch.setattr(b, "_refresh", _listing(b, "two", log=log))
    await _before_model_call(agent)
    assert _sources(agent, a, b) == {"one": "a", "two": "b"}
    assert log == ["a", "b", "b"]


async def test_mcp_listing_replaces_a_tool_replaced_in_place(
    monkeypatch: pytest.MonkeyPatch,
):
    """A server's tool replaced in place stays the server's: listed again, it is replaced."""
    a = TemporalMCPClient("a")
    agent = TemporalAgent(model="m", tools=[a])
    monkeypatch.setattr(a, "_refresh", _listing(a, "echo"))

    await _before_model_call(agent)
    agent.tool_registry.replace(_local_echo)  # for example a logging wrapper
    await _before_model_call(agent)
    assert _sources(agent, a) == {"echo": "a"}

    agent.tool_registry.replace(_local_echo)
    monkeypatch.setattr(a, "_refresh", _listing(a))
    await _before_model_call(agent)
    assert agent.tool_registry.registry == {}


async def test_mcp_tool_takes_a_sandbox_tool_name(monkeypatch: pytest.MonkeyPatch):
    """A listed tool takes the name of a sandbox tool, as a tool in tools=[...] does."""
    a = TemporalMCPClient("a")
    agent = TemporalAgent(model="m", tools=[a], sandbox=TemporalSandbox("sb"))
    monkeypatch.setattr(a, "_refresh", _listing(a, "sandbox_shell"))

    await _before_model_call(agent)
    assert _sources(agent, a) == {"sandbox_file_editor": "other", "sandbox_shell": "a"}


async def test_mcp_tool_named_like_a_dynamic_tool_is_refused(
    monkeypatch: pytest.MonkeyPatch,
):
    """A listed name held by a dynamic tool, such as structured output's, is refused."""
    a = TemporalMCPClient("a")
    agent = TemporalAgent(model="m", tools=[a])
    agent.tool_registry.register_dynamic_tool(_local_echo)
    monkeypatch.setattr(a, "_refresh", _listing(a, "echo"))

    with pytest.raises(
        ValueError, match="MCP server 'a' lists it, and another tool of the agent"
    ):
        await _before_model_call(agent)
    assert agent.tool_registry.registry == {}
