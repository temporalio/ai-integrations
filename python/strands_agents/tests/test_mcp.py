import asyncio
import re
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from mcp import Client as MCPClient
from mcp import StdioServerParameters, stdio_client
from mcp.types import ListToolsResult, Tool
from strands import tool
from strands.hooks import BeforeModelCallEvent
from strands.types.exceptions import EventLoopException

from temporalio import activity, workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
from temporalio.mcp import TemporalMCPClient as DurableMCPClient
from temporalio.strands_agents import (
    MCPToolNameConflictError,
    StrandsPlugin,
    TemporalAgent,
    TemporalMCPClient,
    TemporalSandbox,
)
from temporalio.strands_agents._temporal_mcp_tool import TemporalMCPTool
from temporalio.worker import Replayer, Worker
from temporalio.workflow import NondeterminismError
from tests.common import get_activities, load_history
from tests.mock_model import MockModel

_ECHO_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).parent / "echo_mcp_server.py")],
)


def _echo_client_factory() -> MCPClient:
    return MCPClient(stdio_client(_ECHO_PARAMS))


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

    # A default that captures the server's parameters, as in a loop over
    # servers; the plugin must still call the factory with no arguments.
    def echo_factory(params: StdioServerParameters = _ECHO_PARAMS) -> MCPClient:
        return MCPClient(stdio_client(params))

    plugin = StrandsPlugin(
        models={
            "mock": lambda: MockModel(
                [
                    {"name": "echo", "input": {"message": "hello"}},
                    "Done!",
                ]
            )
        },
        mcp_clients={"echo": echo_factory},
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
        "temporalio.mcp.echo.list-tools",
        "invoke_model",
        "temporalio.mcp.echo.call-tool",
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
    # Count how often the worker opens and closes a connection. One
    # lazily-opened connection serves the list-tools discovery and both tool
    # calls (1); reconnecting per call would make it more.
    factory_calls = [0]
    closed = asyncio.Event()

    class ClosingClient(MCPClient):
        async def __aexit__(self, *args: Any) -> None:
            await super().__aexit__(*args)
            closed.set()

    def counting_factory() -> MCPClient:
        factory_calls[0] += 1
        return ClosingClient(stdio_client(_ECHO_PARAMS))

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
        assert not closed.is_set()

    assert factory_calls[0] == 1
    # Worker shutdown closes the cached connection. Leaving ``async with Worker``
    # does not wait for that, so wait for it here.
    await asyncio.wait_for(closed.wait(), timeout=10)

    history = await handle.fetch_history()
    assert get_activities(history) == [
        "strands_get_model_config",
        "temporalio.mcp.echo_cached.list-tools",
        "invoke_model",
        "temporalio.mcp.echo_cached.call-tool",
        "invoke_model",
        "temporalio.mcp.echo_cached.call-tool",
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
    """A zero idle timeout closes the connection after every activity."""
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
        mcp_connection_idle_timeout=timedelta(0),
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

    # One connection for list-tools and another for call-tool.
    assert factory_calls[0] == 2


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
    list_tools = "temporalio.mcp.echo_nocache.list-tools"
    assert activities.count(list_tools) == activities.count("invoke_model")
    assert activities.count(list_tools) == 3

    await Replayer(
        workflows=[MCPNoCacheWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


def _two_servers_agent(prefix_b: str | None = None) -> TemporalAgent:
    return TemporalAgent(
        model="mock",
        start_to_close_timeout=timedelta(seconds=30),
        tools=[
            TemporalMCPClient(
                server="echo_a", start_to_close_timeout=timedelta(seconds=30)
            ),
            TemporalMCPClient(
                server="echo_b",
                prefix=prefix_b,
                start_to_close_timeout=timedelta(seconds=30),
            ),
        ],
    )


@workflow.defn
class MCPTwoServersWorkflow:
    def __init__(self) -> None:
        self.agent = _two_servers_agent()

    @workflow.run
    async def run(self, prompt: str) -> str:
        try:
            result = await self.agent.invoke_async(prompt)
        except ValueError as err:
            return f"refused: {err}"
        return str(result)


@workflow.defn
class MCPPrefixedServersWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return str(await _two_servers_agent(prefix_b="b").invoke_async(prompt))


@workflow.defn
class MCPTwoServersUncaughtWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return str(await _two_servers_agent().invoke_async(prompt))


async def test_mcp_duplicate_tool_names_are_refused(client: Client):
    """Two servers that list the same tool name on one agent raise, as in Strands.

    Uncaught, the error fails the Workflow instead of retrying its Workflow Task,
    and is non-retryable with the plugin only on the Worker.
    """
    task_queue = f"test_mcp_duplicate_tool_names_are_refused-{uuid4()}"
    plugin = StrandsPlugin(
        models={"mock": lambda: MockModel(["Done!"])},
        mcp_clients={"echo_a": _echo_client_factory, "echo_b": _echo_client_factory},
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[MCPTwoServersWorkflow, MCPTwoServersUncaughtWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            MCPTwoServersWorkflow.run,
            "echo hi",
            id=f"test_mcp_duplicate_tool_names_are_refused_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == (
            "refused: Tool name 'echo' already exists. "
            "MCP servers 'echo_a' and 'echo_b' both list it. Set a prefix (or a "
            "different one) on one server's TemporalMCPClient, or drop the tool "
            "from one with its tool_filters."
        )
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                MCPTwoServersUncaughtWorkflow.run,
                "echo hi",
                id=f"test_mcp_duplicate_tool_names_uncaught_{uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=30),
            )
    cause = exc_info.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "MCPToolNameConflictError"
    assert cause.non_retryable

    history = await handle.fetch_history()
    assert get_activities(history) == [
        "strands_get_model_config",
        "temporalio.mcp.echo_a.list-tools",
        "temporalio.mcp.echo_b.list-tools",
    ]
    await Replayer(
        workflows=[MCPTwoServersWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


# Histories recorded on main at 275b1e6, after the MCP Activities moved to
# temporalio-mcp and before duplicates were refused, against a local dev server;
# only the worker identity and task queue name were normalized.
# legacy_mcp_distinct_names: MCPPrefixedServersWorkflow, where echo_b's
# TemporalMCPClient has prefix="b", and the model calls echo, then b_echo.
# legacy_mcp_duplicate_names: MCPTwoServersWorkflow, with no prefixes, and the
# model calls echo, which ran on echo_b.


async def test_mcp_replays_history_with_distinct_names():
    """Runs whose tool names never clashed schedule the same commands as before."""
    await Replayer(
        workflows=[MCPPrefixedServersWorkflow], plugins=[StrandsPlugin(models={})]
    ).replay_workflow(load_history("legacy_mcp_distinct_names"))


async def test_mcp_rejects_history_with_duplicate_names():
    """A run that met a duplicate name now refuses it, so its history no longer replays."""
    # The run now completes where it scheduled the model call.
    with pytest.raises(NondeterminismError, match="ActivityTaskScheduled"):
        await Replayer(
            workflows=[MCPTwoServersWorkflow], plugins=[StrandsPlugin(models={})]
        ).replay_workflow(load_history("legacy_mcp_duplicate_names"))


@workflow.defn
class MCPSharedClientWorkflow:
    @workflow.run
    async def run(self) -> str:
        shared = TemporalMCPClient(
            "shifting", start_to_close_timeout=timedelta(seconds=30)
        )
        first, second = (
            TemporalAgent(
                model="mock",
                start_to_close_timeout=timedelta(seconds=30),
                tools=[shared],
            )
            for _agent in range(2)
        )
        results = [await agent.invoke_async("go") for agent in (first, second, first)]
        return " | ".join(str(result) for result in results)


@workflow.defn
class MCPMovingToolWorkflow:
    @workflow.run
    async def run(self) -> str:
        agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=30),
            tools=[
                TemporalMCPClient("mv_b", start_to_close_timeout=timedelta(seconds=30)),
                TemporalMCPClient("mv_a", start_to_close_timeout=timedelta(seconds=30)),
            ],
        )
        results = [await agent.invoke_async("go") for _turn in range(2)]
        return " | ".join(str(result) for result in results)


# Histories recorded on main at 275b1e6, as above, with stand-in list-tools and
# call-tool Activities whose tools change between listings.
# legacy_mcp_shared_client_drop: `shifting` lists echo, then other; agents A, B,
# A share it, and A calls echo on its second turn, which main kept for it.
# legacy_mcp_tool_moves: turn one, mv_a lists safe and echo, mv_b other; turn
# two, mv_a lists safe and other, mv_b echo. Main lost echo that turn, so the
# model's call to it found no tool.
@pytest.mark.parametrize(
    ("workflow_class", "name", "swap"),
    [
        (
            MCPSharedClientWorkflow,
            "legacy_mcp_shared_client_drop",
            r"'temporalio\.mcp\.shifting\.call-tool'.*'temporalio\.mcp\.shifting\.list-tools'",
        ),
        (
            MCPMovingToolWorkflow,
            "legacy_mcp_tool_moves",
            r"'temporalio\.mcp\.mv_b\.list-tools'.*'temporalio\.mcp\.mv_b\.call-tool'",
        ),
    ],
)
async def test_mcp_rejects_history_with_changed_tools(
    workflow_class: type, name: str, swap: str
):
    """Runs whose servers changed their tools now keep the right ones, so they differ."""
    # The recorded Activity and the one the run now schedules.
    with pytest.raises(NondeterminismError, match=swap):
        await Replayer(
            workflows=[workflow_class], plugins=[StrandsPlugin(models={})]
        ).replay_workflow(load_history(name))


@tool
def ping() -> str:
    """Return pong."""
    return "pong"


_late_listings: Counter[str] = Counter()


@activity.defn(name="temporalio.mcp.late.list-tools")
async def _late_list_tools(
    request: dict[str, Any],  # pyright: ignore[reportUnusedParameter]
) -> dict[str, Any]:
    """List nothing on a run's first call, then a tool named like `ping`."""
    run = activity.info().workflow_id or ""
    _late_listings[run] += 1
    if _late_listings[run] == 1:
        return {"tools": []}
    schema = {"type": "object", "properties": {}}
    return {"tools": [{"name": "ping", "inputSchema": schema}]}


@workflow.defn
class MCPConflictAfterToolWorkflow:
    @workflow.run
    async def run(self, catch: bool) -> str:
        agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=30),
            tools=[
                ping,
                TemporalMCPClient("late", start_to_close_timeout=timedelta(seconds=30)),
            ],
        )
        try:
            return str(await agent.invoke_async("ping"))
        except MCPToolNameConflictError as err:
            if not catch:
                raise
            return f"refused: {err}"


@pytest.mark.parametrize("plugin_on", ["worker", "client"])
async def test_mcp_conflict_after_a_tool_call(client: Client, plugin_on: str):
    """A refusal after a tool call reaches Workflow code as itself, not wrapped by Strands.

    Uncaught, it fails the Workflow without a retry, with the plugin on the
    Worker only or on the client.
    """
    task_queue = f"test_mcp_conflict_after_a_tool_call-{uuid4()}"
    plugin = StrandsPlugin(
        models={"mock": lambda: MockModel([{"name": "ping", "input": {}}] * 2)}
    )
    worker_plugins = [plugin]
    if plugin_on == "client":
        config = client.config()
        config["plugins"] = [*config["plugins"], plugin]
        client = Client(**config)
        worker_plugins = []

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[MCPConflictAfterToolWorkflow],
        activities=[_late_list_tools],
        plugins=worker_plugins,
        max_cached_workflows=0,
    ):
        for catch in (True, False):
            handle = await client.start_workflow(
                MCPConflictAfterToolWorkflow.run,
                catch,
                id=f"test_mcp_conflict_after_a_tool_call_{catch}_{uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=20),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
            if catch:
                assert (await handle.result()).startswith(
                    "refused: Tool name 'ping' already exists. "
                    "MCP server 'late' lists it"
                )
                continue
            with pytest.raises(WorkflowFailureError) as exc_info:
                await handle.result()
            latest = await client.get_workflow_handle(handle.id).describe()
            assert latest.run_id == handle.first_execution_run_id
    cause = exc_info.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "MCPToolNameConflictError"
    assert cause.non_retryable


@workflow.defn
class HookErrorAfterToolWorkflow:
    @workflow.run
    async def run(self) -> str:
        model_calls = 0

        def fail_after_tool(
            event: BeforeModelCallEvent,  # pyright: ignore[reportUnusedParameter]
        ) -> None:
            nonlocal model_calls
            model_calls += 1
            if model_calls == 2:
                raise ValueError("hook failed")

        agent = TemporalAgent(
            model="mock", start_to_close_timeout=timedelta(seconds=30), tools=[ping]
        )
        agent.hooks.add_callback(BeforeModelCallEvent, fail_after_tool)
        try:
            await agent.invoke_async("ping")
        except EventLoopException as err:
            return f"wrapped {type(err.__cause__).__name__}"
        return "not raised"


async def test_other_errors_after_a_tool_call_stay_wrapped(client: Client):
    """Only a tool name conflict is unwrapped: Strands still wraps a hook's ValueError."""
    task_queue = f"test_other_errors_after_a_tool_call_stay_wrapped-{uuid4()}"
    plugin = StrandsPlugin(
        models={"mock": lambda: MockModel([{"name": "ping", "input": {}}])}
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[HookErrorAfterToolWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        result = await client.execute_workflow(
            HookErrorAfterToolWorkflow.run,
            id=f"test_other_errors_after_a_tool_call_stay_wrapped_{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )
    assert result == "wrapped ValueError"


def _listed_tool(name: str) -> Tool:
    return Tool.model_validate(
        {"name": name, "inputSchema": {"type": "object", "properties": {}}}
    )


def _mcp_tool(server: str, name: str) -> TemporalMCPTool:
    return TemporalMCPTool(DurableMCPClient(server), _listed_tool(name))


async def _before_model_call(agent: TemporalAgent) -> None:
    """Run the agent's hooks as Strands does before each model call."""
    await agent.hooks.invoke_callbacks_async(BeforeModelCallEvent(agent=agent))


def _listing(
    provider: TemporalMCPClient, *names: str, log: list[str] | None = None
) -> Any:
    """Stand in for the list-tools Activity; each call makes new tools, as _refresh() does."""

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
    d = TemporalMCPClient("d", cache_tools=cache_tools)
    agent = TemporalAgent(model="m", tools=[a, c, d])
    log: list[str] = []
    monkeypatch.setattr(a, "_refresh", _listing(a, "echo", log=log))
    monkeypatch.setattr(c, "_refresh", _listing(c, "echo", log=log))
    monkeypatch.setattr(d, "_refresh", _listing(d, "other", log=log))

    for _turn in range(2):
        with pytest.raises(
            ValueError,
            match="Tool name 'echo' already exists. MCP servers 'a' and 'c' both list it.",
        ):
            await _before_model_call(agent)
        assert agent.tool_registry.registry == {}
    # A refused turn lists no further, and a cached listing is checked again,
    # not listed again.
    assert log == (["a", "c"] if cache_tools else ["a", "c", "a", "c"])

    # An agent built from the same clients after they listed is refused the same way.
    late = TemporalAgent(model="m", tools=[a, c, d])
    with pytest.raises(
        MCPToolNameConflictError, match="MCP servers 'a' and 'c' both list it."
    ):
        await _before_model_call(late)


@pytest.mark.parametrize(
    ("names", "handles", "message"),
    [
        (("search", "search"), 1, "MCP server 'a' lists it twice."),
        (
            ("foo-bar", "foo_bar"),
            1,
            "Tool name 'foo_bar' already exists as 'foo-bar'. "
            "MCP server 'a' lists both names.",
        ),
        (
            ("search",),
            2,
            "The agent has two TemporalMCPClient handles for MCP server 'a'. "
            "Give one a different prefix, narrow one's tool_filters, or pass only "
            "one.",
        ),
    ],
)
async def test_mcp_server_listing_a_name_twice_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...],
    handles: int,
    message: str,
):
    """A name one server lists twice is refused too, also through two handles."""
    clients = [TemporalMCPClient("a", cache_tools=i > 0) for i in range(handles)]
    agent = TemporalAgent(model="m", tools=clients)
    for client in clients:
        monkeypatch.setattr(client, "_refresh", _listing(client, *names))

    with pytest.raises(MCPToolNameConflictError, match=re.escape(message)):
        await _before_model_call(agent)


async def test_mcp_conflict_names_the_servers_tool(monkeypatch: pytest.MonkeyPatch):
    """With a prefix, the error names the server's tool, which tool_filters match.

    Two handles of one server are refused only when both list a name.
    """

    def server(name: str, *names: str, **kwargs: Any) -> TemporalMCPClient:
        client = TemporalMCPClient(name, **kwargs)

        async def list_tools() -> ListToolsResult:
            return ListToolsResult(tools=[_listed_tool(n) for n in names])

        monkeypatch.setattr(client._client, "list_tools", list_tools)
        return client

    a = server("a", "x", prefix="p")
    with pytest.raises(
        MCPToolNameConflictError,
        match=re.escape(
            "Tool name 'p_x' already exists. "
            "MCP servers 'a' (as 'x') and 'b' (as 'x') both list it."
        ),
    ):
        await _before_model_call(
            TemporalAgent(model="m", tools=[a, server("b", "x", prefix="p")])
        )
    b = server("b", "x", prefix="p", tool_filters={"rejected": ["x"]})
    agent = TemporalAgent(model="m", tools=[a, b])
    await _before_model_call(agent)
    assert set(agent.tool_registry.registry) == {"p_x"}

    with pytest.raises(
        MCPToolNameConflictError,
        match=re.escape(
            "MCP server 'a' (as 'x') lists it, and another tool of the agent has "
            "it. Set a prefix (or a different one) on the server's "
            "TemporalMCPClient, or drop 'x' with its tool_filters."
        ),
    ):
        await _before_model_call(
            TemporalAgent(model="m", tools=[_mcp_tool("local", "p_x"), a])
        )

    names = ("search", "create_pr")
    fast = server("s", *names, tool_filters={"allowed": ["search", "create_pr"]})
    slow = server("s", *names, tool_filters={"allowed": ["create_pr"]})
    with pytest.raises(MCPToolNameConflictError, match="two TemporalMCPClient handles"):
        await _before_model_call(TemporalAgent(model="m", tools=[fast, slow]))
    fast = server("s", *names, tool_filters={"allowed": ["search"]})
    agent = TemporalAgent(model="m", tools=[fast, slow])
    await _before_model_call(agent)
    assert set(agent.tool_registry.registry) == set(names)


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

    # Built after the client listed, an agent gets the same tools.
    late = TemporalAgent(model="m", tools=[a, a])
    await _before_model_call(late)
    assert _sources(late, a) == {"echo": "a"}


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

    # An agent built after the client listed gets the tools at its first turn.
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
@pytest.mark.parametrize(("first", "second"), [("foo-bar", "foo_bar"), ("x_y", "x-y")])
async def test_mcp_refused_listing_registers_nothing(
    monkeypatch: pytest.MonkeyPatch, cache_tools: bool, first: str, second: str
):
    """A name that differs from another only by '-' or '_' refuses the whole turn."""
    h = TemporalMCPClient("h")
    n = TemporalMCPClient("n", cache_tools=cache_tools)
    agent = TemporalAgent(model="m", tools=[h, n])
    monkeypatch.setattr(h, "_refresh", _listing(h, first))
    monkeypatch.setattr(n, "_refresh", _listing(n, "safe", second))

    for _turn in range(2):
        with pytest.raises(
            ValueError,
            match=f"Tool name '{second}' already exists as '{first}'. "
            "MCP servers 'h' and 'n' both list it.",
        ):
            await _before_model_call(agent)
        assert agent.tool_registry.registry == {}

    # h drops its name; n's tools register and stay n's on the next turns.
    monkeypatch.setattr(h, "_refresh", _listing(h, "other"))
    for _turn in range(2):
        await _before_model_call(agent)
        assert _sources(agent, h, n) == {"other": "h", "safe": "n", second: "n"}


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
    """A server's tool replaced in place stays the server's: listed again, it is replaced.

    A name the server drops is the agent's again.
    """
    a = TemporalMCPClient("a")
    agent = TemporalAgent(model="m", tools=[a])
    monkeypatch.setattr(a, "_refresh", _listing(a, "echo"))

    await _before_model_call(agent)
    agent.tool_registry.replace(_local_echo)  # for example a logging wrapper
    await _before_model_call(agent)
    assert _sources(agent, a) == {"echo": "a"}

    # A dynamic replacement, such as a tool loaded from a file, is replaced too,
    # and goes with the name.
    reloaded = _mcp_tool("file", "echo")
    reloaded.mark_dynamic()
    agent.tool_registry.replace(reloaded)
    await _before_model_call(agent)
    assert _sources(agent, a) == {"echo": "a"}
    assert agent.tool_registry.dynamic_tools == {}

    agent.tool_registry.replace(reloaded)
    monkeypatch.setattr(a, "_refresh", _listing(a))
    await _before_model_call(agent)
    assert agent.tool_registry.registry == {}
    assert agent.tool_registry.dynamic_tools == {}

    # A tool put under the dropped name stays, and the server listing the name
    # again is refused.
    agent.tool_registry.register_tool(_local_echo)
    await _before_model_call(agent)
    assert agent.tool_registry.registry == {"echo": _local_echo}
    monkeypatch.setattr(a, "_refresh", _listing(a, "echo"))
    with pytest.raises(MCPToolNameConflictError, match="MCP server 'a' lists it"):
        await _before_model_call(agent)
    assert agent.tool_registry.registry == {"echo": _local_echo}


@tool(name="foo_bar")
def _local_foo_bar() -> str:
    """Return LOCAL."""
    return "LOCAL"


@tool(name="x-y")
def _local_x_y() -> str:
    """Return LOCAL."""
    return "LOCAL"


@pytest.mark.parametrize(
    ("listed", "message"),
    [
        (
            "echo",
            "Tool name 'echo' already exists. MCP server 'a' lists it, and another "
            "tool of the agent has it. Set a prefix (or a different one) on the "
            "server's TemporalMCPClient, or drop 'echo' with its tool_filters.",
        ),
        ("foo-bar", "Tool name 'foo-bar' already exists as 'foo_bar'."),
        ("x_y", "Tool name 'x_y' already exists as 'x-y'."),
        ("sandbox_shell", "Tool name 'sandbox_shell' already exists."),
    ],
)
async def test_mcp_tool_named_like_another_tool_is_refused(
    monkeypatch: pytest.MonkeyPatch, listed: str, message: str
):
    """A listed name held by another tool of the agent, or a '-'/'_' twin, is refused.

    So is it for an agent built after the client listed, whatever the tools' order.
    """
    a = TemporalMCPClient("a")
    monkeypatch.setattr(a, "_refresh", _listing(a, listed))
    local = [_local_echo, _local_foo_bar, _local_x_y]
    for tools in ([*local, a], [a, *local]):
        agent = TemporalAgent(model="m", tools=tools, sandbox=TemporalSandbox("sb"))
        with pytest.raises(MCPToolNameConflictError, match=re.escape(message)):
            await _before_model_call(agent)
        assert not any(
            isinstance(held, TemporalMCPTool)
            for held in agent.tool_registry.registry.values()
        )


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

    # Strands compares dynamic tools' names exactly, so foo-bar may join foo_bar.
    agent.tool_registry.register_dynamic_tool(_mcp_tool("x", "foo_bar"))
    monkeypatch.setattr(a, "_refresh", _listing(a, "foo-bar"))
    await _before_model_call(agent)
    assert _sources(agent, a) == {"foo-bar": "a"}


async def test_mcp_tool_added_while_listing_is_refused(
    monkeypatch: pytest.MonkeyPatch,
):
    """A tool added while a listing runs, as by a Signal handler, counts when the turn is checked."""
    a = TemporalMCPClient("a")
    b = TemporalMCPClient("b")
    agent = TemporalAgent(model="m", tools=[a, b])
    listing_b = _listing(b, "echo")

    async def b_while_signal_adds_echo() -> None:
        agent.tool_registry.register_tool(_local_echo)
        await listing_b()

    monkeypatch.setattr(a, "_refresh", _listing(a, "safe"))
    monkeypatch.setattr(b, "_refresh", b_while_signal_adds_echo)
    with pytest.raises(
        MCPToolNameConflictError, match="MCP server 'b' lists it, and another tool"
    ):
        await _before_model_call(agent)
    assert agent.tool_registry.registry == {"echo": _local_echo}

    # A tool removed while a listing runs no longer holds its name.
    async def b_while_echo_is_removed() -> None:
        agent.tool_registry.registry.pop("echo")
        await listing_b()

    monkeypatch.setattr(b, "_refresh", b_while_echo_is_removed)
    await _before_model_call(agent)
    assert _sources(agent, a, b) == {"safe": "a", "echo": "b"}


async def test_mcp_dynamic_tool_added_while_listing_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dynamic tool added while a listing runs counts when the turn is checked."""
    a, b = TemporalMCPClient("a"), TemporalMCPClient("b")
    agent = TemporalAgent(model="m", tools=[a, b])
    listing_b = _listing(b, "echo")

    async def b_while_signal_adds_dynamic_echo() -> None:
        agent.tool_registry.register_dynamic_tool(_local_echo)
        await listing_b()

    monkeypatch.setattr(a, "_refresh", _listing(a, "safe"))
    monkeypatch.setattr(b, "_refresh", b_while_signal_adds_dynamic_echo)
    with pytest.raises(MCPToolNameConflictError, match="MCP server 'b' lists it"):
        await _before_model_call(agent)
    assert agent.tool_registry.registry == {}
    assert agent.tool_registry.dynamic_tools == {"echo": _local_echo}


async def test_mcp_listing_is_read_when_it_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later listing of a shared client by another agent does not change this turn's."""
    p1, p2 = TemporalMCPClient("p1"), TemporalMCPClient("p2")
    first = TemporalAgent(model="m", tools=[p1, p2])
    second = TemporalAgent(model="m", tools=[p1])
    monkeypatch.setattr(p1, "_refresh", _listing(p1, "x"))
    listing_p2 = _listing(p2, "y")

    async def p2_while_second_lists_p1() -> None:
        monkeypatch.setattr(p1, "_refresh", _listing(p1, "y"))
        await _before_model_call(second)
        await listing_p2()

    monkeypatch.setattr(p2, "_refresh", p2_while_second_lists_p1)
    await _before_model_call(first)
    assert set(first.tool_registry.registry) == {"x", "y"}
    assert first.tool_registry.registry["y"] is p2._tools[0]
