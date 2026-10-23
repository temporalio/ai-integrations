import asyncio
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp.mcp_client import MCPClient
from strands.types.exceptions import MCPClientInitializationError

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.strands_agents import (
    StrandsPlugin,
    TemporalAgent,
    TemporalMCPClient,
    _temporal_mcp_client,
)
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


def _slow_echo_client_factory(delay: float, startup_timeout: int) -> MCPClient:
    return MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=[str(Path(__file__).parent / "echo_mcp_server.py")],
                env={"ECHO_SERVER_START_DELAY": str(delay)},
            )
        ),
        startup_timeout=startup_timeout,
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
class MCPSlowStartWorkflow:
    def __init__(self) -> None:
        echo = TemporalMCPClient(
            server="echo_slow",
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=1),
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


async def test_mcp_startup_timeout(client: Client):
    """A server slower than the client's startup_timeout fails to connect, as natively."""
    task_queue = f"test_mcp_startup_timeout-{uuid4()}"
    plugin = StrandsPlugin(
        models={"mock": lambda: MockModel(["Done!"])},
        mcp_clients={"echo_slow": lambda: _slow_echo_client_factory(3, 1)},
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[MCPSlowStartWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            MCPSlowStartWorkflow.run,
            "echo hello",
            id=f"test_mcp_startup_timeout_{uuid4()}",
            task_queue=task_queue,
        )
        try:
            await handle.result()
            raise AssertionError("the run completed")
        except WorkflowFailureError as err:
            assert isinstance(err.cause, ActivityError)
            cause = err.cause.cause
            assert isinstance(cause, ApplicationError)
            assert cause.type == "MCPClientInitializationError"
            assert "startup_timeout (1 seconds)" in cause.message


async def _cancel_first_caller(
    server: str,
    list_tools: Callable[[], Awaitable[list[_temporal_mcp_client._MCPToolInfo]]],
) -> _temporal_mcp_client._ConnectionRecord:
    """Start a list-tools call, cancel it during the handshake, return its record."""
    first = asyncio.ensure_future(list_tools())
    while server not in _temporal_mcp_client._CONNECTIONS:
        await asyncio.sleep(0.01)
    record = _temporal_mcp_client._CONNECTIONS[server]
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    return record


async def test_mcp_startup_deadline_survives_cancelled_caller():
    """A cancelled caller neither stops the startup deadline nor fails other callers."""
    connections = _temporal_mcp_client._CONNECTIONS

    # Initialization outlasts the deadline: the next caller gets the deadline's
    # error, and the owner is cancelled and retired.
    slow = _temporal_mcp_client.build_list_tools_activity(
        "echo_cancelled_slow", lambda: _slow_echo_client_factory(5, 1)
    )
    record = await _cancel_first_caller("echo_cancelled_slow", slow)
    with pytest.raises(MCPClientInitializationError) as raised:
        await slow()
    assert isinstance(raised.value.__cause__, TimeoutError)
    assert "echo_cancelled_slow" not in connections
    await asyncio.wait([record._owner], timeout=10)
    assert record._owner.done()

    # The only caller is cancelled, so it arms the record's idle timer before
    # the deadline retires the record. When that timer fires, it must leave the
    # replacement, which answers at once and is busy with a call, in place.
    starts = iter([(5, 1)])
    idle = timedelta(seconds=3)

    def first_slow_factory() -> MCPClient:
        return _slow_echo_client_factory(*next(starts, (0, 30)))

    first_slow = _temporal_mcp_client.build_list_tools_activity(
        "echo_cancelled_idle", first_slow_factory, idle
    )
    record = await _cancel_first_caller("echo_cancelled_idle", first_slow)
    assert record._idle_handle is not None
    while connections.get("echo_cancelled_idle") is record:
        await asyncio.sleep(0.01)
    _, _, replacement = await _temporal_mcp_client.get_connection(
        "echo_cancelled_idle", first_slow_factory, idle
    )
    try:
        while record._idle_task is None:
            await asyncio.sleep(0.01)
        await record._idle_task
        assert connections["echo_cancelled_idle"] is replacement
        assert not replacement._owner.done()
        assert [tool.name for tool in await first_slow()] == ["echo"]
    finally:
        replacement.release()
        await _temporal_mcp_client._evict_connection("echo_cancelled_idle")

    # Initialization finishes: the cancelled caller does not discard the
    # connection, which serves the next callers.
    fast = _temporal_mcp_client.build_list_tools_activity(
        "echo_cancelled_fast", _echo_client_factory
    )
    record = await _cancel_first_caller("echo_cancelled_fast", fast)
    try:
        assert [tool.name for tool in await fast()] == ["echo"]
        assert [tool.name for tool in await fast()] == ["echo"]
        assert connections["echo_cancelled_fast"] is record
    finally:
        await _temporal_mcp_client._evict_connection("echo_cancelled_fast")


async def test_mcp_worker_shutdown_during_slow_startup(client: Client):
    """Worker shutdown waits for the startup deadline, not the server's late reply."""
    task_queue = f"test_mcp_worker_shutdown_during_slow_startup-{uuid4()}"
    plugin = StrandsPlugin(
        models={"mock": lambda: MockModel(["Done!"])},
        mcp_clients={"echo_slow": lambda: _slow_echo_client_factory(15, 2)},
    )

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[MCPSlowStartWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    )
    # worker.run() returns only after the plugin's cleanup has closed its MCP
    # connections; `async with Worker` would cancel that cleanup instead.
    run = asyncio.ensure_future(worker.run())
    handle = await client.start_workflow(
        MCPSlowStartWorkflow.run,
        "echo hello",
        id=f"test_mcp_worker_shutdown_during_slow_startup_{uuid4()}",
        task_queue=task_queue,
    )
    while "echo_slow" not in _temporal_mcp_client._CONNECTIONS:
        await asyncio.sleep(0.01)
    record = _temporal_mcp_client._CONNECTIONS["echo_slow"]
    started = time.monotonic()
    await worker.shutdown()
    # Shutdown cancels the waiting list-tools activity; the cleanup then waits
    # for the owner, which the 2-second deadline cancels, not for the server's
    # reply after 15 seconds.
    await asyncio.wait_for(run, timeout=12)
    assert time.monotonic() - started < 10
    assert record._owner.done()
    assert "echo_slow" not in _temporal_mcp_client._CONNECTIONS
    await handle.terminate()
