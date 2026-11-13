import asyncio
import importlib.metadata
import socket
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

# Imported outside imports_passed_through() on purpose: MCPPlugin must pass
# mcp_types through the sandbox, or this becomes a sandbox-local copy of the
# class and the isinstance check in NativeMCPWorkflow fails.
import mcp_types
import pytest

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.mcp import TemporalMCPClient

with workflow.unsafe.imports_passed_through():
    import uvicorn
    from mcp import Client, MCPError, StdioServerParameters, stdio_client
    from mcp.server.lowlevel import Server
    from mcp.server.mcpserver import MCPServer
    from mcp.types import (
        INTERNAL_ERROR,
        CallToolRequestParams,
        CallToolResult,
        ListToolsResult,
        TextContent,
        TextResourceContents,
        Tool,
    )

    from temporalio.client import Client as TemporalClient
    from temporalio.client import WorkflowFailureError
    from temporalio.exceptions import ActivityError, ApplicationError, RetryState
    from temporalio.mcp import MCPPlugin
    from temporalio.worker import Replayer
    from tests.helpers import new_worker


def rich_server() -> MCPServer[Any]:
    server = MCPServer("rich")

    @server.tool()
    def echo(value: str) -> str:  # type: ignore[reportUnusedFunction]
        return value

    @server.prompt()
    def greeting(name: str) -> str:  # type: ignore[reportUnusedFunction]
        return f"Hello {name}"

    @server.resource("test://static", name="static")
    def static_resource() -> str:  # type: ignore[reportUnusedFunction]
        return "resource contents"

    @server.resource("test://items/{item}", name="item")
    def item_resource(item: str) -> str:  # type: ignore[reportUnusedFunction]
        return item

    return server


@workflow.defn
class NativeMCPWorkflow:
    @workflow.run
    async def run(self) -> tuple[str, str, str, str, str, str, str]:
        client = TemporalMCPClient("rich")
        tools = await client.list_tools()
        # The second call is served from replay-safe workflow state.
        assert await client.list_tools() is tools
        tool_result = await client.call_tool("echo", {"value": "hello"})
        assert isinstance(tool_result.content[0], mcp_types.TextContent)
        prompts = await client.list_prompts()
        prompt = await client.get_prompt("greeting", {"name": "Temporal"})
        resources = await client.list_resources()
        templates = await client.list_resource_templates()
        resource = await client.read_resource("test://static")
        return (
            tools.tools[0].name,
            tool_result.content[0].text,
            prompts.prompts[0].name,
            cast(TextContent, prompt.messages[0].content).text,
            resources.resources[0].name,
            templates.resource_templates[0].name,
            cast(TextResourceContents, resource.contents[0]).text,
        )


@workflow.defn
class ConfiguredMCPWorkflow:
    @workflow.run
    async def run(self, mode: str) -> str:
        if mode == "retry-policy-only":
            config: workflow.ActivityConfig = {
                "retry_policy": RetryPolicy(maximum_attempts=1)
            }
        elif mode == "explicit-none":
            config = {
                "start_to_close_timeout": None,
                "schedule_to_close_timeout": None,
                "retry_policy": RetryPolicy(maximum_attempts=1),
            }
        elif mode == "heartbeat-off":
            config = {"heartbeat_timeout": None}
        else:
            config = {"schedule_to_close_timeout": timedelta(hours=1)}
        client = TemporalMCPClient("rich", activity_config=config)
        result = await client.call_tool("echo", {"value": mode})
        return cast(TextContent, result.content[0]).text


@workflow.defn
class TransportMCPWorkflow:
    @workflow.run
    async def run(self, server: str) -> str:
        client = TemporalMCPClient(server)
        tools = await client.list_tools()
        assert [tool.name for tool in tools.tools] == ["echo"]
        result = await client.call_tool("echo", {"value": "transport"})
        return cast(TextContent, result.content[0]).text


@workflow.defn
class RetriedToolWorkflow:
    @workflow.run
    async def run(self, tool: str, factory_argument: str | None) -> str:
        client = TemporalMCPClient(
            "retries",
            activity_config={
                "retry_policy": RetryPolicy(
                    initial_interval=timedelta(milliseconds=1), maximum_attempts=2
                )
            },
            factory_argument=factory_argument,
        )
        result = await client.call_tool(tool, {})
        return cast(TextContent, result.content[0]).text


async def activity_names(handle: Any) -> list[str]:
    names: list[str] = []
    async for event in handle.fetch_history_events():
        if event.HasField("activity_task_scheduled_event_attributes"):
            names.append(
                event.activity_task_scheduled_event_attributes.activity_type.name
            )
    return names


async def test_native_workflow_operations_and_replay(client: TemporalClient) -> None:
    server = rich_server()
    plugin = MCPPlugin({"rich": lambda: Client(server)})
    async with new_worker(client, NativeMCPWorkflow, plugins=[plugin]) as worker:
        handle = await client.start_workflow(
            NativeMCPWorkflow.run,
            id=f"native-mcp-{uuid4()}",
            task_queue=worker.task_queue,
        )
        assert await handle.result() == (
            "echo",
            "hello",
            "greeting",
            "Hello Temporal",
            "static",
            "item",
            "resource contents",
        )
        names = await activity_names(handle)
        history = await handle.fetch_history()

    assert names == [
        "temporalio.mcp.rich.list-tools",
        "temporalio.mcp.rich.call-tool",
        "temporalio.mcp.rich.list-prompts",
        "temporalio.mcp.rich.get-prompt",
        "temporalio.mcp.rich.list-resources",
        "temporalio.mcp.rich.list-resource-templates",
        "temporalio.mcp.rich.read-resource",
    ]
    await Replayer(workflows=[NativeMCPWorkflow], plugins=[plugin]).replay_workflow(
        history
    )


async def test_activity_config_keeps_default_timeout_only_when_needed(
    client: TemporalClient,
) -> None:
    server = rich_server()
    plugin = MCPPlugin({"rich": lambda: Client(server)})
    async with new_worker(client, ConfiguredMCPWorkflow, plugins=[plugin]) as worker:
        for mode in (
            "retry-policy-only",
            "explicit-none",
            "heartbeat-off",
            "schedule-to-close-only",
        ):
            handle = await client.start_workflow(
                ConfiguredMCPWorkflow.run,
                mode,
                id=f"configured-mcp-{mode}-{uuid4()}",
                task_queue=worker.task_queue,
            )
            assert await handle.result() == mode
            [(start_to_close, schedule_to_close, heartbeat)] = await activity_timeouts(
                handle
            )
            # The 10-second heartbeat default is added unless the config sets one,
            # even to None.
            assert heartbeat == (
                None if mode == "heartbeat-off" else timedelta(seconds=10)
            )
            if mode in {"retry-policy-only", "explicit-none", "heartbeat-off"}:
                # The default start-to-close timeout was merged in.
                assert (start_to_close, schedule_to_close) == (
                    timedelta(minutes=1),
                    None,
                )
            else:
                # A caller-supplied schedule-to-close timeout is left alone: the
                # one-minute default is not added (the server fills start-to-close
                # from schedule-to-close when the command leaves it unset).
                assert schedule_to_close == timedelta(hours=1)
                assert start_to_close != timedelta(minutes=1)


# A retry can't fix a result the client rejects, so it gets one attempt. A
# server error gets another.
@pytest.mark.parametrize("factory_argument", [None, "acme"])
async def test_only_errors_a_retry_can_fix_are_retried(
    client: TemporalClient, factory_argument: str | None
) -> None:
    calls: list[str] = []

    async def list_tools(_ctx: Any, _params: Any) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                Tool(
                    name="rejected",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                ),
                Tool(name="flaky", input_schema={"type": "object"}),
            ]
        )

    async def call_tool(_ctx: Any, params: CallToolRequestParams) -> CallToolResult:
        calls.append(params.name)
        if calls == ["rejected", "flaky"]:
            raise MCPError(INTERNAL_ERROR, "try again")
        # No structured content, so the client rejects "rejected"'s result.
        return CallToolResult(content=[TextContent(text=params.name)])

    server = Server("retries", on_list_tools=list_tools, on_call_tool=call_tool)
    plugin = MCPPlugin({"retries": lambda _argument: Client(server)})
    async with new_worker(client, RetriedToolWorkflow, plugins=[plugin]) as worker:
        with pytest.raises(WorkflowFailureError) as err:
            await client.execute_workflow(
                RetriedToolWorkflow.run,
                args=["rejected", factory_argument],
                id=f"mcp-rejected-{uuid4()}",
                task_queue=worker.task_queue,
            )
        assert (
            await client.execute_workflow(
                RetriedToolWorkflow.run,
                args=["flaky", factory_argument],
                id=f"mcp-flaky-{uuid4()}",
                task_queue=worker.task_queue,
            )
            == "flaky"
        )

    assert calls == ["rejected", "flaky", "flaky"]
    activity_error = err.value.cause
    assert isinstance(activity_error, ActivityError)
    assert activity_error.retry_state == RetryState.NON_RETRYABLE_FAILURE
    assert isinstance(activity_error.cause, ApplicationError)
    assert activity_error.cause.type == "MCPProtocolError"


async def activity_timeouts(
    handle: Any,
) -> list[tuple[timedelta | None, timedelta | None, timedelta | None]]:
    timeouts: list[tuple[timedelta | None, timedelta | None, timedelta | None]] = []
    async for event in handle.fetch_history_events():
        if event.HasField("activity_task_scheduled_event_attributes"):
            attrs = event.activity_task_scheduled_event_attributes
            start_to_close = attrs.start_to_close_timeout.ToTimedelta()
            schedule_to_close = attrs.schedule_to_close_timeout.ToTimedelta()
            heartbeat = attrs.heartbeat_timeout.ToTimedelta()
            timeouts.append(
                (
                    start_to_close or None,
                    schedule_to_close or None,
                    heartbeat or None,
                )
            )
    return timeouts


async def run_transport_workflow(
    client: TemporalClient,
    name: str,
    factory: Callable[[], Client],
) -> None:
    async with new_worker(
        client,
        TransportMCPWorkflow,
        plugins=[MCPPlugin({name: factory}, connection_idle_timeout=None)],
    ) as worker:
        result = await client.execute_workflow(
            TransportMCPWorkflow.run,
            name,
            id=f"mcp-{name}-{uuid4()}",
            task_queue=worker.task_queue,
        )
    assert result == "transport"


async def test_stdio_transport(client: TemporalClient) -> None:
    server_path = Path(__file__).parent / "echo_mcp_server.py"
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )
    await run_transport_workflow(
        client,
        "stdio",
        lambda: Client(stdio_client(parameters)),
    )


@asynccontextmanager
async def streamable_http_server() -> AsyncGenerator[str, None]:
    server = MCPServer("http-echo")

    @server.tool()
    def echo(value: str) -> str:  # type: ignore[reportUnusedFunction]
        return value

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    app = server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        host="127.0.0.1",
    )
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    task = asyncio.create_task(uvicorn_server.serve(sockets=[sock]))
    try:
        for _ in range(500):
            if uvicorn_server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        else:
            raise RuntimeError("Streamable HTTP MCP server did not start")
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        uvicorn_server.should_exit = True
        try:
            await asyncio.gather(task, return_exceptions=True)
        finally:
            sock.close()


async def test_streamable_http_transport(client: TemporalClient) -> None:
    async with streamable_http_server() as url:
        await run_transport_workflow(client, "http", lambda: Client(url))


def test_plugin_reports_package_version() -> None:
    assert MCPPlugin({}).version() == importlib.metadata.version("temporalio-mcp")
