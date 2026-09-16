from collections.abc import Callable
from datetime import timedelta
from typing import Any, cast
from uuid import uuid4

import pytest
from agents import Agent, AgentBase, RunContextWrapper, Runner
from agents.mcp import MCPServer as AgentsMCPServer
from agents.mcp import MCPServerStdio
from mcp import Client as MCPClient
from mcp.server.mcpserver import MCPServer as SDKMCPServer
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    Prompt,
    Resource,
    ResourceTemplate,
    Tool,
)

import temporalio.openai_agents as openai_agents
from temporalio import workflow
from temporalio.client import Client
from temporalio.exceptions import ApplicationError
from temporalio.mcp import TemporalMCPClient
from temporalio.openai_agents import ModelActivityParameters
from temporalio.openai_agents._mcp_backend import (
    _mcp_server_backend_factory,
    _OpenAIMCPServerBackend,
)
from temporalio.openai_agents._temporal_mcp_server import (
    _TemporalMCPServer,
)
from temporalio.openai_agents.testing import (
    AgentEnvironment,
    ResponseBuilders,
    TestModel,
)
from tests.helpers import new_worker


@workflow.defn
class MCPWorkflow:
    @workflow.run
    async def run(self, cache_tools_list: bool) -> str:
        server = openai_agents.workflow.temporal_mcp_server(
            "hello", cache_tools_list=cache_tools_list
        )
        agent = Agent[str](
            name="MCP workflow",
            instructions="Use the tools.",
            mcp_servers=[server],
        )
        result = await Runner.run(agent, "Say hello to Tom and Tim.")
        return result.final_output


@workflow.defn
class MCPFactoryArgumentWorkflow:
    @workflow.run
    async def run(self) -> str:
        server = openai_agents.workflow.temporal_mcp_server(
            "hello", factory_argument={"tenant": "acme"}
        )
        agent = Agent[str](
            name="MCP workflow",
            instructions="Use the tools.",
            mcp_servers=[server],
        )
        result = await Runner.run(agent, "Say hello to Tom and Tim.")
        return result.final_output


def model() -> TestModel:
    return TestModel.returning_responses(
        [
            ResponseBuilders.tool_call(
                '{"name":"Tom"}', "say_hello", call_id="call-1", item_id="item-1"
            ),
            ResponseBuilders.tool_call(
                '{"name":"Tim"}', "say_hello", call_id="call-2", item_id="item-2"
            ),
            ResponseBuilders.output_message("Hi Tom and Tim!"),
        ]
    )


def server() -> SDKMCPServer[Any]:
    value = SDKMCPServer("hello")

    @value.tool()
    def say_hello(name: str) -> str:  # type: ignore[reportUnusedFunction]
        return f"Hello {name}"

    return value


class InProcessMCPServer(AgentsMCPServer):
    def __init__(self, server: SDKMCPServer[Any]) -> None:
        super().__init__()
        self._client = MCPClient(server)
        self.session: MCPClient | None = None

    @property
    def name(self) -> str:
        return "hello"

    async def connect(self) -> None:
        await self._client.__aenter__()
        self.session = self._client

    async def cleanup(self) -> None:
        await self._client.__aexit__(None, None, None)

    async def list_tools(
        self,
        run_context: RunContextWrapper[Any] | None = None,
        agent: AgentBase | None = None,
    ) -> list[Tool]:
        return (await self._client.list_tools()).tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        return await self._client.call_tool(tool_name, arguments, meta=cast(Any, meta))

    async def list_prompts(self) -> ListPromptsResult:
        return await self._client.list_prompts()

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        return await self._client.get_prompt(name, arguments)


@pytest.mark.parametrize(
    ("cache_tools_list", "expected_list_activities"), ((True, 1), (False, 3))
)
async def test_openai_agents_uses_mcp_v2_and_caches_connection(
    client: Client, cache_tools_list: bool, expected_list_activities: int
) -> None:
    created = 0
    mcp_server = server()

    def factory() -> AgentsMCPServer:
        nonlocal created
        created += 1
        return InProcessMCPServer(mcp_server)

    async with AgentEnvironment(
        model=model(),
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30)
        ),
        mcp_servers={"hello": factory},
    ) as environment:
        plugged_client = environment.applied_on_client(client)
        async with new_worker(plugged_client, MCPWorkflow) as worker:
            handle = await plugged_client.start_workflow(
                MCPWorkflow.run,
                cache_tools_list,
                id=f"mcp-v2-{uuid4()}",
                task_queue=worker.task_queue,
            )
            result = await handle.result()
            history = await handle.fetch_history()

    assert result == "Hi Tom and Tim!"
    assert created == 1
    assert (
        sum(
            event.activity_task_scheduled_event_attributes.activity_type.name
            == "temporalio.mcp.hello.list-tools"
            for event in history.events
            if event.HasField("activity_task_scheduled_event_attributes")
        )
        == expected_list_activities
    )


async def test_factory_argument_uses_fresh_client_per_activity(
    client: Client,
) -> None:
    arguments: list[Any] = []
    mcp_server = server()

    def factory(argument: Any) -> AgentsMCPServer:
        arguments.append(argument)
        return InProcessMCPServer(mcp_server)

    async with AgentEnvironment(
        model=model(),
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=30)
        ),
        mcp_servers={"hello": factory},
    ) as environment:
        plugged_client = environment.applied_on_client(client)
        async with new_worker(plugged_client, MCPFactoryArgumentWorkflow) as worker:
            result = await plugged_client.execute_workflow(
                MCPFactoryArgumentWorkflow.run,
                id=f"mcp-v2-argument-{uuid4()}",
                task_queue=worker.task_queue,
            )

    assert result == "Hi Tom and Tim!"
    assert arguments == [{"tenant": "acme"}] * 3


def test_callable_worker_side_tool_filter_is_rejected() -> None:
    def factory() -> AgentsMCPServer:
        value = InProcessMCPServer(server())
        value.tool_filter = lambda context, tool: True  # type: ignore[attr-defined]
        return value

    with pytest.raises(ApplicationError) as err:
        cast(Callable[[], Any], _mcp_server_backend_factory("hello", factory))()
    assert err.value.non_retryable
    assert "temporal_mcp_server()" in str(err.value)


def test_static_worker_side_tool_filter_is_allowed() -> None:
    def factory() -> AgentsMCPServer:
        value = InProcessMCPServer(server())
        value.tool_filter = {"blocked_tool_names": ["say_hello"]}  # type: ignore[attr-defined]
        return value

    backend = cast(
        Callable[[], _OpenAIMCPServerBackend],
        _mcp_server_backend_factory("hello", factory),
    )()
    assert isinstance(backend, _OpenAIMCPServerBackend)
    assert backend._server.tool_filter == {  # type: ignore[attr-defined]
        "blocked_tool_names": ["say_hello"]
    }


def test_cached_tools_applies_static_filter() -> None:
    class CachedClient:
        name = "hello"
        cached_tools = ListToolsResult(
            tools=[
                Tool(name="allowed", input_schema={}),
                Tool(name="blocked", input_schema={}),
            ]
        )

    client = cast(TemporalMCPClient, cast(object, CachedClient()))
    value = _TemporalMCPServer(client, tool_filter={"blocked_tool_names": ["blocked"]})

    cached_tools = value.cached_tools
    assert cached_tools is not None
    assert [tool.name for tool in cached_tools] == ["allowed"]


def test_cached_tools_is_unavailable_for_dynamic_filter() -> None:
    class CachedClient:
        name = "hello"
        cached_tools = ListToolsResult(
            tools=[Tool(name="context-dependent", input_schema={})]
        )

    client = cast(TemporalMCPClient, cast(object, CachedClient()))
    value = _TemporalMCPServer(client, tool_filter=lambda context, tool: True)

    assert value.cached_tools is None


async def test_repeated_pagination_cursor_is_non_retryable() -> None:
    class RepeatingCursorServer:
        async def list_resources(
            self, _cursor: str | None = None
        ) -> ListResourcesResult:
            return ListResourcesResult(resources=[], next_cursor="repeated")

    backend = _OpenAIMCPServerBackend(
        cast(AgentsMCPServer, cast(object, RepeatingCursorServer()))
    )
    with pytest.raises(ApplicationError) as err:
        await backend.list_resources()

    assert err.value.type == "MCPProtocolError"
    assert err.value.non_retryable


@pytest.mark.parametrize("method", ["list_tools", "list_prompts"])
async def test_agents_repeated_pagination_cursor_is_non_retryable(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = MCPServerStdio(params={"command": "unused"}, name="repeating")
    server.session = cast(Any, object())

    async def list_tools_page(
        _session: object, _cursor: str | None = None
    ) -> ListToolsResult:
        return ListToolsResult(tools=[], next_cursor="repeated")

    async def list_prompts_page(
        _session: object, _cursor: str | None = None
    ) -> ListPromptsResult:
        return ListPromptsResult(prompts=[], next_cursor="repeated")

    monkeypatch.setattr(server, "_list_tools_page", list_tools_page)
    monkeypatch.setattr(server, "_list_prompts_page", list_prompts_page)
    backend = _OpenAIMCPServerBackend(server)
    with pytest.raises(ApplicationError) as err:
        await getattr(backend, method)()

    assert err.value.type == "MCPProtocolError"
    assert err.value.non_retryable


async def test_paginated_results_allow_missing_later_ttl() -> None:
    class MissingLaterTTLServer:
        async def list_resources(
            self, cursor: str | None = None
        ) -> ListResourcesResult:
            if cursor is None:
                return ListResourcesResult.model_validate(
                    {
                        "resources": [Resource(name="one", uri="file:///one")],
                        "nextCursor": "next",
                        "ttlMs": 300,
                    }
                )
            return ListResourcesResult.model_validate(
                {"resources": [Resource(name="two", uri="file:///two")]}
            )

    backend = _OpenAIMCPServerBackend(
        cast(AgentsMCPServer, cast(object, MissingLaterTTLServer()))
    )

    resources = await backend.list_resources()

    assert [str(resource.uri) for resource in resources.resources] == [
        "file:///one",
        "file:///two",
    ]
    assert resources.ttl_ms == 0


async def test_list_results_preserve_mcp_envelopes() -> None:
    class MetadataServer:
        async def list_prompts(self) -> ListPromptsResult:
            return ListPromptsResult.model_validate(
                {
                    "prompts": [Prompt(name="prompt")],
                    "_meta": {"prompt": "metadata"},
                    "ttlMs": 300,
                    "cacheScope": "private",
                }
            )

        async def list_resources(
            self, cursor: str | None = None
        ) -> ListResourcesResult:
            if cursor is None:
                return ListResourcesResult.model_validate(
                    {
                        "resources": [Resource(name="one", uri="file:///one")],
                        "nextCursor": "next",
                        "_meta": {"first": 1, "shared": "first"},
                        "ttlMs": 300,
                    }
                )
            return ListResourcesResult.model_validate(
                {
                    "resources": [Resource(name="two", uri="file:///two")],
                    "_meta": {"second": 2, "shared": "second"},
                    "ttlMs": 200,
                    "cacheScope": "private",
                }
            )

        async def list_resource_templates(
            self, cursor: str | None = None
        ) -> ListResourceTemplatesResult:
            del cursor
            return ListResourceTemplatesResult.model_validate(
                {
                    "resourceTemplates": [
                        ResourceTemplate(name="template", uri_template="file:///{name}")
                    ],
                    "_meta": {"template": "metadata"},
                    "ttlMs": 100,
                    "cacheScope": "private",
                }
            )

    backend = _OpenAIMCPServerBackend(
        cast(AgentsMCPServer, cast(object, MetadataServer()))
    )

    prompts = await backend.list_prompts()
    assert prompts.meta == {"prompt": "metadata"}
    assert prompts.ttl_ms == 300
    assert prompts.cache_scope == "private"

    resources = await backend.list_resources()
    assert [str(resource.uri) for resource in resources.resources] == [
        "file:///one",
        "file:///two",
    ]
    assert resources.next_cursor is None
    assert resources.meta == {"first": 1, "shared": "second", "second": 2}
    assert resources.ttl_ms == 200
    assert resources.cache_scope == "private"

    templates = await backend.list_resource_templates()
    assert templates.meta == {"template": "metadata"}
    assert templates.ttl_ms == 100
    assert templates.cache_scope == "private"
