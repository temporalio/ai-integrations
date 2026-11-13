"""MCP tools through ``TemporalMCPToolProvider`` and ``temporalio-mcp``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest
from agent_framework import Agent, tool
from mcp import Client as MCPClient

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
from temporalio.microsoft_agent_framework import (
    MicrosoftAgentFrameworkPlugin,
    TemporalChatClient,
    TemporalMCPToolProvider,
)
from temporalio.microsoft_agent_framework.testing import (
    fake_chat_client,
    function_call_response,
    text_response,
)
from tests.helpers.microsoft_agent_framework import (
    INVOKE_MODEL,
    contents_summary,
    count_scheduled_activities,
    new_worker,
    started_activities,
    workflow_id,
)

# The server belongs to the worker side of these tests. This module also holds
# the workflow, so the sandbox re-imports it; constructing MCPServer imports
# rich, which reads the random module at import.
with workflow.unsafe.imports_passed_through():
    from tests.echo_mcp_server import server

LIST_TOOLS = "temporalio.mcp.echo.list-tools"
CALL_TOOL = "temporalio.mcp.echo.call-tool"


@dataclass
class MCPInput:
    prompt: str = "echo ab twice"
    cache_tools: bool = False
    prefix: str | None = None
    allowed: list[str] | None = None
    approval: bool = False
    runs: int = 1
    conflict: str | None = None
    max_attempts: int | None = None


@workflow.defn
class MCPWorkflow:
    @workflow.run
    async def run(self, input: MCPInput) -> dict[str, Any]:
        retry_policy = (
            RetryPolicy(
                maximum_attempts=input.max_attempts,
                initial_interval=timedelta(milliseconds=10),
            )
            if input.max_attempts
            else None
        )
        providers = [
            TemporalMCPToolProvider(
                "echo",
                cache_tools=input.cache_tools,
                tool_name_prefix=input.prefix,
                allowed_tools=input.allowed,
                approval_mode="always_require" if input.approval else None,
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=retry_policy,
            )
        ]
        tools: list[Any] = []
        if input.conflict == "two_providers":
            providers.append(
                TemporalMCPToolProvider(
                    "echo", start_to_close_timeout=timedelta(seconds=10)
                )
            )
        if input.conflict == "agent_tool":

            @tool
            async def echo(value: str) -> str:
                """A local tool with an MCP tool's name."""
                return value

            tools = [echo]
        agent = Agent(
            client=TemporalChatClient("model"), tools=tools, context_providers=providers
        )
        session = agent.create_session()
        texts: list[str] = []
        contents: list[list[str]] = []
        for _ in range(input.runs):
            response = await agent.run(input.prompt, session=session)
            if input.approval and response.user_input_requests:
                response = await agent.run(
                    [
                        c.to_function_approval_response(True)
                        for c in response.user_input_requests
                    ],
                    session=session,
                )
            texts.append(response.text)
            contents.append(contents_summary(response))
        return {"texts": texts, "contents": contents}


def _plugin(
    script: list[Any], **kwargs: Any
) -> tuple[MicrosoftAgentFrameworkPlugin, Any, list[int]]:
    connections: list[int] = []

    def factory() -> MCPClient:
        connections.append(len(connections) + 1)
        return MCPClient(server)

    chat = fake_chat_client(script)
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={"model": chat}, mcp_clients={"echo": factory}, **kwargs
    )
    return plugin, chat, connections


async def _run(
    client: Client, plugin: MicrosoftAgentFrameworkPlugin, input: MCPInput
) -> tuple[Any, Any]:
    async with new_worker(client, MCPWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            MCPWorkflow.run,
            input,
            id=workflow_id("mcp"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    return result, handle


_ECHO_SCRIPT = [
    function_call_response("echo", {"value": "ab", "times": 2}, call_id="call-1"),
    text_response("Echoed."),
]


async def test_mcp_list_and_call(client: Client) -> None:
    plugin, chat, _ = _plugin(_ECHO_SCRIPT)
    result, handle = await _run(client, plugin, MCPInput())
    assert result["texts"] == ["Echoed."]
    assert result["contents"][0] == [
        "call:echo",
        'result:{"result": "abab"}',
        "text:Echoed.",
    ]
    assert await count_scheduled_activities(handle) == {
        INVOKE_MODEL: 2,
        LIST_TOOLS: 1,
        CALL_TOOL: 1,
    }
    # The model received the server's full input schema, not just the names.
    tools = {t.name: t for t in chat().calls[0]["options"]["tools"]}
    assert set(tools) == {"echo", "shout", "fail", "wait"}
    echo = tools["echo"].parameters()
    assert echo["required"] == ["value"]
    assert echo["properties"]["times"] == {
        "default": 1,
        "title": "Times",
        "type": "integer",
    }
    assert tools["echo"].description == "Return the input repeated."


@pytest.mark.parametrize(("cache_tools", "listings"), [(False, 2), (True, 1)])
async def test_cache_tools(client: Client, cache_tools: bool, listings: int) -> None:
    plugin, _, _ = _plugin([text_response("Nothing to do.")])
    result, handle = await _run(
        client, plugin, MCPInput(runs=2, cache_tools=cache_tools)
    )
    assert result["texts"] == ["Nothing to do.", "Nothing to do."]
    assert await count_scheduled_activities(handle) == {
        INVOKE_MODEL: 2,
        LIST_TOOLS: listings,
    }


async def test_prefix_and_allowed_tools(client: Client) -> None:
    plugin, chat, _ = _plugin(
        [
            function_call_response("srv_echo", {"value": "ab"}, call_id="call-1"),
            text_response("Echoed."),
        ]
    )
    result, handle = await _run(
        client, plugin, MCPInput(prefix="srv", allowed=["echo"])
    )
    assert result["contents"][0] == [
        "call:srv_echo",
        'result:{"result": "ab"}',
        "text:Echoed.",
    ]
    assert [t.name for t in chat().calls[0]["options"]["tools"]] == ["srv_echo"]
    assert await count_scheduled_activities(handle) == {
        INVOKE_MODEL: 2,
        LIST_TOOLS: 1,
        CALL_TOOL: 1,
    }


async def test_mcp_tool_approval(client: Client) -> None:
    plugin, _, _ = _plugin(_ECHO_SCRIPT)
    result, handle = await _run(client, plugin, MCPInput(approval=True))
    assert result["texts"] == ["Echoed."]
    assert 'result:{"result": "abab"}' in result["contents"][0]
    # The approval reply is a second agent.run, and cache_tools=False lists per run.
    assert await count_scheduled_activities(handle) == {
        INVOKE_MODEL: 2,
        LIST_TOOLS: 2,
        CALL_TOOL: 1,
    }


@pytest.mark.parametrize("conflict", ["agent_tool", "two_providers"])
async def test_tool_name_conflict(client: Client, conflict: str) -> None:
    plugin, _, _ = _plugin([text_response("unused")])
    with pytest.raises(WorkflowFailureError) as excinfo:
        await _run(client, plugin, MCPInput(conflict=conflict))
    cause = excinfo.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "MicrosoftAgentFrameworkWorkflowError"
    assert "'echo'" in cause.message


async def test_mcp_error_result_becomes_tool_error(client: Client) -> None:
    plugin, _, _ = _plugin(
        [
            function_call_response("fail", {"reason": "disk full"}, call_id="call-1"),
            text_response("The tool failed."),
        ]
    )
    result, handle = await _run(client, plugin, MCPInput())
    assert result["texts"] == ["The tool failed."]
    (tool_result,) = [c for c in result["contents"][0] if c.startswith("result:")]
    assert tool_result.startswith("result:Error: Function failed.")
    assert await count_scheduled_activities(handle) == {
        INVOKE_MODEL: 2,
        LIST_TOOLS: 1,
        CALL_TOOL: 1,
    }


async def test_mcp_retry_then_tool_error(client: Client) -> None:
    connections: list[int] = []

    def flaky() -> MCPClient:
        connections.append(len(connections) + 1)
        if len(connections) > 1:
            raise ConnectionError("MCP server is down")
        return MCPClient(server)

    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client(_ECHO_SCRIPT)},
        mcp_clients={"echo": flaky},
        # Every Activity reconnects, so the second one meets the outage.
        mcp_connection_idle_timeout=timedelta(0),
    )
    result, handle = await _run(client, plugin, MCPInput(max_attempts=2))
    assert result["texts"] == ["Echoed."]
    (tool_result,) = [c for c in result["contents"][0] if c.startswith("result:")]
    assert tool_result.startswith("result:Error: Function failed.")
    assert await count_scheduled_activities(handle) == {
        INVOKE_MODEL: 2,
        LIST_TOOLS: 1,
        CALL_TOOL: 1,
    }
    attempts = {a.attempt for a in await started_activities(handle)}
    assert 2 in attempts
    assert len(connections) == 3


@pytest.mark.parametrize(
    ("idle_timeout", "connections"), [(None, 1), (timedelta(0), 2)]
)
async def test_idle_timeout_closes_connection(
    client: Client, idle_timeout: timedelta | None, connections: int
) -> None:
    kwargs = (
        {} if idle_timeout is None else {"mcp_connection_idle_timeout": idle_timeout}
    )
    plugin, _, opened = _plugin(_ECHO_SCRIPT, **kwargs)
    result, _ = await _run(client, plugin, MCPInput())
    assert result["texts"] == ["Echoed."]
    assert len(opened) == connections


def test_provider_requires_mcp_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_temporalio_mcp(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "temporalio.mcp":
            raise ImportError("no temporalio.mcp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_temporalio_mcp)
    with pytest.raises(
        RuntimeError, match=r"temporalio-microsoft-agent-framework\[mcp\]"
    ):
        TemporalMCPToolProvider("echo")
    with pytest.raises(
        RuntimeError, match=r"temporalio-microsoft-agent-framework\[mcp\]"
    ):
        MicrosoftAgentFrameworkPlugin(mcp_clients={"echo": lambda: MCPClient(server)})
