"""MCP tools through ``TemporalMCPToolProvider`` and ``temporalio-mcp``."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest
from agent_framework import Agent, tool
from mcp import Client as MCPClient
from mcp import StdioServerParameters, stdio_client

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
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
    scheduled_activities,
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
    approval_mapping: bool = False
    parse: bool = False
    detailed_errors: bool = False
    # heartbeat_set tells an explicit None (heartbeats off) from unset.
    heartbeat_set: bool = False
    heartbeat_seconds: float | None = None


def _count_blocks(result: Any) -> str:
    """A ``parse_tool_results`` replacement: the number of content blocks."""
    return f"{len(result.content)} blocks"


def _result_items(response: Any) -> list[list[dict[str, Any]]]:
    """The content items of every function result, as plain dicts.

    For a tool that returns content, the framework joins the text into
    ``result`` and keeps the full list on ``items``.
    """
    items: list[list[dict[str, Any]]] = []
    for message in response.messages:
        for content in message.contents:
            if content.type != "function_result":
                continue
            listed = getattr(content, "items", None)
            if isinstance(listed, list):
                items.append(
                    [
                        {
                            "type": c.type,
                            "text": c.text,
                            "uri": c.uri,
                            "media_type": c.media_type,
                        }
                        for c in listed
                    ]
                )
            else:
                items.append([{"type": "raw", "text": str(content.result)}])
    return items


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
        approval_mode: Any = "always_require" if input.approval else None
        if input.approval_mapping:
            # MCPTool's per-tool form, which the provider refuses.
            approval_mode = {"always_require_approval": ["echo"]}
        heartbeat: dict[str, Any] = {}
        if input.heartbeat_set:
            heartbeat["heartbeat_timeout"] = (
                timedelta(seconds=input.heartbeat_seconds)
                if input.heartbeat_seconds is not None
                else None
            )
        providers = [
            TemporalMCPToolProvider(
                "echo",
                cache_tools=input.cache_tools,
                tool_name_prefix=input.prefix,
                allowed_tools=input.allowed,
                approval_mode=approval_mode,
                parse_tool_results=_count_blocks if input.parse else None,
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=retry_policy,
                **heartbeat,
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
        chat_client = TemporalChatClient(
            "model",
            function_invocation_configuration=(
                {"include_detailed_errors": True} if input.detailed_errors else None
            ),
        )
        agent = Agent(client=chat_client, tools=tools, context_providers=providers)
        session = agent.create_session()
        texts: list[str] = []
        contents: list[list[str]] = []
        results: list[list[list[dict[str, Any]]]] = []
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
            results.append(_result_items(response))
        return {"texts": texts, "contents": contents, "results": results}


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
    # The model received the server's full input schema, not just the names,
    # and the slash in files/read was normalized away.
    tools = {t.name: t for t in chat().calls[0]["options"]["tools"]}
    assert set(tools) == {
        "echo",
        "shout",
        "fail",
        "wait",
        "files-read",
        "mixed",
        "empty",
    }
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


@pytest.mark.parametrize("prefix", [None, "gh_", "gh"])
async def test_mcp_tool_names_normalized(client: Client, prefix: str | None) -> None:
    exposed = "files-read" if prefix is None else "gh_files-read"
    plugin, chat, _ = _plugin(
        [
            function_call_response(exposed, {"path": "/etc/hosts"}, call_id="call-1"),
            text_response("Read."),
        ]
    )
    result, handle = await _run(
        client, plugin, MCPInput(prefix=prefix, allowed=["files/read"])
    )
    # The model sees the normalized name (the slash became a dash, the
    # separator between the prefix and the name is not doubled); the call
    # reaches the server under its own name.
    assert [t.name for t in chat().calls[0]["options"]["tools"]] == [exposed]
    assert result["contents"][0] == [
        f"call:{exposed}",
        'result:{"result": "contents of /etc/hosts"}',
        "text:Read.",
    ]
    assert await count_scheduled_activities(handle) == {
        INVOKE_MODEL: 2,
        LIST_TOOLS: 1,
        CALL_TOOL: 1,
    }


async def test_mcp_approval_mapping_rejected(client: Client) -> None:
    plugin, _, _ = _plugin([text_response("unused")])
    with pytest.raises(WorkflowFailureError) as excinfo:
        await _run(client, plugin, MCPInput(approval_mapping=True))
    cause = excinfo.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "MicrosoftAgentFrameworkWorkflowError"
    assert "always_require_approval" in cause.message


async def test_mcp_mixed_content(client: Client) -> None:
    plugin, _, _ = _plugin(
        [
            function_call_response("mixed", {}, call_id="call-1"),
            function_call_response("empty", {}, call_id="call-2"),
            text_response("Seen."),
        ]
    )
    result, _ = await _run(client, plugin, MCPInput())
    assert result["texts"] == ["Seen."]
    mixed, empty = result["results"][0]
    # Text and embedded text become text; images, blobs and links become URIs.
    assert mixed == [
        {"type": "text", "text": "hello", "uri": None, "media_type": None},
        {
            "type": "data",
            "text": None,
            "uri": "data:image/png;base64,aW1n",
            "media_type": "image/png",
        },
        {"type": "text", "text": "embedded", "uri": None, "media_type": None},
        {
            "type": "data",
            "text": None,
            "uri": "data:application/octet-stream;base64,YmxvYg==",
            "media_type": "application/octet-stream",
        },
        {
            "type": "uri",
            "text": None,
            "uri": "https://example.com/doc",
            "media_type": "text/html",
        },
    ]
    assert empty == [{"type": "text", "text": "null", "uri": None, "media_type": None}]


async def test_mcp_parse_tool_results(client: Client) -> None:
    plugin, _, _ = _plugin(
        [
            function_call_response("mixed", {}, call_id="call-1"),
            text_response("Counted."),
        ]
    )
    result, _ = await _run(client, plugin, MCPInput(parse=True))
    assert result["texts"] == ["Counted."]
    # The replacement parser's string is the whole tool result.
    assert result["contents"][0][1] == "result:5 blocks"


async def test_mcp_error_result_becomes_tool_error(client: Client) -> None:
    plugin, _, _ = _plugin(
        [
            function_call_response("fail", {"reason": "disk full"}, call_id="call-1"),
            text_response("The tool failed."),
        ]
    )
    result, handle = await _run(client, plugin, MCPInput(detailed_errors=True))
    assert result["texts"] == ["The tool failed."]
    (tool_result,) = [c for c in result["contents"][0] if c.startswith("result:")]
    assert tool_result.startswith("result:Error: Function failed.")
    # The server's error text reaches the model with include_detailed_errors
    # (the MCP server prefixes it with the tool's name).
    assert "MCP tool returned an error: Error executing tool fail: disk full" in (
        tool_result
    )
    assert await count_scheduled_activities(handle) == {
        INVOKE_MODEL: 2,
        LIST_TOOLS: 1,
        CALL_TOOL: 1,
    }


async def test_mcp_retry_then_tool_error(client: Client) -> None:
    connections: list[int] = []

    # After the first connection the server is gone: the next client's
    # process exits at once, so connecting fails and temporalio-mcp retries.
    dead = StdioServerParameters(
        command=sys.executable, args=["-c", "import sys; sys.exit(3)"]
    )

    def flaky() -> MCPClient:
        connections.append(len(connections) + 1)
        if len(connections) > 1:
            return MCPClient(stdio_client(dead))
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
    # None is passed as such: it keeps the connection until shutdown, where
    # leaving the argument out would test the 5 minute default instead.
    plugin, _, opened = _plugin(_ECHO_SCRIPT, mcp_connection_idle_timeout=idle_timeout)
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


@pytest.mark.parametrize(
    ("kind", "attempts"),
    [("raises", 2), ("wrong_type", 1)],
    ids=["raises", "wrong_type"],
)
async def test_mcp_factory_failure(client: Client, kind: str, attempts: int) -> None:
    def factory() -> Any:
        if kind == "raises":
            raise KeyError("MCP_TOKEN")
        return object()

    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client(["unused"])},
        mcp_clients={"echo": factory},
    )
    async with new_worker(client, MCPWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            MCPWorkflow.run,
            MCPInput(max_attempts=2),
            id=workflow_id("mcp-factory"),
            task_queue=worker.task_queue,
        )
        with pytest.raises(WorkflowFailureError) as excinfo:
            await handle.result()
    cause = excinfo.value.cause
    assert isinstance(cause, ActivityError)
    failure = cause.__cause__
    assert isinstance(failure, ApplicationError)
    # Same policy as the chat client factories: a factory that raises is
    # retried under the call's retry policy (two attempts here), one that
    # returns the wrong type fails the listing at once. Either way no model
    # call runs.
    assert failure.type == "MicrosoftAgentFrameworkMCPClientFactoryError"
    assert failure.non_retryable is (kind == "wrong_type")
    assert "'echo'" in failure.message
    assert await count_scheduled_activities(handle) == {LIST_TOOLS: 1}
    (started,) = await started_activities(handle)
    assert started.attempt == attempts


def test_provider_passes_heartbeat_timeout_only_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import temporalio.mcp

    configs: list[dict[str, Any]] = []

    class Recorder:
        def __init__(
            self, server: str, *, activity_config: Any, cache_tools_list: bool
        ):
            configs.append(dict(activity_config))

    monkeypatch.setattr(temporalio.mcp, "TemporalMCPClient", Recorder)
    TemporalMCPToolProvider("echo")
    TemporalMCPToolProvider("echo", heartbeat_timeout=timedelta(seconds=3))
    TemporalMCPToolProvider("echo", heartbeat_timeout=None)
    default, three_seconds, off = configs
    # Left unset, the key is absent so temporalio-mcp applies its own default;
    # a value, including an explicit None, reaches temporalio-mcp as given.
    assert "heartbeat_timeout" not in default
    assert three_seconds["heartbeat_timeout"] == timedelta(seconds=3)
    assert "heartbeat_timeout" in off and off["heartbeat_timeout"] is None


@pytest.mark.parametrize(
    ("heartbeat_set", "seconds", "expected"),
    [(False, None, 10), (True, 3, 3), (True, None, 0)],
    ids=["default", "set", "off"],
)
async def test_mcp_heartbeat_timeout(
    client: Client, heartbeat_set: bool, seconds: float | None, expected: int
) -> None:
    plugin, _, _ = _plugin(_ECHO_SCRIPT)
    result, handle = await _run(
        client,
        plugin,
        MCPInput(heartbeat_set=heartbeat_set, heartbeat_seconds=seconds),
    )
    assert result["texts"] == ["Echoed."]
    # temporalio-mcp's 10 second default, the provider's value, or none at all
    # (an explicit None turns heartbeats off), on both MCP Activities.
    timeouts = {
        attrs.activity_type.name: attrs.heartbeat_timeout.ToTimedelta()
        for attrs in await scheduled_activities(handle)
        if attrs.activity_type.name in (LIST_TOOLS, CALL_TOOL)
    }
    assert timeouts == {
        LIST_TOOLS: timedelta(seconds=expected),
        CALL_TOOL: timedelta(seconds=expected),
    }
