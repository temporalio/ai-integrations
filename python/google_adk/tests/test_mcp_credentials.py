"""Worker-side MCP credentials use only application and user identity."""

import inspect
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.openapi.models import APIKey
from google.adk.agents.callback_context import CallbackContext
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_manager import CredentialManager
from google.adk.events import EventActions
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager
from google.adk.tools.mcp_tool.mcp_tool import McpTool
from google.adk.tools.tool_context import ToolContext
from mcp import ClientSession
from mcp.types import CallToolResult, Tool

from temporalio import activity, workflow
from temporalio.google_adk import GoogleAdkPlugin, _mcp
from temporalio.google_adk._mcp import (
    TemporalMcpToolSetProvider,
    TemporalStatefulMcpToolSetProvider,
    TemporalToolContext,
    TemporalToolSession,
    _CallToolArguments,
    _CallToolResult,
    _StatefulCallToolArguments,
    _TemporalStatefulTool,
    _TemporalTool,
)
from temporalio.workflow import ActivityHandle

_API_KEY = "worker-api-key-canary"
_STATE_SECRET = "session-state-secret-canary"


def _api_key_scheme() -> APIKey:
    return APIKey.model_validate(
        {"type": "apiKey", "name": "X-API-Key", "in": "header"}
    )


def _identity(app_name: str, user_id: str) -> TemporalToolContext:
    return TemporalToolContext(
        None,
        "call-id",
        EventActions(),
        session=TemporalToolSession(app_name),
        user_id=user_id,
    )


@pytest.mark.parametrize("stateful", [False, True])
async def test_api_key_authenticates_inside_activity_without_entering_payloads(
    stateful: bool, monkeypatch: pytest.MonkeyPatch
):
    session = MagicMock(spec=ClientSession)
    session.call_tool = AsyncMock(
        return_value=CallToolResult.model_validate(
            {"content": [{"type": "text", "text": "success"}], "isError": False}
        )
    )
    manager = MagicMock(spec=MCPSessionManager)
    manager.create_session = AsyncMock(return_value=session)
    native = McpTool(
        mcp_tool=Tool.model_validate(
            {"name": "lookup", "description": "Look up data", "inputSchema": {}}
        ),
        mcp_session_manager=manager,
        auth_scheme=_api_key_scheme(),
        auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.API_KEY, api_key=_API_KEY
        ),
    )
    toolset = MagicMock(spec=McpToolset)
    toolset.get_tools = AsyncMock(return_value=[native])
    toolset.close = AsyncMock()
    wrapped: _TemporalTool | _TemporalStatefulTool
    argument_type: type[_CallToolArguments] | type[_StatefulCallToolArguments]
    if stateful:
        provider = TemporalStatefulMcpToolSetProvider("tools", lambda _: toolset)
        [connect] = provider._get_activities()
        call_tool = inspect.getclosurevars(connect).nonlocals["call_tool"]
        running = MagicMock(spec=_mcp._RunningToolset)
        running.get = AsyncMock(return_value=toolset)
        provider._toolsets[provider.name + "@run"] = running
        monkeypatch.setattr(
            activity, "info", lambda: SimpleNamespace(workflow_run_id="run")
        )
        wrapped = _TemporalStatefulTool(
            provider.name,
            cast(ActivityHandle, object()),
            {},
            native._get_declaration(),
            name=native.name,
            description=native.description,
            tool_definition=native.raw_mcp_tool,
        )
        argument_type = _StatefulCallToolArguments
    else:
        stateless_provider = TemporalMcpToolSetProvider("tools", lambda _: toolset)
        _, call_tool = stateless_provider._get_activities()
        wrapped = _TemporalTool(
            "tools",
            "tenant",
            None,
            native._get_declaration(),
            name=native.name,
            description=native.description,
            tool_definition=native.raw_mcp_tool,
        )
        argument_type = _CallToolArguments

    converter = GoogleAdkPlugin()._configure_data_converter(None).payload_converter
    serialized: list[bytes] = []

    async def execute(*args: Any, **_kwargs: Any) -> _CallToolResult:
        arguments = args[-1]
        argument_payloads = converter.to_payloads([arguments])
        serialized.extend(payload.data for payload in argument_payloads)
        [decoded] = converter.from_payloads(argument_payloads, [argument_type])
        assert decoded.tool_context.session.app_name == "app"
        assert decoded.tool_context.user_id == "alice"
        result = await call_tool(decoded)
        result_payloads = converter.to_payloads([result])
        serialized.extend(payload.data for payload in result_payloads)
        [result] = converter.from_payloads(result_payloads, [_CallToolResult])
        assert result.tool_context.session.app_name == "app"
        assert result.tool_context.user_id == "alice"
        return result

    if stateful:
        monkeypatch.setattr(_mcp, "_execute_operation", execute)
    else:
        monkeypatch.setattr(workflow, "execute_activity", execute)
    context = MagicMock(spec=ToolContext)
    context.tool_confirmation = None
    context.function_call_id = "call-id"
    context._event_actions = EventActions()
    context.session = SimpleNamespace(
        app_name="app", state={"secret": _STATE_SECRET, "api_key": _API_KEY}
    )
    context.state = context.session.state
    context.user_id = "alice"

    result = await wrapped.run_async(args={"query": "value"}, tool_context=context)

    assert result["content"] == [{"type": "text", "text": "success"}]
    manager.create_session.assert_awaited_once_with(headers={"X-API-Key": _API_KEY})
    session.call_tool.assert_awaited_once()
    assert session.call_tool.call_args.args == ("lookup",)
    assert session.call_tool.call_args.kwargs["arguments"] == {"query": "value"}
    assert len(serialized) == 2
    for payload in serialized:
        assert _API_KEY.encode() not in payload
        assert _STATE_SECRET.encode() not in payload
    if not stateful:
        toolset.close.assert_awaited_once()


async def test_credential_locks_follow_application_and_user_identity():
    manager = CredentialManager(
        AuthConfig(
            auth_scheme=_api_key_scheme(),
            raw_auth_credential=AuthCredential(
                auth_type=AuthCredentialTypes.API_KEY, api_key=_API_KEY
            ),
        )
    )
    alice = _identity("app", "alice")
    same_user = _identity("app", "alice")
    bob = _identity("app", "bob")
    other_app = _identity("other-app", "alice")

    alice_lock = manager._get_credential_lock(
        cast(CallbackContext, cast(object, alice))
    )

    assert (
        manager._get_credential_lock(cast(CallbackContext, cast(object, same_user)))
        is alice_lock
    )
    assert (
        manager._get_credential_lock(cast(CallbackContext, cast(object, bob)))
        is not alice_lock
    )
    assert (
        manager._get_credential_lock(cast(CallbackContext, cast(object, other_app)))
        is not alice_lock
    )


def test_legacy_tool_context_payload_decodes_without_identity():
    converter = GoogleAdkPlugin()._configure_data_converter(None).payload_converter
    payloads = converter.to_payloads(
        [
            {
                "tool_confirmation": None,
                "function_call_id": "call-id",
                "event_actions": {},
            }
        ]
    )

    [context] = converter.from_payloads(payloads, [TemporalToolContext])

    assert context.function_call_id == "call-id"
    assert context.session is None
    assert context.user_id is None
