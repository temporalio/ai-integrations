import warnings
from typing import Any

import pytest
from mcp.types import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
)

try:
    from mcp import MCPError as McpError  # type: ignore[attr-defined]  # mcp>=2
except ImportError:  # mcp<2
    from mcp.shared.exceptions import McpError  # type: ignore[no-redef]

from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.openai_agents import StatelessMCPServerProvider
from temporalio.openai_agents._mcp import (
    _StatelessCallToolsArguments,
    _StatelessGetPromptArguments,
    _StatelessListPromptsArguments,
    _StatelessListToolsArguments,
)


class _FailingServer:
    """Stand-in MCP server whose every operation raises ``McpError``."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.cleaned_up = False

    async def connect(self) -> None:
        pass

    async def cleanup(self) -> None:
        self.cleaned_up = True

    async def list_tools(self, *args: Any, **kwargs: Any):
        raise self._error

    async def call_tool(self, *args: Any, **kwargs: Any):
        raise self._error

    async def list_prompts(self, *args: Any, **kwargs: Any):
        raise self._error

    async def get_prompt(self, *args: Any, **kwargs: Any):
        raise self._error


def _activity(server: _FailingServer, name: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        provider = StatelessMCPServerProvider("srv", lambda: server)  # type: ignore[arg-type,return-value]
    for act in provider._get_activities():
        if activity._Definition.must_from_callable(act).name == f"srv-stateless-{name}":
            return act
    raise AssertionError(name)


@pytest.mark.parametrize(
    "code,retryable",
    [
        (PARSE_ERROR, False),
        (INVALID_REQUEST, False),
        (METHOD_NOT_FOUND, False),
        (INVALID_PARAMS, False),
        (INTERNAL_ERROR, True),
    ],
)
@pytest.mark.parametrize(
    "operation,args",
    [
        ("list-tools", _StatelessListToolsArguments(None)),
        ("call-tool-v2", _StatelessCallToolsArguments("t", {}, None)),
        ("list-prompts", _StatelessListPromptsArguments(None)),
        ("get-prompt-v2", _StatelessGetPromptArguments("p", {}, None)),
    ],
)
async def test_stateless_mcp_protocol_error_retryability(
    code: int, retryable: bool, operation: str, args: Any
) -> None:
    server = _FailingServer(McpError(code, "boom"))
    with pytest.raises(ApplicationError) as exc:
        await _activity(server, operation)(args)
    assert exc.value.type == "MCPProtocolError"
    assert exc.value.non_retryable is (not retryable)
    assert server.cleaned_up
