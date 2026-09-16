import asyncio
from datetime import timedelta
from typing import Any, cast

import pytest
from mcp import Client, MCPError
from mcp.server.mcpserver import MCPServer
from mcp.types import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    URL_ELICITATION_REQUIRED,
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    Prompt,
    PromptMessage,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
    TextContent,
    TextResourceContents,
    Tool,
)

from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.mcp import _activities
from temporalio.mcp._activities import _MCPActivities
from temporalio.mcp._client import _MCPClientBackend


class FakeClient:
    protocol_version = "2026-07-28"

    def __init__(self) -> None:
        self.metas: list[dict[str, Any] | None] = []
        self.closed = False

    async def __aenter__(self) -> "FakeClient":
        self.closed = False
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.closed = True

    def _page(self, cursor: str | None) -> tuple[str, str | None]:
        return ("one", "next") if cursor is None else ("two", None)

    def _envelope(self, name: str) -> dict[str, Any]:
        return {
            "meta": {f"page-{name}": True},
            "ttl_ms": 2000 if name == "one" else 1000,
            "cache_scope": "public" if name == "one" else "private",
            "result_type": "test/list",
        }

    async def list_tools(
        self, *, cursor: str | None, cache_mode: str
    ) -> ListToolsResult:
        assert cache_mode == "bypass"
        name, next_cursor = self._page(cursor)
        return ListToolsResult(
            tools=[Tool(name=name, input_schema={"type": "object"})],
            next_cursor=next_cursor,
            **self._envelope(name),
        )

    async def list_prompts(
        self, *, cursor: str | None, cache_mode: str
    ) -> ListPromptsResult:
        assert cache_mode == "bypass"
        name, next_cursor = self._page(cursor)
        return ListPromptsResult(
            prompts=[Prompt(name=name)],
            next_cursor=next_cursor,
            **self._envelope(name),
        )

    async def list_resources(
        self, *, cursor: str | None, cache_mode: str
    ) -> ListResourcesResult:
        assert cache_mode == "bypass"
        name, next_cursor = self._page(cursor)
        return ListResourcesResult(
            resources=[Resource(name=name, uri=f"test://{name}")],
            next_cursor=next_cursor,
            **self._envelope(name),
        )

    async def list_resource_templates(
        self, *, cursor: str | None, cache_mode: str
    ) -> ListResourceTemplatesResult:
        assert cache_mode == "bypass"
        name, next_cursor = self._page(cursor)
        return ListResourceTemplatesResult(
            resource_templates=[
                ResourceTemplate(name=name, uri_template=f"test://{name}/{{id}}")
            ],
            next_cursor=next_cursor,
            **self._envelope(name),
        )

    async def call_tool(
        self,
        name: str,
        _arguments: dict[str, Any] | None,
        *,
        meta: dict[str, Any] | None,
    ) -> CallToolResult:
        if self.closed:
            raise RuntimeError("MCP client is closed")
        self.metas.append(meta)
        return CallToolResult(content=[TextContent(text=name)])

    async def get_prompt(
        self, name: str, _arguments: dict[str, str] | None
    ) -> GetPromptResult:
        return GetPromptResult(
            messages=[PromptMessage(role="user", content=TextContent(text=name))]
        )

    async def read_resource(self, uri: str, *, cache_mode: str) -> ReadResourceResult:
        assert cache_mode == "bypass"
        return ReadResourceResult(
            contents=[TextResourceContents(uri=uri, text="contents")]
        )


def _activity_by_name(support: _MCPActivities, name: str) -> Any:
    for fn in support.activities:
        definition = activity._Definition.from_callable(fn)
        if definition is not None and definition.name == name:
            return fn
    raise AssertionError(f"Activity {name!r} was not registered")


async def test_operations_are_plain_json_and_lists_are_fully_paginated() -> None:
    client = FakeClient()
    support = _MCPActivities(
        {"test": lambda: _MCPClientBackend(cast(Any, client))},
        idle_timeout=timedelta(minutes=5),
    )
    functions: dict[str, Any] = {}
    for fn in support.activities:
        definition = activity._Definition.from_callable(fn)
        assert definition is not None and definition.name is not None
        functions[definition.name] = fn
    request: dict[str, Any] = {"factory_argument": None}
    try:
        for operation, result_key in (
            ("list-tools", "tools"),
            ("list-prompts", "prompts"),
            ("list-resources", "resources"),
            ("list-resource-templates", "resource_templates"),
        ):
            result = await functions[f"temporalio.mcp.test.{operation}"](request)
            assert [item["name"] for item in result[result_key]] == ["one", "two"]
            assert result["next_cursor"] is None
            assert result["meta"] == {"page-one": True, "page-two": True}
            assert result["ttl_ms"] == 1000
            assert result["cache_scope"] == "private"
            assert result["result_type"] == "test/list"

        tool_result = await functions["temporalio.mcp.test.call-tool"](
            {**request, "name": "echo", "arguments": {}, "meta": {"trace": "value"}}
        )
        assert tool_result["content"][0]["text"] == "echo"

        prompt_result = await functions["temporalio.mcp.test.get-prompt"](
            {**request, "name": "prompt", "arguments": {}}
        )
        assert prompt_result["messages"][0]["content"]["text"] == "prompt"

        resource_result = await functions["temporalio.mcp.test.read-resource"](
            {**request, "uri": "test://resource"}
        )
        assert resource_result["contents"][0]["text"] == "contents"

        # call_tool is the only operation that carries request metadata.
        assert client.metas == [{"trace": "value"}]
    finally:
        await support._pool.close()


async def test_repeated_pagination_cursor_fails_without_retry() -> None:
    class RepeatingCursorClient(FakeClient):
        def _page(self, cursor: str | None) -> tuple[str, str | None]:
            return "item", "repeated"

    backend = _MCPClientBackend(cast(Any, RepeatingCursorClient()))
    async with backend:
        with pytest.raises(ApplicationError, match="repeated pagination cursor") as err:
            await backend.list_tools()
    assert err.value.type == "MCPProtocolError"
    assert err.value.non_retryable is True


async def test_unknown_resource_protocol_error_fails_without_retry() -> None:
    server = MCPServer("resources")

    @server.resource("test://known", name="known")
    def known_resource() -> str:  # type: ignore[reportUnusedFunction]
        return "known"

    support = _MCPActivities(
        {"test": lambda: _MCPClientBackend(Client(server))},
        idle_timeout=timedelta(minutes=5),
    )
    read_resource = _activity_by_name(
        support,
        "temporalio.mcp.test.read-resource",
    )
    try:
        with pytest.raises(ApplicationError, match="Unknown resource") as err:
            await read_resource({"factory_argument": None, "uri": "test://unknown"})
    finally:
        await support._pool.close()

    assert err.value.type == "MCPProtocolError"
    assert err.value.non_retryable is True
    assert err.value.details == (INVALID_PARAMS, {"uri": "test://unknown"})


async def test_internal_protocol_error_remains_retryable() -> None:
    class InternalErrorClient(FakeClient):
        async def get_prompt(
            self, name: str, _arguments: dict[str, str] | None
        ) -> GetPromptResult:
            raise MCPError(INTERNAL_ERROR, f"Failed to get {name}")

    support = _MCPActivities(
        {"test": lambda: _MCPClientBackend(cast(Any, InternalErrorClient()))},
        idle_timeout=timedelta(minutes=5),
    )
    get_prompt = _activity_by_name(
        support,
        "temporalio.mcp.test.get-prompt",
    )
    try:
        with pytest.raises(ApplicationError, match="Failed to get prompt") as err:
            await get_prompt(
                {"factory_argument": None, "name": "prompt", "arguments": {}}
            )
    finally:
        await support._pool.close()

    assert err.value.type == "MCPProtocolError"
    assert err.value.non_retryable is False
    assert err.value.details == (INTERNAL_ERROR,)


async def test_permanent_protocol_error_fails_without_retry() -> None:
    error_data = {
        "elicitations": [{"url": "https://example.test/login", "description": "Log in"}]
    }

    class ElicitingClient(FakeClient):
        async def call_tool(
            self,
            name: str,
            _arguments: dict[str, Any] | None,
            *,
            meta: dict[str, Any] | None,
        ) -> CallToolResult:
            raise MCPError(
                URL_ELICITATION_REQUIRED,
                f"{name} needs a browser",
                error_data,
            )

    support = _MCPActivities(
        {"test": lambda: _MCPClientBackend(cast(Any, ElicitingClient()))},
        idle_timeout=timedelta(minutes=5),
    )
    call_tool = _activity_by_name(support, "temporalio.mcp.test.call-tool")
    try:
        with pytest.raises(ApplicationError, match="needs a browser") as err:
            await call_tool(
                {
                    "factory_argument": None,
                    "name": "login",
                    "arguments": {},
                    "meta": None,
                }
            )
    finally:
        await support._pool.close()

    assert err.value.type == "MCPProtocolError"
    assert err.value.non_retryable is True
    assert err.value.details == (URL_ELICITATION_REQUIRED, error_data)


async def test_invalid_server_response_fails_without_retry() -> None:
    class InvalidResponseClient(FakeClient):
        async def get_prompt(
            self, name: str, _arguments: dict[str, str] | None
        ) -> GetPromptResult:
            return GetPromptResult.model_validate({"messages": "not-a-list"})

    support = _MCPActivities(
        {"test": lambda: _MCPClientBackend(cast(Any, InvalidResponseClient()))},
        idle_timeout=timedelta(minutes=5),
    )
    get_prompt = _activity_by_name(
        support,
        "temporalio.mcp.test.get-prompt",
    )
    try:
        with pytest.raises(ApplicationError, match="invalid response") as err:
            await get_prompt(
                {"factory_argument": None, "name": "prompt", "arguments": {}}
            )
    finally:
        await support._pool.close()

    assert err.value.type == "MCPProtocolError"
    assert err.value.non_retryable is True
    [validation_errors] = err.value.details
    assert validation_errors[0]["loc"] == ("messages",)


async def test_worker_reentry_before_close_starts_keeps_connection_open() -> None:
    client = FakeClient()
    support = _MCPActivities(
        {"test": lambda: _MCPClientBackend(cast(Any, client))},
        idle_timeout=None,
    )
    start_next_worker = asyncio.Event()
    next_worker_entered = asyncio.Event()
    release_next_worker = asyncio.Event()
    next_worker: asyncio.Task[CallToolResult] | None = None

    async def run_next_worker(first_backend: Any) -> CallToolResult:
        await start_next_worker.wait()
        async with support.run_context():
            async with support._pool.backend("test", factory_argument=None) as backend:
                # The next Worker entered before the prior Worker's queued
                # close task, so it must retain this still-live generation.
                assert backend is first_backend
                next_worker_entered.set()
                await release_next_worker.wait()
                return await backend.call_tool("echo", {}, None)

    try:
        async with support.run_context():
            async with support._pool.backend(
                "test", factory_argument=None
            ) as first_backend:
                pass
            next_worker = asyncio.create_task(run_next_worker(first_backend))
            start_next_worker.set()

        await next_worker_entered.wait()
        release_next_worker.set()
        result = await next_worker
        assert cast(TextContent, result.content[0]).text == "echo"
    finally:
        release_next_worker.set()
        if next_worker is not None:
            await asyncio.gather(next_worker, return_exceptions=True)
        await support._pool.close()


async def test_shared_plugin_closes_after_last_run_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = _MCPActivities({}, idle_timeout=None)
    closes = 0

    async def close() -> None:
        nonlocal closes
        closes += 1

    monkeypatch.setattr(support._pool, "close", close)
    async with support.run_context():
        async with support.run_context():
            pass
        assert closes == 0
    assert closes == 1
    assert not support._run_contexts


async def test_run_context_close_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = _MCPActivities({}, idle_timeout=None)
    never = asyncio.Event()

    async def close() -> None:
        await never.wait()

    monkeypatch.setattr(support._pool, "close", close)
    monkeypatch.setattr(_activities, "_CLOSE_TIMEOUT_SECONDS", 0.01)
    async with support.run_context():
        pass

    assert len(support._abandoned_closes) == 1
    close_task = next(iter(support._abandoned_closes))
    close_task.cancel()
    await asyncio.gather(close_task, return_exceptions=True)


async def test_run_context_finishes_close_after_worker_wrapper_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = _MCPActivities({}, idle_timeout=None)
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    async def close() -> None:
        close_started.set()
        await allow_close.wait()

    async def run() -> None:
        async with support.run_context():
            pass

    monkeypatch.setattr(support._pool, "close", close)
    task = asyncio.create_task(run())
    await close_started.wait()
    task.cancel()
    allow_close.set()
    await task
