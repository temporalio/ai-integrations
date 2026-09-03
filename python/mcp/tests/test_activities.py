import asyncio
from datetime import timedelta
from typing import Any, cast

import pytest
from mcp.types import (
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
from temporalio.contrib.mcp import _activities
from temporalio.contrib.mcp._activities import _MCPActivities
from temporalio.contrib.mcp._client import _MCPClientBackend
from temporalio.exceptions import ApplicationError


class FakeClient:
    protocol_version = "2026-07-28"

    def __init__(self) -> None:
        self.metas: list[dict[str, Any] | None] = []

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def _page(self, cursor: str | None) -> tuple[str, str | None]:
        return ("one", "next") if cursor is None else ("two", None)

    async def list_tools(
        self, *, cursor: str | None, cache_mode: str
    ) -> ListToolsResult:
        assert cache_mode == "bypass"
        name, next_cursor = self._page(cursor)
        return ListToolsResult(
            tools=[Tool(name=name, input_schema={"type": "object"})],
            next_cursor=next_cursor,
        )

    async def list_prompts(self, *, cursor: str | None) -> ListPromptsResult:
        name, next_cursor = self._page(cursor)
        return ListPromptsResult(prompts=[Prompt(name=name)], next_cursor=next_cursor)

    async def list_resources(self, *, cursor: str | None) -> ListResourcesResult:
        name, next_cursor = self._page(cursor)
        return ListResourcesResult(
            resources=[Resource(name=name, uri=f"test://{name}")],
            next_cursor=next_cursor,
        )

    async def list_resource_templates(
        self, *, cursor: str | None
    ) -> ListResourceTemplatesResult:
        name, next_cursor = self._page(cursor)
        return ListResourceTemplatesResult(
            resource_templates=[
                ResourceTemplate(name=name, uri_template=f"test://{name}/{{id}}")
            ],
            next_cursor=next_cursor,
        )

    async def call_tool(
        self,
        name: str,
        _arguments: dict[str, Any] | None,
        *,
        meta: dict[str, Any] | None,
    ) -> CallToolResult:
        self.metas.append(meta)
        return CallToolResult(content=[TextContent(text=name)])

    async def get_prompt(
        self, name: str, _arguments: dict[str, str] | None
    ) -> GetPromptResult:
        return GetPromptResult(
            messages=[PromptMessage(role="user", content=TextContent(text=name))]
        )

    async def read_resource(self, uri: str) -> ReadResourceResult:
        return ReadResourceResult(
            contents=[TextResourceContents(uri=uri, text="contents")]
        )


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
            result = await functions[f"temporalio.contrib.mcp.test.{operation}"](
                request
            )
            assert [item["name"] for item in result[result_key]] == ["one", "two"]
            assert result["next_cursor"] is None

        tool_result = await functions["temporalio.contrib.mcp.test.call-tool"](
            {**request, "name": "echo", "arguments": {}, "meta": {"trace": "value"}}
        )
        assert tool_result["content"][0]["text"] == "echo"

        prompt_result = await functions["temporalio.contrib.mcp.test.get-prompt"](
            {**request, "name": "prompt", "arguments": {}}
        )
        assert prompt_result["messages"][0]["content"]["text"] == "prompt"

        resource_result = await functions["temporalio.contrib.mcp.test.read-resource"](
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
