"""A deliberately non-idempotent MCP service for native effect recovery probes."""

from __future__ import annotations

from typing import Any

from mcp.server import Server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
    ToolAnnotations,
)

from tests.hybrid.engine import native_id
from tests.hybrid.native_store import NativeStore

NAME = "mcp__external__charge"


def configure_effects(store: NativeStore) -> None:
    with store.connect() as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS charges (seq INTEGER PRIMARY KEY, id TEXT, amount INTEGER)"
        )

    async def list_tools(ctx: Any, params: Any) -> ListToolsResult:
        del ctx, params
        return ListToolsResult(
            tools=[
                Tool(
                    name="charge",
                    description="Charge an amount once.",
                    input_schema={
                        "type": "object",
                        "properties": {"amount": {"type": "integer"}},
                        "required": ["amount"],
                    },
                    annotations=ToolAnnotations(
                        read_only_hint=False,
                        destructive_hint=True,
                        idempotent_hint=False,
                        open_world_hint=True,
                    ),
                )
            ]
        )

    async def charge(ctx: Any, params: Any) -> CallToolResult:
        tid = native_id(ctx.meta)
        with store.connect() as db:
            # There is intentionally no uniqueness constraint or service dedup.
            db.execute(
                "INSERT INTO charges(id,amount) VALUES (?,?)",
                (tid, params.arguments["amount"]),
            )
        if store.phase == NAME + "-after-effect":
            store.mark(tid, "effect-without-result")
            await store.held.wait()
        return CallToolResult(content=[TextContent(type="text", text="CHARGED 7")])

    server = Server("external", on_list_tools=list_tools, on_call_tool=charge)
    store.tools.add(NAME)
    store.mcp_servers = {
        "external": {"type": "sdk", "name": "external", "instance": server}
    }
