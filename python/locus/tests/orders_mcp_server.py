"""MCP v2 server used by the MCP tests: an order lookup, a structured result, a failure and an image."""

import os

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("orders")

# A PNG signature and header chunk: valid bytes, but not valid UTF-8.
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


@server.tool()
def lookup_order(order_id: str) -> str:
    """Return the status of an order."""
    with open(os.environ["ORDERS_MCP_CALLS_FILE"], "a") as calls:
        calls.write(f"{order_id}\n")
    return f"order {order_id}: shipped"


@server.tool()
def order_totals(order_id: str) -> dict[str, float]:
    """Return an order's totals."""
    del order_id
    return {"subtotal": 40.0, "tax": 3.5}


@server.tool()
def cancel_order(order_id: str) -> str:
    """Cancel an order."""
    raise ToolError(f"order {order_id} already shipped")


@server.tool()
def label(order_id: str) -> Image:
    """Return an order's shipping label."""
    del order_id
    return Image(data=PNG, format="png")


if __name__ == "__main__":
    server.run()
