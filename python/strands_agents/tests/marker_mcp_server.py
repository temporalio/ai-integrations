import base64

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent

mcp = FastMCP("marker-server")

# A PNG signature and header chunk: valid bytes, but not valid UTF-8.
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


@mcp.tool()
def lookup() -> CallToolResult:
    """Return an image next to structured content and metadata that look like encoded bytes."""
    return CallToolResult(
        content=[
            TextContent(type="text", text="found"),
            ImageContent(
                type="image",
                data=base64.b64encode(PNG).decode(),
                mimeType="image/png",
            ),
        ],
        structuredContent={
            "__bytes_encoded__": True,
            "data": "SGVsbG8=",
            "other": "retained",
        },
        _meta={"__bytes_encoded__": True, "data": 7, "nested": {"data": "x"}},
    )


if __name__ == "__main__":
    mcp.run()
