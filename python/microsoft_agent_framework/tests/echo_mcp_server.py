"""In-process MCP server (MCP Python SDK v2) for the tests."""

import asyncio
import base64

from mcp import types
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("echo")


@server.tool()
def echo(value: str, times: int = 1) -> str:
    """Return the input repeated."""
    return value * times


@server.tool()
def shout(value: str) -> str:
    """Return the input in upper case."""
    return value.upper()


@server.tool()
def fail(reason: str) -> str:
    """Fail with the given reason."""
    # ToolError reaches the client as an error result with this message; any
    # other exception is redacted to "Error executing tool fail".
    raise ToolError(reason)


@server.tool()
async def wait(seconds: float) -> str:
    """Wait for the given number of seconds."""
    await asyncio.sleep(seconds)
    return f"waited {seconds}"


@server.tool(name="files/read")
def files_read(path: str) -> str:
    """Read a file (a tool whose name holds a slash)."""
    return f"contents of {path}"


@server.tool()
def mixed() -> list[types.ContentBlock]:
    """Return text, an image, two embedded resources and a resource link."""
    return [
        types.TextContent(type="text", text="hello"),
        types.ImageContent(
            type="image", data=base64.b64encode(b"img").decode(), mime_type="image/png"
        ),
        types.EmbeddedResource(
            type="resource",
            resource=types.TextResourceContents(
                uri="file:///a.txt", text="embedded", mime_type="text/plain"
            ),
        ),
        types.EmbeddedResource(
            type="resource",
            resource=types.BlobResourceContents(
                uri="file:///b.bin",
                blob=base64.b64encode(b"blob").decode(),
                mime_type="application/octet-stream",
            ),
        ),
        types.ResourceLink(
            type="resource_link",
            name="doc",
            uri="https://example.com/doc",
            mime_type="text/html",
        ),
    ]


@server.tool(structured_output=False)
def empty() -> None:
    """Return nothing."""
    return None


if __name__ == "__main__":
    server.run()
