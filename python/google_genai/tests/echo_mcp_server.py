"""A minimal MCP echo server for the google_genai MCP tests.

Run as a script it serves over stdio; imported, ``mcp`` can be connected to
in-memory via ``mcp.shared.memory``.
"""

try:
    from mcp.server.mcpserver import MCPServer  # type: ignore[import-not-found]
except ImportError:  # MCP 1.x
    from mcp.server.fastmcp import (  # type: ignore[attr-defined, no-redef]
        FastMCP as MCPServer,  # pyright: ignore[reportAttributeAccessIssue]
    )

mcp = MCPServer("echo-server")


@mcp.tool()  # pyright: ignore[reportUntypedFunctionDecorator]
def echo(message: str) -> str:
    """Return the input message unchanged."""
    return message


if __name__ == "__main__":
    mcp.run()
