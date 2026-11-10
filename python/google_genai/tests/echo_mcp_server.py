"""A minimal MCP echo server for the google_genai MCP tests.

``echo`` returns its input; ``fail`` raises, so the server returns a tool error
result. Run as a script it serves over stdio; imported, ``mcp`` can be
connected to in-memory via ``mcp.shared.memory``.
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


@mcp.tool()  # pyright: ignore[reportUntypedFunctionDecorator]
def fail(message: str) -> str:
    """Raise, so the server returns a tool error result."""
    raise ValueError(f"cannot echo {message}")


if __name__ == "__main__":
    mcp.run()
