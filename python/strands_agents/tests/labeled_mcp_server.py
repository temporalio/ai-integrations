import sys

from mcp.types import ToolAnnotations

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # mcp 2 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import (  # type: ignore[no-redef]  # pyright: ignore[reportMissingImports]
        MCPServer as FastMCP,
    )

# The first argument labels this server so a test can tell which one answered.
label = sys.argv[1]

mcp = FastMCP(f"labeled-{label}")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))  # pyright: ignore[reportUntypedFunctionDecorator]
def echo(message: str) -> str:
    """Return the input message tagged with this server's label."""
    return f"{label}:{message}"


if __name__ == "__main__":
    mcp.run()
