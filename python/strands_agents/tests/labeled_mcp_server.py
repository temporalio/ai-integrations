import sys

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

# The first argument labels this server so a test can tell which one answered.
label = sys.argv[1]

mcp = MCPServer(f"labeled-{label}")


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False))
def echo(message: str) -> str:
    """Return the input message tagged with this server's label."""
    return f"{label}:{message}"


if __name__ == "__main__":
    mcp.run()
