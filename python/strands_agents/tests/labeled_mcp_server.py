import sys

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

# The first argument labels this server so a test can tell which one answered.
label = sys.argv[1]

mcp = FastMCP(f"labeled-{label}")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def echo(message: str) -> str:
    """Return the input message tagged with this server's label."""
    return f"{label}:{message}"


if __name__ == "__main__":
    mcp.run()
