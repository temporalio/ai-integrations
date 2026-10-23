try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # mcp 2 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import (  # type: ignore[no-redef]  # pyright: ignore[reportMissingImports]
        MCPServer as FastMCP,
    )

mcp = FastMCP("echo-server")


@mcp.tool()  # pyright: ignore[reportUntypedFunctionDecorator]
def echo(message: str) -> str:
    """Return the input message unchanged."""
    return message


if __name__ == "__main__":
    mcp.run()
