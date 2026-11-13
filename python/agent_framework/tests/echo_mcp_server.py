"""In-process MCP server (MCP Python SDK v2) for the tests."""

from mcp.server.mcpserver import MCPServer

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
    raise RuntimeError(reason)


if __name__ == "__main__":
    server.run()
