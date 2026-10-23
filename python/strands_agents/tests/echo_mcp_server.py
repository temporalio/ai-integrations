import os
import time

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-server")


@mcp.tool()
def echo(message: str) -> str:
    """Return the input message unchanged."""
    return message


if __name__ == "__main__":
    # Lets a test delay the server's answer to initialize.
    time.sleep(float(os.environ.get("ECHO_SERVER_START_DELAY", "0")))
    mcp.run()
