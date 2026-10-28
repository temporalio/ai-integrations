import os

from mcp.server.fastmcp import FastMCP, Image

mcp = FastMCP("image-server")

# A PNG signature and header chunk: valid bytes, but not valid UTF-8.
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


@mcp.tool()
def screenshot() -> Image:
    """Return a PNG and record that the tool ran."""
    with open(os.environ["IMAGE_MCP_CALLS_FILE"], "a") as calls:
        calls.write("call\n")
    return Image(data=PNG, format="png")


if __name__ == "__main__":
    mcp.run()
