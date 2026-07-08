"""A tiny, self-contained MCP server (stdio transport).

`07_mcp_tools.py` launches this automatically as a subprocess; it needs no
external service or API key. You can also run it standalone.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


if __name__ == "__main__":
    mcp.run(transport="stdio")
