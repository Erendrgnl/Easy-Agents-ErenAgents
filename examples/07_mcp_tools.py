"""07 — MCP tools (plus a custom tool) in one agent.

``Agent.from_mcp`` loads tools from one or more MCP servers and merges them with
any custom Python tools. This example is self-contained: it launches the local
stdio server ``mcp_math_server.py`` as a subprocess, so no external service or
key is needed for the tools themselves (you still need an LLM key).

Run:  python examples/07_mcp_tools.py
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

from erenagents import Agent

load_dotenv()

MODEL = "openai:gpt-4o-mini"
MATH_SERVER = str(Path(__file__).parent / "mcp_math_server.py")


@tool
def greet(name: str) -> str:
    """Return a greeting for the given name."""
    return f"Hello, {name}!"


async def main() -> None:
    # stdio transport: the server is run as a subprocess with this same Python.
    mcp_servers = {
        "math": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [MATH_SERVER],
        }
    }

    agent = await Agent.from_mcp(
        mcp_servers,
        model=MODEL,
        tools=[greet],          # custom tool combined with the MCP tools
        temperature=0,
    )

    print("Loaded tools:", [t.name for t in agent.tools])

    result = await agent.ainvoke(
        "First greet Eren, then add 21 and 21 and multiply the result by 2."
    )
    print("\nreply:", result["content"])
    print("usage:", result.get("usage"))


if __name__ == "__main__":
    asyncio.run(main())
