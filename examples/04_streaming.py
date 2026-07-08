"""04 — Streaming: token level and step level.

- ``astream``       yields text deltas as they are generated (chatbot UX).
- ``astream_steps`` yields ``{"node", "update"}`` after each graph step, which
  makes tool calls and intermediate steps visible (useful in a UI).

Run:  python examples/04_streaming.py
"""

import asyncio

from dotenv import load_dotenv
from langchain_core.tools import tool

from erenagents import Agent

load_dotenv()

MODEL = "openai:gpt-4o-mini"


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def main() -> None:
    agent = Agent(model=MODEL, temperature=0.3)

    print("=== Token streaming ===")
    async for token in agent.astream("Write a 3-sentence paragraph about Istanbul."):
        print(token, end="", flush=True)
    print("\n")

    print("=== Step streaming (with a tool) ===")
    tool_agent = Agent(model=MODEL, tools=[add], temperature=0)
    async for step in tool_agent.astream_steps("What is 21 + 21?"):
        print(f"  node={step['node']}")


if __name__ == "__main__":
    asyncio.run(main())
