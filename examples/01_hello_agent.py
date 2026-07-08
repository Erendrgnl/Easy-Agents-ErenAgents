"""01 — Hello agent: the smallest possible agent.

Shows both the async (`ainvoke`) and sync (`invoke`) entry points. A plain
string is enough input; the reply text is in ``result["content"]``.

Run:  python examples/01_hello_agent.py
"""

import asyncio

from dotenv import load_dotenv

from erenagents import Agent

load_dotenv()

MODEL = "openai:gpt-4o-mini"  # swap for any provider you have a key for


def sync_demo() -> None:
    agent = Agent(model=MODEL, temperature=0.3)
    result = agent.invoke("Say hello in one short sentence.")
    print("sync :", result["content"])


async def async_demo() -> None:
    agent = Agent(model=MODEL, temperature=0.3)
    result = await agent.ainvoke("Now say goodbye in one short sentence.")
    print("async:", result["content"])


if __name__ == "__main__":
    sync_demo()
    asyncio.run(async_demo())
