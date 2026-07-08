"""03 — Conversation memory with thread ids.

The agent keeps per-conversation history keyed by ``thread_id``. Reuse the same
id to continue a conversation; use a new id to start fresh. (Memory is in-memory
by default; pass a custom ``checkpointer`` for persistence, or
``checkpointer=None`` to disable it.)

Run:  python examples/03_conversation_memory.py
"""

import asyncio

from dotenv import load_dotenv

from erenagents import Agent

load_dotenv()

MODEL = "openai:gpt-4o-mini"


async def main() -> None:
    agent = Agent(model=MODEL, temperature=0)

    session = {"configurable": {"thread_id": "user-123"}}
    await agent.ainvoke("Hi, my name is Eren.", config=session)
    remembered = await agent.ainvoke("What is my name?", config=session)
    print("same thread   :", remembered["content"])          # -> knows "Eren"

    # A different thread_id shares nothing with the one above.
    other = {"configurable": {"thread_id": "someone-else"}}
    forgotten = await agent.ainvoke("What is my name?", config=other)
    print("other thread  :", forgotten["content"])           # -> does not know


if __name__ == "__main__":
    asyncio.run(main())
