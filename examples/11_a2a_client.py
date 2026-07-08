"""11 — Reach a served agent by URL.

Talks to the agent from ``10_a2a_server.py``. Start that first (in another
terminal), then run this.

- ``get_agent_card``  — discovery / reachability check.
- ``call_a2a_agent``  — single-shot request/response.
- ``chat_a2a_agent``  — multi-turn: reuse ``context_id`` to keep memory.

Run:  python examples/11_a2a_client.py
"""

import asyncio
import sys

import httpx

from erenagents.a2a import call_a2a_agent, chat_a2a_agent, get_agent_card

URL = "http://localhost:9999/"


async def main() -> None:
    print(f"Connecting to {URL}")
    try:
        card = await get_agent_card(URL)
        print(f"Found agent: {card.name} (v{card.version}) — {card.description}\n")

        # Single-shot call.
        reply = await call_a2a_agent(URL, "Hello! Who are you?")
        print("reply:", reply, "\n")

        # Multi-turn: pass the returned context_id back to keep the conversation.
        first = await chat_a2a_agent(URL, "My name is Eren.")
        second = await chat_a2a_agent(
            URL, "What is my name?", context_id=first["context_id"]
        )
        print("multi-turn:", second["text"])

    except httpx.ConnectError:
        print(f"Could not connect to {URL}. Start 10_a2a_server.py first.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
