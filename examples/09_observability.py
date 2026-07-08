"""09 — Observability: token usage and Langfuse tracing.

Every result carries token ``usage`` when the provider reports it. For full
traces + cost, enable a backend with ``observe="langfuse"`` (needs the
``langfuse`` extra and ``LANGFUSE_*`` env vars). The Langfuse part is skipped
automatically when those keys are absent.

Run:  python examples/09_observability.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_core.tools import tool

from erenagents import Agent

load_dotenv()

MODEL = "openai:gpt-4o-mini"


@tool
def weather(city: str) -> str:
    """Return the weather for a city."""
    return f"{city}: 22°C, sunny."


async def main() -> None:
    # Token usage comes back on the result for free.
    agent = Agent(model=MODEL, tools=[weather], temperature=0)
    result = await agent.ainvoke("What's the weather in Rome? Keep it short.")
    print("reply:", result["content"])
    print("usage:", result.get("usage"))

    # Optional: trace to Langfuse (LLM call, tools, and cost).
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        traced = Agent(model=MODEL, tools=[weather], temperature=0, observe="langfuse")
        session = {"configurable": {"thread_id": "obs-demo"}}
        await traced.ainvoke("And the weather in Oslo?", config=session)

        # Flush buffered events before the process exits.
        from langfuse import get_client
        get_client().flush()
        print("Sent a trace to Langfuse — check the Traces tab.")
    else:
        print("Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY to enable tracing.")


if __name__ == "__main__":
    asyncio.run(main())
