"""02 — System prompt and LLM parameters.

A ``system_prompt`` steers the agent's behaviour on every call. Extra keyword
arguments (``temperature``, ``max_tokens``, ...) are forwarded straight to the
underlying model.

Run:  python examples/02_system_prompt_and_params.py
"""

import asyncio

from dotenv import load_dotenv

from erenagents import Agent

load_dotenv()

MODEL = "openai:gpt-4o-mini"


async def main() -> None:
    agent = Agent(
        model=MODEL,
        system_prompt="You are a helpful pirate. Answer in pirate slang, briefly.",
        temperature=0.9,   # higher = more creative
        max_tokens=120,    # cap the response length
    )

    result = await agent.ainvoke("Explain what an API is.")
    print(result["content"])


if __name__ == "__main__":
    asyncio.run(main())
