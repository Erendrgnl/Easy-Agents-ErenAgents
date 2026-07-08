"""05 — Custom tools.

Any LangChain ``@tool`` function can be given to an agent; the LLM decides when
to call them and the agent runs the tool loop for you.

Run:  python examples/05_custom_tools.py
"""

import asyncio

from dotenv import load_dotenv
from langchain_core.tools import tool

from erenagents import Agent

load_dotenv()

MODEL = "openai:gpt-4o-mini"


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"{city}: 24°C, sunny"


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main() -> None:
    agent = Agent(model=MODEL, tools=[get_weather, add], temperature=0)

    result = await agent.ainvoke(
        "What's the weather in Paris, and what is 12 + 30?"
    )
    print("reply:", result["content"])
    print("usage:", result.get("usage"))   # token counts when the provider reports them


if __name__ == "__main__":
    asyncio.run(main())
