"""12 — One agent calling another (agent-as-tool).

``a2a_tool`` wraps a remote A2A agent as a LangChain tool. Give it to a local
agent and its LLM can delegate work to the remote agent.

Prerequisite: start ``10_a2a_server.py`` first (it becomes the remote agent).

Run:  python examples/12_agent_as_tool.py
"""

import asyncio
import sys

import httpx

from dotenv import load_dotenv

from erenagents import Agent
from erenagents.a2a import a2a_tool, get_agent_card

load_dotenv()

MODEL = "openai:gpt-4o-mini"
REMOTE_URL = "http://localhost:9999/"


async def main() -> None:
    # Make sure the remote agent is up.
    try:
        await get_agent_card(REMOTE_URL)
    except httpx.ConnectError:
        print(f"Remote agent not reachable at {REMOTE_URL}. Start 10_a2a_server.py first.")
        sys.exit(1)

    # Wrap the remote agent as a tool the local agent can call.
    greeter = a2a_tool(
        REMOTE_URL,
        name="greeter",
        description="Delegate greetings and small talk to the remote greeting agent.",
    )

    orchestrator = Agent(
        model=MODEL,
        tools=[greeter],
        system_prompt="When the user wants a greeting, use the 'greeter' tool.",
        temperature=0,
    )

    result = await orchestrator.ainvoke("Please greet Eren warmly.")
    print("reply:", result["content"])


if __name__ == "__main__":
    asyncio.run(main())
