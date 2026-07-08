"""10 — Serve an agent over the A2A protocol.

``agent.serve(...)`` exposes the agent as an HTTP endpoint that other A2A
clients (including other ErenAgents) can reach. The AgentCard is built
automatically. This call blocks — run it, then use ``11_a2a_client.py`` from a
second terminal.

Run:  python examples/10_a2a_server.py
"""

from dotenv import load_dotenv

from erenagents import Agent

load_dotenv()

MODEL = "openai:gpt-4o-mini"


def main() -> None:
    agent = Agent(
        model=MODEL,
        system_prompt="You are a friendly assistant. Keep replies short.",
        temperature=0.3,
    )

    print("Serving agent at http://localhost:9999/  (Ctrl+C to stop)\n")
    agent.serve(
        host="localhost",
        port=9999,
        name="Greeting Agent",
        description="A simple, friendly greeting agent.",
    )


if __name__ == "__main__":
    main()
