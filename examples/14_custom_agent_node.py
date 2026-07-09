"""14 — Customizing the LLM step by overriding `_agent_node` (RAG-style).

``_agent_node`` is the graph node that calls the model. Override it to take full
control of that step. Here we inject retrieved context before the model runs — a
minimal Retrieval-Augmented Generation pattern — using a tiny in-memory
knowledge base (swap it for a real vector store).

Contract to keep:
  * return ``{"messages": [...]}`` (the AgentState shape);
  * return the raw AI message so the tool loop still works when tools are bound;
  * ``self.llm`` is the tool/schema-bound model, ``self.llm_core`` the plain one.

Run:  python examples/14_custom_agent_node.py
"""

import asyncio

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

from erenagents import Agent
from erenagents.agent import AgentState


load_dotenv()

MODEL = "openai:gpt-4o-mini"

# A tiny "knowledge base". A real system would query a vector store here.
KNOWLEDGE = {
    "refund": "Refunds are processed within 5 business days to the original payment method.",
    "shipping": "Standard shipping takes 3-5 business days; express takes 1-2.",
    "warranty": "All products include a 2-year limited warranty.",
}


def retrieve(query: str) -> str:
    """Naive keyword retrieval — replace with a real vector search."""
    hits = [text for key, text in KNOWLEDGE.items() if key in query.lower()]
    return "\n".join(hits)


class RagAgent(Agent):
    """Injects retrieved context into the LLM step via ``_agent_node``."""

    async def _agent_node(self, state: AgentState) -> dict:
        query = state["messages"][-1].content

        # Build the prompt: system prompt (if any) + retrieved context + history.
        prefix = []
        if self.system_prompt:
            prefix.append(SystemMessage(content=self.system_prompt))
        context = retrieve(query)
        if context:
            prefix.append(
                SystemMessage(content=f"Use ONLY this context to answer:\n{context}")
            )

        messages = prefix + list(state["messages"])
        result = await self.llm.ainvoke(messages)
        return {"messages": [result]}


async def main() -> None:
    agent = RagAgent(
        model=MODEL,
        system_prompt="You are a concise support assistant.",
        temperature=0,
    )

    for question in ["How long do refunds take?", "What is your warranty policy?"]:
        result = await agent.ainvoke(question)
        print(f"Q: {question}\nA: {result['content']}\n")


if __name__ == "__main__":
    asyncio.run(main())
