"""13 — Customizing an erenagents Agent's own graph.

Every ``Agent`` is a LangGraph graph under the hood. You can:

  * inspect it   -> ``agent.get_graph()`` (compiled) / ``agent.get_workflow()``
  * customize it -> subclass ``Agent`` and override ``_build_workflow``, reusing
    the built-in ``_agent_node`` while adding your own nodes and edges.

This wraps the standard LLM node with a custom pre-processing node (runs before
the model) and a post-processing node (runs after), all sharing the same message
state. The public API (``ainvoke``/``astream``/memory) keeps working unchanged.

Note: this simple override assumes no tools. If you pass ``tools=[...]``, also
re-add the tool node/edges (see ``Agent._build_workflow`` for the pattern).

Run:  python examples/13_custom_graph.py
"""

import asyncio

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, StateGraph

from erenagents import Agent
from erenagents.agent import AgentState  # the shared {"messages": [...]} state


load_dotenv()

MODEL = "openai:gpt-4o-mini"


class WrappedAgent(Agent):
    """Graph: preprocess -> agent -> postprocess -> END."""

    def _build_workflow(self):
        wf = StateGraph(AgentState)
        wf.add_node("preprocess", self._preprocess)
        wf.add_node("agent", self._agent_node)       # reuse the built-in LLM node
        wf.add_node("postprocess", self._postprocess)

        wf.set_entry_point("preprocess")
        wf.add_edge("preprocess", "agent")
        wf.add_edge("agent", "postprocess")
        wf.add_edge("postprocess", END)

        self.workflow = wf   # __init__ compiles this for you afterwards

    async def _preprocess(self, state: AgentState) -> dict:
        # Runs before the model — e.g. inject a dynamic instruction.
        print("[preprocess] messages so far:", len(state["messages"]))
        return {"messages": [SystemMessage(content="Answer in exactly one sentence.")]}

    async def _postprocess(self, state: AgentState) -> dict:
        # Runs after the model — e.g. tweak/annotate the answer.
        answer = state["messages"][-1].content
        return {"messages": [AIMessage(content=f"{answer}\n— via WrappedAgent")]}


async def main() -> None:
    agent = WrappedAgent(model=MODEL, temperature=0.3)

    # The customization is real — the compiled graph now has three nodes.
    print("graph nodes:", list(agent.get_graph().get_graph().nodes))

    result = await agent.ainvoke("What is the capital of France?")
    print(result["content"])


if __name__ == "__main__":
    asyncio.run(main())
