"""08 — Providers and OpenAI-compatible (vLLM) endpoints.

Models are named ``provider:model``. You can also swap the model on an existing
agent with ``update_llm``, and point at any OpenAI-compatible server (vLLM,
LM Studio, ...) via ``base_url``.

Run:  python examples/08_providers_and_vllm.py
"""

import asyncio
import os

from dotenv import load_dotenv

from erenagents import Agent

load_dotenv()


async def main() -> None:
    # 1) provider:model strings. Add more once you have the matching keys, e.g.
    #    "anthropic:claude-haiku-4-5-20251001", "google:gemini-2.0-flash".
    for model in ["openai:gpt-4o-mini"]:
        agent = Agent(model=model, temperature=0)
        reply = await agent.ainvoke("Reply with exactly: ok")
        print(f"{model} -> {reply['content']}")

    # 2) Switch the model on an existing agent at runtime.
    agent = Agent(model="openai:gpt-4o-mini", temperature=0)
    agent.update_llm(model="openai:gpt-4o")
    print("after update_llm ->", agent.model_name)

    # 3) OpenAI-compatible server (vLLM, LM Studio, ...). api_key defaults to
    #    "EMPTY" when a base_url is given, which is what vLLM expects.
    base_url = os.getenv("VLLM_BASE_URL")   # e.g. http://localhost:8000/v1
    if base_url:
        local = Agent(model="vllm:my-local-model", base_url=base_url, temperature=0)
        print("vllm ->", (await local.ainvoke("Hello"))["content"])
    else:
        print("Set VLLM_BASE_URL to try an OpenAI-compatible endpoint.")


if __name__ == "__main__":
    asyncio.run(main())
