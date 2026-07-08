"""06 — Structured (typed) output.

Pass a Pydantic model as ``output_schema`` and the agent returns a validated
instance under ``result["parsed"]`` (the JSON text is still in
``result["content"]``).

Run:  python examples/06_structured_output.py
"""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from erenagents import Agent

load_dotenv()

MODEL = "openai:gpt-4o-mini"


class Person(BaseModel):
    """A person extracted from free text."""

    name: str = Field(description="The person's given name")
    age: int = Field(description="Age in years")
    city: str = Field(description="City they live in")


async def main() -> None:
    agent = Agent(model=MODEL, output_schema=Person, temperature=0)

    result = await agent.ainvoke("Eren is 30 years old and lives in Istanbul.")

    person: Person = result["parsed"]        # a validated Person instance
    print("parsed  :", person)
    print("name    :", person.name)
    print("age     :", person.age)
    print("raw json:", result["content"])


if __name__ == "__main__":
    asyncio.run(main())
