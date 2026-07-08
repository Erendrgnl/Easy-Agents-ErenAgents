# ErenAgents

A LangChain / LangGraph based library for building agents fast.

🇹🇷 [Türkçe sürüm](README.tr.md) · 📚 Reference: [English](docs/REFERENCE.en.md) · [Türkçe](docs/REFERENCE.tr.md)

- **Multi-provider**: OpenAI, Anthropic, Google (Gemini) and OpenAI-compatible endpoints (vLLM, local servers).
- **Tools**: custom LangChain tools and MCP servers (stdio/HTTP) together.
- **Streaming**: token-level and step-level streaming (sync + async).
- **Structured output**: typed output via a Pydantic schema (OpenAI, Anthropic, Google).
- **Observability**: token usage in the response; pluggable callbacks for traces + cost (Langfuse).
- **A2A**: serve any agent as an HTTP endpoint in one line and reach it by URL.

## Installation

```bash
git clone https://github.com/Erendrgnl/Easy-Agents-ErenAgents.git
cd Easy-Agents-ErenAgents

# with uv (recommended)
uv venv && uv pip install -e ".[all]"

# or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"        # all providers
# or a single provider: pip install -e ".[openai]"
```

Set provider credentials via environment variables (or a `.env` file in your
project root), e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`.

## Quick start

```python
import asyncio
from erenagents import Agent

async def main():
    agent = Agent(model="openai:gpt-4o", temperature=0.1)
    result = await agent.ainvoke("Hello!")     # a plain string is enough
    print(result["content"])

asyncio.run(main())
```

Input is flexible: a `str`, a single `BaseMessage`, a list of those, or a
`{"messages": [...]}` dict. For synchronous use, call `agent.invoke(...)`.

### Streaming

```python
# token-level (chatbot UX)
async for token in agent.astream("Tell me a story"):
    print(token, end="", flush=True)

# step-level (tool calls, intermediate steps)
async for step in agent.astream_steps("What's the weather?"):
    print(step["node"], step["update"])

# synchronous token streaming
for token in agent.stream("Tell me a story"):
    print(token, end="", flush=True)
```

> Note: token-level streaming is not possible with `output_schema`; in that case
> `astream` yields the full result once.

### Tools

```python
from langchain_core.tools import tool
from erenagents import Agent

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

agent = Agent(model="openai:gpt-4o", tools=[add])
```

### MCP (Model Context Protocol) tools

MCP server tools can be combined with custom Python tools. The `Agent.from_mcp`
async factory loads the MCP tools and merges them:

```python
import sys
from erenagents import Agent

mcp_servers = {
    # stdio: runs the server as a subprocess
    "math": {"transport": "stdio", "command": sys.executable, "args": ["math_server.py"]},
    # or a remote MCP server over HTTP
    # "serpapi": {"transport": "streamable_http", "url": "http://localhost:8000/mcp"},
}

agent = await Agent.from_mcp(
    mcp_servers,
    model="openai:gpt-4o",
    tools=[my_custom_tool],   # custom tools + MCP tools together
)

# add MCP tools to an existing agent later:
await agent.add_mcp_tools(mcp_servers)
```

To only load tools: `from erenagents.tools import load_mcp_tools`.

### Structured (Pydantic) output

```python
from pydantic import BaseModel
from erenagents import Agent

class BookReview(BaseModel):
    name: str
    rating: int

agent = Agent(model="anthropic:claude-haiku-4-5-20251001", output_schema=BookReview)
result = await agent.ainvoke("Give 'Crime and Punishment' a rating of 9")
print(result["parsed"])          # BookReview(name='Crime and Punishment', rating=9)
print(result["parsed"].rating)   # typed access
```

### vLLM / OpenAI-compatible endpoint

```python
agent = Agent(
    model="vllm:meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
    # if you omit api_key, "EMPTY" is used automatically for vLLM
)
```

### Observability (tokens, cost, traces)

Every `ainvoke`/`invoke` call returns token usage when the provider reports it:

```python
result = await agent.ainvoke("Hello")
print(result["usage"])   # {'input_tokens': 11, 'output_tokens': 43, 'total_tokens': 54}
```

For full traces and **automatic cost calculation**, plug in Langfuse (or any
LangChain callback). Langfuse computes cost from its own model price catalog:

```bash
pip install "erenagents[langfuse]"
export LANGFUSE_PUBLIC_KEY=...  LANGFUSE_SECRET_KEY=...  LANGFUSE_HOST=...
```

```python
# easy path — named backend
agent = Agent(model="openai:gpt-4o", observe="langfuse")

# or any LangChain callback handler
from erenagents.observability import get_langfuse_handler
agent = Agent(model="openai:gpt-4o", callbacks=[get_langfuse_handler()])
```

Callbacks apply to the whole graph (LLM calls, tools, steps); `thread_id` shows
up as a session in Langfuse. This also works through A2A automatically.

### Serve over A2A and reach it

Serve an agent at a URL in one line (the AgentCard is built automatically):

```python
from erenagents import Agent

agent = Agent(model="openai:gpt-4o")
agent.serve(host="localhost", port=9999, name="Greeting Agent")
# -> http://localhost:9999/  (agent card: /.well-known/agent-card.json)
```

Reach that agent by URL from another process:

```python
from erenagents.a2a import call_a2a_agent, get_agent_card

card = await get_agent_card("http://localhost:9999/")   # discovery / reachability
reply = await call_a2a_agent("http://localhost:9999/", "Hello!")
print(reply)
```

## Model selection

Use the `provider:model` format: `openai:gpt-4o`,
`anthropic:claude-haiku-4-5-20251001`, `google:gemini-2.5-flash`, `vllm:<model>`.
For recognized models the prefix is optional (`gpt-4o`, `claude-...`,
`gemini-...` are resolved automatically). Extra LLM parameters (`temperature`,
`max_tokens`, ...) are passed directly to `Agent(...)`.

## Memory

An in-memory checkpointer is enabled by default; calls with the same `thread_id`
share conversation history:

```python
config = {"configurable": {"thread_id": "user-123"}}
await agent.ainvoke("My name is Eren", config=config)
await agent.ainvoke("What was my name?", config=config)   # "Eren"
```

Pass a persistent checkpointer, or disable it with `Agent(checkpointer=None)`.

## License

GNU AGPL-3.0-or-later — see [LICENSE](LICENSE).
