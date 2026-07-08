# ErenAgents — Reference

Complete usage reference for the ErenAgents SDK.

🇹🇷 [Türkçe](REFERENCE.tr.md) · 🏠 [README](../README.md)

---

## Contents

1. [Installation](#installation)
2. [The `Agent` class](#the-agent-class)
3. [Models & providers](#models--providers)
4. [Tools](#tools)
5. [MCP servers](#mcp-servers)
6. [Structured output](#structured-output)
7. [Streaming](#streaming)
8. [Memory](#memory)
9. [Observability](#observability)
10. [A2A: serving & reaching agents](#a2a-serving--reaching-agents)
11. [Examples](#examples)

---

## Installation

```bash
pip install -e ".[all]"            # all providers + langfuse
pip install -e ".[openai]"         # single provider
pip install -e ".[langfuse]"       # observability extra
pip install -e ".[dev]"            # test tooling
```

Extras: `openai`, `anthropic`, `google`, `langfuse`, `all`, `dev`.

```python
import erenagents
print(erenagents.__version__)
```

---

## The `Agent` class

```python
from erenagents import Agent
```

### Constructor

```python
Agent(
    model="openai:gpt-4o",
    tools=None,
    system_prompt=None,
    output_schema=None,
    provider=None,
    base_url=None,
    api_key=None,
    checkpointer=<in-memory>,
    callbacks=None,
    observe=None,
    **llm_kwargs,
)
```

| Parameter | Type | Description |
|---|---|---|
| `model` | `str` | `provider:model` (e.g. `"openai:gpt-4o"`) or a bare name whose provider is inferred (`"gpt-4o"`). |
| `tools` | `list[BaseTool]` | LangChain tools available to the agent. |
| `system_prompt` | `str` | System prompt prepended to every call. |
| `output_schema` | `type[BaseModel]` | Pydantic model for structured output. |
| `provider` | `str` | Provider override when `model` has no prefix. |
| `base_url` | `str` | Custom endpoint for OpenAI-compatible servers (vLLM). |
| `api_key` | `str` | API key; defaults to `"EMPTY"` for vLLM-style endpoints. |
| `checkpointer` | saver / `None` | Memory backend. Default: in-memory; `None` disables it. |
| `callbacks` | `list` | LangChain callback handlers applied to every run. |
| `observe` | `str` | Named observability backend, e.g. `"langfuse"`. |
| `**llm_kwargs` | | Extra LLM params (`temperature`, `max_tokens`, ...). |

### Methods

| Method | Description |
|---|---|
| `await ainvoke(input, config=None)` | Run asynchronously. Returns a dict. |
| `invoke(input, config=None)` | Synchronous wrapper of `ainvoke`. |
| `astream(input, config=None)` | Async generator of text tokens. |
| `stream(input, config=None)` | Synchronous token generator. |
| `astream_steps(input, config=None)` | Async generator of `{"node", "update"}` step events. |
| `await Agent.from_mcp(mcp_servers, *, tools=None, **kwargs)` | Classmethod: build an agent with MCP tools. |
| `await add_mcp_tools(mcp_servers)` | Load MCP tools into an existing agent. |
| `update_tools(new_tools)` | Replace tools and rebuild the workflow. |
| `update_llm(model=None, provider=None, base_url=None, api_key=None, **kw)` | Swap the LLM. |
| `serve(*, host, port, name, description, **card_kwargs)` | Serve over A2A (blocking). |
| `get_graph()` / `get_workflow()` | Access the compiled graph / `StateGraph`. |

### Input formats

`ainvoke`/`astream`/etc. accept any of:

```python
await agent.ainvoke("a plain string")
await agent.ainvoke(HumanMessage(content="a single message"))
await agent.ainvoke(["a list", "of strings", AIMessage(content="or messages")])
await agent.ainvoke({"messages": [HumanMessage(content="raw state dict")]})
```

### Return shape

```python
result = await agent.ainvoke("Hello")
result["content"]            # str — the final reply
result.get("usage")          # {"input_tokens", "output_tokens", "total_tokens"} if reported
result.get("parsed")         # Pydantic instance, only when output_schema is set
```

---

## Models & providers

Model strings use `provider:model`. Supported providers: `openai`, `anthropic`,
`google`/`gemini`, `vllm` (and OpenAI-compatible aliases `openai-compatible`, `custom`).

```python
Agent(model="openai:gpt-4o")
Agent(model="anthropic:claude-haiku-4-5-20251001")
Agent(model="google:gemini-2.5-flash")

# Provider inferred from a bare name (gpt-* / claude-* / gemini-*):
Agent(model="gpt-4o")

# Explicit provider override:
Agent(model="gpt-4o", provider="openai")
```

### vLLM / OpenAI-compatible

```python
agent = Agent(
    model="vllm:meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",   # the server's /v1 URL
    api_key="EMPTY",                        # optional; default for vLLM
)
```

### `ModelRouter` (low-level)

```python
from erenagents import ModelRouter

provider, name = ModelRouter.parse_model_string("openai:gpt-4o")   # ("openai", "gpt-4o")
llm, effective_provider = ModelRouter.get_llm("openai:gpt-4o", temperature=0)
```

---

## Tools

Any LangChain tool works. Define one with the `@tool` decorator:

```python
from langchain_core.tools import tool
from erenagents import Agent

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""   # the docstring is shown to the model
    return a + b

agent = Agent(model="openai:gpt-4o", tools=[add])
```

The agent runs a tool-calling loop automatically (call tool → feed result back →
continue until a final answer). Replace tools at runtime with `update_tools(...)`.

---

## MCP servers

Load tools from MCP servers and combine them with custom tools.

```python
import sys
from erenagents import Agent

mcp_servers = {
    "math": {"transport": "stdio", "command": sys.executable, "args": ["math_server.py"]},
    "search": {"transport": "streamable_http", "url": "http://localhost:8000/mcp"},
}

agent = await Agent.from_mcp(mcp_servers, model="openai:gpt-4o", tools=[my_tool])
```

Load tools without an agent:

```python
from erenagents.tools import load_mcp_tools

tools = await load_mcp_tools(mcp_servers)              # all servers
tools = await load_mcp_tools(mcp_servers, server_name="math")   # one server
```

Connection configs follow the `langchain_mcp_adapters` format (stdio, HTTP, SSE,
websocket). `langchain-mcp-adapters` is a core dependency, so no extra is needed.

---

## Structured output

Pass a Pydantic model; the typed instance comes back under `"parsed"`.

```python
from pydantic import BaseModel
from erenagents import Agent

class Review(BaseModel):
    name: str
    rating: int

agent = Agent(model="openai:gpt-4o", output_schema=Review)
result = await agent.ainvoke("Rate 'Dune' 8/10")
result["parsed"]          # Review(name='Dune', rating=8)
result["parsed"].rating   # 8
```

Notes:
- The method is chosen per provider (`json_schema` for OpenAI, `function_calling` otherwise).
- Combining `output_schema` with `tools` is currently supported only for
  OpenAI-compatible providers.
- Structured output cannot be token-streamed (see [Streaming](#streaming)).

---

## Streaming

```python
# Token-level — yields text deltas as they are generated.
async for token in agent.astream("Write a haiku"):
    print(token, end="", flush=True)

# Step-level — yields one event per graph node (surfaces tool calls).
async for step in agent.astream_steps("What's 3 + 5?"):
    print(step["node"], step["update"])    # e.g. "agent" / "tools" / "agent"

# Synchronous token streaming.
for token in agent.stream("Write a haiku"):
    print(token, end="", flush=True)
```

When `output_schema` is set, `astream` yields the full result once (no token streaming).

---

## Memory

A thread-based in-memory checkpointer is enabled by default. Reuse a `thread_id`
to keep conversation history:

```python
config = {"configurable": {"thread_id": "user-123"}}
await agent.ainvoke("My name is Eren", config=config)
await agent.ainvoke("What was my name?", config=config)   # remembers "Eren"
```

Custom or disabled memory:

```python
Agent(model="openai:gpt-4o", checkpointer=None)        # stateless
Agent(model="openai:gpt-4o", checkpointer=my_saver)    # e.g. Postgres/Redis saver
```

---

## Observability

### Token usage

`ainvoke`/`invoke` return provider-agnostic token counts when available:

```python
result = await agent.ainvoke("Hello")
result["usage"]   # {"input_tokens": 11, "output_tokens": 43, "total_tokens": 54}
```

### Callbacks & Langfuse

Any LangChain callback handler can be attached; it covers the whole graph
(LLM calls, tools, steps). Langfuse additionally computes **cost** automatically.

```bash
pip install "erenagents[langfuse]"
# env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
```

```python
# Named backend (easiest):
agent = Agent(model="openai:gpt-4o", observe="langfuse")

# Or pass handlers explicitly:
from erenagents.observability import get_langfuse_handler
agent = Agent(model="openai:gpt-4o", callbacks=[get_langfuse_handler()])
```

`thread_id` is tracked as a Langfuse session. Observability flows through A2A too
(serve the agent with `observe="langfuse"`).

Helpers: `get_langfuse_handler(**kwargs)`, `get_observer(name, **kwargs)` from
`erenagents.observability`.

---

## A2A: serving & reaching agents

### Serve

```python
from erenagents import Agent

agent = Agent(model="openai:gpt-4o")
agent.serve(host="localhost", port=9999, name="My Agent", description="...")
# Blocking. Agent card: http://localhost:9999/.well-known/agent-card.json
```

Lower-level helpers from `erenagents.a2a`:

```python
from erenagents.a2a import serve_a2a, build_agent_card

card = build_agent_card(agent, name="My Agent", url="http://localhost:9999/")
serve_a2a(agent, host="localhost", port=9999, agent_card=card)
```

### Reach

```python
from erenagents.a2a import get_agent_card, call_a2a_agent

card = await get_agent_card("http://localhost:9999/")     # discovery / reachability
reply = await call_a2a_agent("http://localhost:9999/", "Hello!")   # -> str
```

`call_a2a_agent(url, message, *, timeout=60.0, agent_card=None)` returns the
agent's reply text. Pass a pre-resolved `agent_card` to skip discovery.
