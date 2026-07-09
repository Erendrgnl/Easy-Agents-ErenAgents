# ErenAgents — Examples

Numbered, progressive examples that double as a usage reference. Start at `01`
and work down; each one builds on the previous.

## Setup

```bash
# from the repo root, with the package installed (see the main README)
uv pip install -e ".[all]"      # or: pip install -e ".[all]"
```

Create a `.env` file in the repo root with the key(s) for the provider you use:

```dotenv
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=...
# Optional, for example 09:
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
# LANGFUSE_HOST=https://cloud.langfuse.com
```

Every example uses `MODEL = "openai:gpt-4o-mini"` at the top. Swap it for any
provider you have a key for — e.g. `anthropic:claude-haiku-4-5-20251001` or
`google:gemini-2.0-flash` — nothing else changes.

## Run

```bash
python examples/01_hello_agent.py
```

## Index

| # | File | Shows |
|---|------|-------|
| 01 | `01_hello_agent.py` | Smallest agent — async (`ainvoke`) and sync (`invoke`) |
| 02 | `02_system_prompt_and_params.py` | `system_prompt` + LLM params (`temperature`, `max_tokens`) |
| 03 | `03_conversation_memory.py` | Thread-based memory via `thread_id` |
| 04 | `04_streaming.py` | Token streaming (`astream`) + step streaming (`astream_steps`) |
| 05 | `05_custom_tools.py` | Custom Python tools with `@tool` |
| 06 | `06_structured_output.py` | Typed output with a Pydantic `output_schema` |
| 07 | `07_mcp_tools.py` (+ `mcp_math_server.py`) | MCP server tools + a custom tool together |
| 08 | `08_providers_and_vllm.py` | Provider strings, `update_llm`, OpenAI-compatible (vLLM) endpoints |
| 09 | `09_observability.py` | Token `usage` in the result + Langfuse tracing (`observe=`) |
| 10 | `10_a2a_server.py` | Serve an agent over the A2A protocol (one line) |
| 11 | `11_a2a_client.py` | Reach a served agent by URL (single-shot + multi-turn) |
| 12 | `12_agent_as_tool.py` | Let one agent call another via `a2a_tool` |
| 13 | `13_custom_graph.py` | Customize an Agent's own graph (subclass + override `_build_workflow`) |
| 14 | `14_custom_agent_node.py` | Override `_agent_node` to control the LLM step (RAG-style context injection) |

> Examples 07 and 12 launch/need a second process — see the notes at the top of
> each file. Example 09's Langfuse part is skipped unless `LANGFUSE_*` is set.
