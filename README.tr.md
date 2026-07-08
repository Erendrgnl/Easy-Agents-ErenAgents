# ErenAgents

LangChain / LangGraph tabanlı, hızlı agent geliştirmek için bir kütüphane.

🇬🇧 [English version](README.md) · 📚 Referans: [English](docs/REFERENCE.en.md) · [Türkçe](docs/REFERENCE.tr.md)

- **Çok-provider**: OpenAI, Anthropic, Google (Gemini) ve OpenAI-uyumlu endpoint'ler (vLLM, yerel sunucular).
- **Tool desteği**: custom LangChain tool'ları ve MCP sunucuları (stdio/HTTP) birlikte.
- **Streaming**: token-level ve adım-level streaming (sync + async).
- **Structured output**: Pydantic şema ile tipli çıktı (OpenAI, Anthropic, Google).
- **Observability**: token kullanımı response'ta; trace + cost için pluggable callback'ler (Langfuse).
- **A2A**: her agent tek satırda bir HTTP endpoint olarak servis edilir ve URL ile ulaşılır.

## Kurulum

```bash
git clone https://github.com/Erendrgnl/Easy-Agents-ErenAgents.git
cd Easy-Agents-ErenAgents

# uv ile (önerilen)
uv venv && uv pip install -e ".[all]"

# veya pip ile
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"        # tüm provider'lar
# veya tek provider: pip install -e ".[openai]"
```

Provider anahtarlarını ortam değişkeni (veya proje kökünde bir `.env` dosyası) ile
verin, örn. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`.

## Hızlı başlangıç

```python
import asyncio
from erenagents import Agent

async def main():
    agent = Agent(model="openai:gpt-4o", temperature=0.1)
    result = await agent.ainvoke("Merhaba!")     # düz string yeterli
    print(result["content"])

asyncio.run(main())
```

Girdi esnektir: `str`, tek bir `BaseMessage`, bunların listesi veya
`{"messages": [...]}` dict'i kabul edilir. Senkron kullanım için `agent.invoke(...)`.

### Streaming

```python
# token-level (chatbot UX)
async for token in agent.astream("Bir hikaye anlat"):
    print(token, end="", flush=True)

# adım-level (tool çağrıları, ara adımlar)
async for step in agent.astream_steps("Hava nasıl?"):
    print(step["node"], step["update"])

# senkron token streaming
for token in agent.stream("Bir hikaye anlat"):
    print(token, end="", flush=True)
```

> Not: `output_schema` ile token-level streaming yapılamaz; bu durumda `astream`
> tam sonucu tek seferde verir.

### Tool'lar

```python
from langchain_core.tools import tool
from erenagents import Agent

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

agent = Agent(model="openai:gpt-4o", tools=[add])
```

### MCP (Model Context Protocol) tool'ları

MCP sunucu tool'ları custom Python tool'larıyla birleştirilebilir. `Agent.from_mcp`
async factory'si MCP tool'larını yükleyip birleştirir:

```python
import sys
from erenagents import Agent

mcp_servers = {
    # stdio: sunucuyu alt-süreç olarak çalıştırır
    "math": {"transport": "stdio", "command": sys.executable, "args": ["math_server.py"]},
    # veya HTTP üzerinden uzak bir MCP sunucusu
    # "serpapi": {"transport": "streamable_http", "url": "http://localhost:8000/mcp"},
}

agent = await Agent.from_mcp(
    mcp_servers,
    model="openai:gpt-4o",
    tools=[my_custom_tool],   # custom tool'lar + MCP tool'ları birlikte
)

# mevcut bir agent'a sonradan MCP tool eklemek:
await agent.add_mcp_tools(mcp_servers)
```

Sadece tool yüklemek için: `from erenagents.tools import load_mcp_tools`.

### Structured (Pydantic) çıktı

```python
from pydantic import BaseModel
from erenagents import Agent

class BookReview(BaseModel):
    name: str
    rating: int

agent = Agent(model="anthropic:claude-haiku-4-5-20251001", output_schema=BookReview)
result = await agent.ainvoke("'Suç ve Ceza'ya 9 puan ver")
print(result["parsed"])          # BookReview(name='Suç ve Ceza', rating=9)
print(result["parsed"].rating)   # tipli erişim
```

### vLLM / OpenAI-uyumlu endpoint

```python
agent = Agent(
    model="vllm:meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
    # api_key vermezsen vLLM için otomatik "EMPTY" kullanılır
)
```

### Observability (token, cost, trace)

Her `ainvoke`/`invoke` çağrısı, provider raporladığında token kullanımını döndürür:

```python
result = await agent.ainvoke("Merhaba")
print(result["usage"])   # {'input_tokens': 11, 'output_tokens': 43, 'total_tokens': 54}
```

Tam trace ve **otomatik cost hesabı** için Langfuse (veya herhangi bir LangChain
callback'i) takılabilir. Langfuse maliyeti kendi model fiyat kataloğundan hesaplar:

```bash
pip install "erenagents[langfuse]"
export LANGFUSE_PUBLIC_KEY=...  LANGFUSE_SECRET_KEY=...  LANGFUSE_HOST=...
```

```python
# kolay yol — adlandırılmış backend
agent = Agent(model="openai:gpt-4o", observe="langfuse")

# veya herhangi bir LangChain callback handler'ı
from erenagents.observability import get_langfuse_handler
agent = Agent(model="openai:gpt-4o", callbacks=[get_langfuse_handler()])
```

Callback'ler tüm graph'a (LLM çağrıları, tool'lar, adımlar) uygulanır; `thread_id`
Langfuse'da session olarak izlenir. Bu, A2A üzerinden de otomatik çalışır.

### A2A ile serve etme ve ulaşma

Bir agent'ı tek satırda URL'de serve et (AgentCard otomatik kurulur):

```python
from erenagents import Agent

agent = Agent(model="openai:gpt-4o")
agent.serve(host="localhost", port=9999, name="Greeting Agent")
# -> http://localhost:9999/  (agent card: /.well-known/agent-card.json)
```

Başka bir process'ten o agent'a URL ile ulaş:

```python
from erenagents.a2a import call_a2a_agent, get_agent_card

card = await get_agent_card("http://localhost:9999/")   # keşif / reachability
reply = await call_a2a_agent("http://localhost:9999/", "Merhaba!")
print(reply)
```

## Model seçimi

`provider:model` formatı kullanılır: `openai:gpt-4o`,
`anthropic:claude-haiku-4-5-20251001`, `google:gemini-2.5-flash`, `vllm:<model>`.
Tanınan modeller için prefix opsiyoneldir (`gpt-4o`, `claude-...`, `gemini-...`
otomatik çözülür). Ek LLM parametreleri (`temperature`, `max_tokens`, ...)
`Agent(...)`'a doğrudan geçilir.

## Hafıza (memory)

Varsayılan olarak in-memory bir checkpointer açıktır; aynı `thread_id` ile yapılan
çağrılar konuşma geçmişini paylaşır:

```python
config = {"configurable": {"thread_id": "user-123"}}
await agent.ainvoke("Adım Eren", config=config)
await agent.ainvoke("Adım neydi?", config=config)   # "Eren"
```

Kalıcı bir checkpointer geçebilir veya `Agent(checkpointer=None)` ile kapatabilirsin.

## Lisans

GNU AGPL-3.0-or-later — bkz. [LICENSE](LICENSE).
