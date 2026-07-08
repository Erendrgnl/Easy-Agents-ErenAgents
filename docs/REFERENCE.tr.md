# ErenAgents — Referans

ErenAgents SDK için tam kullanım referansı.

🇬🇧 [English](REFERENCE.en.md) · 🏠 [README](../README.tr.md)

---

## İçindekiler

1. [Kurulum](#kurulum)
2. [`Agent` sınıfı](#agent-sınıfı)
3. [Modeller & provider'lar](#modeller--providerlar)
4. [Tool'lar](#toollar)
5. [MCP sunucuları](#mcp-sunucuları)
6. [Structured output](#structured-output)
7. [Streaming](#streaming)
8. [Hafıza](#hafıza)
9. [Observability](#observability)
10. [A2A: serve & ulaşma](#a2a-serve--ulaşma)
11. [Örnekler](#örnekler)

---

## Kurulum

```bash
pip install -e ".[all]"            # tüm provider'lar + langfuse
pip install -e ".[openai]"         # tek provider
pip install -e ".[langfuse]"       # observability extra
pip install -e ".[dev]"            # test araçları
```

Extra'lar: `openai`, `anthropic`, `google`, `langfuse`, `all`, `dev`.

```python
import erenagents
print(erenagents.__version__)
```

---

## `Agent` sınıfı

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

| Parametre | Tip | Açıklama |
|---|---|---|
| `model` | `str` | `provider:model` (örn. `"openai:gpt-4o"`) ya da provider'ı çıkarılabilen düz isim (`"gpt-4o"`). |
| `tools` | `list[BaseTool]` | Agent'ın kullanabileceği LangChain tool'ları. |
| `system_prompt` | `str` | Her çağrıya eklenen sistem prompt'u. |
| `output_schema` | `type[BaseModel]` | Structured output için Pydantic modeli. |
| `provider` | `str` | `model` prefix'siz olduğunda provider override. |
| `base_url` | `str` | OpenAI-uyumlu sunucular için custom endpoint (vLLM). |
| `api_key` | `str` | API anahtarı; vLLM tarzı endpoint'lerde varsayılan `"EMPTY"`. |
| `checkpointer` | saver / `None` | Hafıza backend'i. Varsayılan: in-memory; `None` kapatır. |
| `callbacks` | `list` | Her çalıştırmaya uygulanan LangChain callback'leri. |
| `observe` | `str` | Adlandırılmış observability backend'i, örn. `"langfuse"`. |
| `**llm_kwargs` | | Ek LLM parametreleri (`temperature`, `max_tokens`, ...). |

### Metotlar

| Metot | Açıklama |
|---|---|
| `await ainvoke(input, config=None)` | Asenkron çalıştırır, dict döndürür. |
| `invoke(input, config=None)` | `ainvoke`'un senkron sarmalı. |
| `astream(input, config=None)` | Metin token'ları üreten async generator. |
| `stream(input, config=None)` | Senkron token generator. |
| `astream_steps(input, config=None)` | `{"node", "update"}` adım olayları üreten async generator. |
| `await Agent.from_mcp(mcp_servers, *, tools=None, **kwargs)` | Classmethod: MCP tool'lu agent kurar. |
| `await add_mcp_tools(mcp_servers)` | Mevcut agent'a MCP tool'ları ekler. |
| `update_tools(new_tools)` | Tool'ları değiştirir, workflow'u yeniden kurar. |
| `update_llm(model=None, provider=None, base_url=None, api_key=None, **kw)` | LLM'i değiştirir. |
| `serve(*, host, port, name, description, **card_kwargs)` | A2A üzerinden serve eder (bloklar). |
| `get_graph()` / `get_workflow()` | Derlenmiş graph / `StateGraph` erişimi. |

### Girdi formatları

`ainvoke`/`astream` vb. şunların hepsini kabul eder:

```python
await agent.ainvoke("düz bir string")
await agent.ainvoke(HumanMessage(content="tek bir mesaj"))
await agent.ainvoke(["string", "listesi", AIMessage(content="veya mesajlar")])
await agent.ainvoke({"messages": [HumanMessage(content="ham state dict")]})
```

### Dönüş yapısı

```python
result = await agent.ainvoke("Merhaba")
result["content"]            # str — son yanıt
result.get("usage")          # raporlanırsa {"input_tokens", "output_tokens", "total_tokens"}
result.get("parsed")         # sadece output_schema varsa Pydantic örneği
```

---

## Modeller & provider'lar

Model string'leri `provider:model` formatındadır. Desteklenen provider'lar:
`openai`, `anthropic`, `google`/`gemini`, `vllm` (ve OpenAI-uyumlu alias'lar
`openai-compatible`, `custom`).

```python
Agent(model="openai:gpt-4o")
Agent(model="anthropic:claude-haiku-4-5-20251001")
Agent(model="google:gemini-2.5-flash")

# Düz isimden provider çıkarımı (gpt-* / claude-* / gemini-*):
Agent(model="gpt-4o")

# Açık provider override:
Agent(model="gpt-4o", provider="openai")
```

### vLLM / OpenAI-uyumlu

```python
agent = Agent(
    model="vllm:meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",   # sunucunun /v1 URL'i
    api_key="EMPTY",                        # opsiyonel; vLLM için varsayılan
)
```

### `ModelRouter` (düşük seviye)

```python
from erenagents import ModelRouter

provider, name = ModelRouter.parse_model_string("openai:gpt-4o")   # ("openai", "gpt-4o")
llm, effective_provider = ModelRouter.get_llm("openai:gpt-4o", temperature=0)
```

---

## Tool'lar

Her LangChain tool çalışır. `@tool` dekoratörüyle tanımla:

```python
from langchain_core.tools import tool
from erenagents import Agent

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""   # docstring modele gösterilir
    return a + b

agent = Agent(model="openai:gpt-4o", tools=[add])
```

Agent tool-calling döngüsünü otomatik çalıştırır (tool çağır → sonucu geri besle →
nihai cevaba kadar devam). Çalışma anında `update_tools(...)` ile değiştirilebilir.

---

## MCP sunucuları

MCP sunucularından tool yükle, custom tool'larla birleştir.

```python
import sys
from erenagents import Agent

mcp_servers = {
    "math": {"transport": "stdio", "command": sys.executable, "args": ["math_server.py"]},
    "search": {"transport": "streamable_http", "url": "http://localhost:8000/mcp"},
}

agent = await Agent.from_mcp(mcp_servers, model="openai:gpt-4o", tools=[my_tool])
```

Agent olmadan sadece tool yükle:

```python
from erenagents.tools import load_mcp_tools

tools = await load_mcp_tools(mcp_servers)                       # tüm sunucular
tools = await load_mcp_tools(mcp_servers, server_name="math")   # tek sunucu
```

Connection config'leri `langchain_mcp_adapters` formatındadır (stdio, HTTP, SSE,
websocket). `langchain-mcp-adapters` çekirdek bağımlılık olduğundan extra gerekmez.

---

## Structured output

Bir Pydantic modeli geç; tipli örnek `"parsed"` altında döner.

```python
from pydantic import BaseModel
from erenagents import Agent

class Review(BaseModel):
    name: str
    rating: int

agent = Agent(model="openai:gpt-4o", output_schema=Review)
result = await agent.ainvoke("'Dune'a 8/10 ver")
result["parsed"]          # Review(name='Dune', rating=8)
result["parsed"].rating   # 8
```

Notlar:
- Yöntem provider'a göre seçilir (OpenAI için `json_schema`, diğerleri için `function_calling`).
- `output_schema` + `tools` birlikte şu an yalnızca OpenAI-uyumlu provider'larda desteklenir.
- Structured output token-stream edilemez ([Streaming](#streaming)'e bak).

---

## Streaming

```python
# Token-level — üretildikçe metin parçaları verir.
async for token in agent.astream("Bir haiku yaz"):
    print(token, end="", flush=True)

# Adım-level — her graph node'u için bir olay verir (tool çağrılarını gösterir).
async for step in agent.astream_steps("3 + 5 kaç?"):
    print(step["node"], step["update"])    # örn. "agent" / "tools" / "agent"

# Senkron token streaming.
for token in agent.stream("Bir haiku yaz"):
    print(token, end="", flush=True)
```

`output_schema` ayarlıysa `astream` tam sonucu tek seferde verir (token streaming yok).

---

## Hafıza

Varsayılan olarak thread bazlı in-memory checkpointer açıktır. Aynı `thread_id`'yi
yeniden kullanarak konuşma geçmişini koru:

```python
config = {"configurable": {"thread_id": "user-123"}}
await agent.ainvoke("Adım Eren", config=config)
await agent.ainvoke("Adım neydi?", config=config)   # "Eren"i hatırlar
```

Custom veya kapalı hafıza:

```python
Agent(model="openai:gpt-4o", checkpointer=None)        # stateless
Agent(model="openai:gpt-4o", checkpointer=my_saver)    # örn. Postgres/Redis saver
```

---

## Observability

### Token kullanımı

`ainvoke`/`invoke`, mevcut olduğunda provider-bağımsız token sayılarını döndürür:

```python
result = await agent.ainvoke("Merhaba")
result["usage"]   # {"input_tokens": 11, "output_tokens": 43, "total_tokens": 54}
```

### Callback'ler & Langfuse

Herhangi bir LangChain callback handler'ı takılabilir; tüm graph'ı kapsar
(LLM çağrıları, tool'lar, adımlar). Langfuse ayrıca **cost**'u otomatik hesaplar.

```bash
pip install "erenagents[langfuse]"
# env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
```

```python
# Adlandırılmış backend (en kolayı):
agent = Agent(model="openai:gpt-4o", observe="langfuse")

# Veya handler'ları açıkça geç:
from erenagents.observability import get_langfuse_handler
agent = Agent(model="openai:gpt-4o", callbacks=[get_langfuse_handler()])
```

`thread_id` Langfuse'da session olarak izlenir. Observability A2A üzerinden de akar
(agent'ı `observe="langfuse"` ile serve et).

Yardımcılar: `erenagents.observability`'den `get_langfuse_handler(**kwargs)`,
`get_observer(name, **kwargs)`.

---

## A2A: serve & ulaşma

### Serve

```python
from erenagents import Agent

agent = Agent(model="openai:gpt-4o")
agent.serve(host="localhost", port=9999, name="My Agent", description="...")
# Bloklar. Agent card: http://localhost:9999/.well-known/agent-card.json
```

`erenagents.a2a`'dan düşük seviye yardımcılar:

```python
from erenagents.a2a import serve_a2a, build_agent_card

card = build_agent_card(agent, name="My Agent", url="http://localhost:9999/")
serve_a2a(agent, host="localhost", port=9999, agent_card=card)
```

### Ulaşma

```python
from erenagents.a2a import get_agent_card, call_a2a_agent

card = await get_agent_card("http://localhost:9999/")     # keşif / reachability
reply = await call_a2a_agent("http://localhost:9999/", "Merhaba!")   # -> str
```

`call_a2a_agent(url, message, *, timeout=60.0, agent_card=None)` agent'ın yanıt
metnini döndürür. Önceden çözülmüş bir `agent_card` geçerek keşfi atlayabilirsin.
