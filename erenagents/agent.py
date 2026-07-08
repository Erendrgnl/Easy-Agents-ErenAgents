import asyncio
import uuid
from pathlib import Path
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypedDict,
    Union,
)

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from .router import ModelRouter

# Load .env file
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Sentinel to distinguish "argument not passed" from an explicit None.
_UNSET = object()

# Input types accepted by the public run methods.
AgentInput = Union[str, BaseMessage, Sequence[Union[str, BaseMessage]], Dict[str, Any]]


class AgentState(TypedDict):
    """Agent state schema - simple message management"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _normalize_input(input_data: AgentInput) -> Dict[str, Any]:
    """Normalize flexible user input into the graph's ``{"messages": [...]}`` shape.

    Accepts a plain string, a single ``BaseMessage``, a list of strings/messages,
    or an already-formed state dict.
    """
    if isinstance(input_data, str):
        return {"messages": [HumanMessage(content=input_data)]}
    if isinstance(input_data, BaseMessage):
        return {"messages": [input_data]}
    if isinstance(input_data, dict):
        return input_data
    if isinstance(input_data, Sequence):
        messages: List[BaseMessage] = []
        for item in input_data:
            if isinstance(item, str):
                messages.append(HumanMessage(content=item))
            elif isinstance(item, BaseMessage):
                messages.append(item)
            else:
                raise TypeError(
                    f"Unsupported message item type: {type(item).__name__}. "
                    "Expected str or BaseMessage."
                )
        return {"messages": messages}
    raise TypeError(
        f"Unsupported input type: {type(input_data).__name__}. "
        "Expected str, BaseMessage, a list of those, or a dict."
    )


class Agent:
    """Simple, modular agent built on LangGraph.

    Supports multiple LLM providers (and OpenAI-compatible servers like vLLM),
    custom tools, optional structured (Pydantic) output, thread-based memory,
    and both sync/async invocation and streaming.
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o",
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        checkpointer: Any = _UNSET,
        callbacks: Optional[List[Any]] = None,
        observe: Optional[str] = None,
        **llm_kwargs,
    ):
        """Initialize the Agent.

        Args:
            model: Model name in ``provider:model`` form (e.g. ``"openai:gpt-4o"``)
                or a bare name whose provider can be inferred (e.g. ``"gpt-4o"``).
            tools: Tools available to the agent.
            system_prompt: Optional system prompt prepended to every call.
            output_schema: Pydantic model for structured output.
            provider: Provider override, used when ``model`` has no prefix.
            base_url: Custom endpoint for OpenAI-compatible servers (e.g. vLLM).
            api_key: API key (defaults to ``"EMPTY"`` for vLLM-style endpoints).
            checkpointer: LangGraph checkpointer. Defaults to an in-memory saver;
                pass ``None`` to disable persistence, or a custom saver.
            callbacks: LangChain callback handlers applied to every run (tracing,
                cost, etc.). Any ``BaseCallbackHandler`` works.
            observe: Named observability backend to enable (e.g. ``"langfuse"``).
                Its callback handler is created and appended to ``callbacks``.
            **llm_kwargs: Extra LLM parameters (``temperature``, ``max_tokens``, ...).
        """
        model_string = self._resolve_model_string(model, provider)

        self.model_name = model_string
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.output_schema = output_schema
        self.base_url = base_url
        self.api_key = api_key
        self.llm = None
        self.SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "text/markdown", "text/html"]
        self.checkpointer = MemorySaver() if checkpointer is _UNSET else checkpointer

        self.callbacks: List[Any] = list(callbacks) if callbacks else []
        if observe:
            from .observability import get_observer
            self.callbacks.append(get_observer(observe))

        if output_schema is not None and not issubclass(output_schema, BaseModel):
            raise TypeError("output_schema has to be a Pydantic BaseModel class")

        # Initialize LLM
        self.llm_core, self.provider_name = ModelRouter.get_llm(
            model=model_string,
            base_url=base_url,
            api_key=api_key,
            **llm_kwargs,
        )

        self._build_llm()
        self._build_workflow()
        self._compile()

    @classmethod
    async def from_mcp(
        cls,
        mcp_servers: Dict[str, Dict[str, Any]],
        *,
        tools: Optional[List[BaseTool]] = None,
        **kwargs,
    ) -> "Agent":
        """Create an Agent with tools loaded from MCP servers.

        MCP tools are fetched asynchronously and merged with any custom
        (Python) tools, so both work together.

        Args:
            mcp_servers: Mapping of server name to connection config
                (stdio/HTTP/SSE), in ``langchain_mcp_adapters`` format.
            tools: Extra custom tools to combine with the MCP tools.
            **kwargs: Forwarded to :class:`Agent` (``model``, ``system_prompt``, ...).

        Example:
            >>> agent = await Agent.from_mcp(
            ...     {"math": {"transport": "stdio", "command": "python",
            ...               "args": ["math_server.py"]}},
            ...     model="openai:gpt-4o",
            ...     tools=[my_custom_tool],
            ... )
        """
        from .tools.mcp import load_mcp_tools

        mcp_tools = await load_mcp_tools(mcp_servers)
        combined = list(tools or []) + mcp_tools
        return cls(tools=combined, **kwargs)

    async def add_mcp_tools(
        self,
        mcp_servers: Dict[str, Dict[str, Any]],
    ) -> List[BaseTool]:
        """Load MCP tools and add them to this agent (rebuilds the workflow).

        Returns the newly loaded MCP tools.
        """
        from .tools.mcp import load_mcp_tools

        mcp_tools = await load_mcp_tools(mcp_servers)
        self.update_tools(list(self.tools) + mcp_tools)
        return mcp_tools

    @staticmethod
    def _resolve_model_string(model: str, provider: Optional[str]) -> str:
        """Combine a model name and optional provider into a ``provider:model`` string."""
        if provider and ":" not in model:
            return f"{provider}:{model}"
        return model

    def _structured_output_method(self) -> str:
        """Pick the structured-output method supported by the active provider."""
        # OpenAI (and OpenAI-compatible) support native JSON schema; others use
        # tool/function calling under the hood.
        return "json_schema" if self.provider_name == "openai" else "function_calling"

    def _build_llm(self):
        """Bind tools and/or structured output to the core LLM."""
        if self.output_schema:
            method = self._structured_output_method()
            kwargs: Dict[str, Any] = {"include_raw": True}
            if method == "json_schema":
                kwargs["strict"] = True
                if self.tools:
                    # OpenAI-compatible providers can mix tools with structured output.
                    kwargs["tools"] = self.tools
            elif self.tools:
                raise NotImplementedError(
                    "Combining tools with output_schema is currently only supported "
                    f"for OpenAI-compatible providers (got provider '{self.provider_name}')."
                )
            self.llm = self.llm_core.with_structured_output(
                self.output_schema, method=method, **kwargs
            )
        elif self.tools:
            self.llm = self.llm_core.bind_tools(self.tools)
        else:
            self.llm = self.llm_core

    def _build_workflow(self):
        """Build the workflow graph."""
        self.workflow = StateGraph(AgentState)
        self.workflow.add_node("agent", self._agent_node)

        if self.tools:
            tool_node = ToolNode(self.tools, handle_tool_errors=True)
            self.workflow.add_node("tools", tool_node)
            self.workflow.add_conditional_edges(
                "agent",
                self._should_continue,
                {"tools": "tools", END: END},
            )
            self.workflow.add_edge("tools", "agent")
        else:
            self.workflow.add_edge("agent", END)

        self.workflow.set_entry_point("agent")

    def _compile(self):
        """Compile the workflow."""
        self.compiled_graph = self.workflow.compile(checkpointer=self.checkpointer)

    async def _agent_node(self, state: AgentState) -> AgentState:
        """Agent node - invoke LLM and return response."""
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + list(state["messages"])
        else:
            messages = state["messages"]
        result = await self.llm.ainvoke(messages)

        # With structured output, result is a dict with 'raw' and 'parsed'.
        if self.output_schema and isinstance(result, dict):
            raw_msg = result.get("raw")
            parsed_output = result.get("parsed")

            if parsed_output:
                structured_response = AIMessage(content=parsed_output.model_dump_json())
                return {"messages": [raw_msg, structured_response]}
            # No parse yet (e.g. tool call) - continue the loop.
            return {"messages": [raw_msg]}

        return {"messages": [result]}

    def _should_continue(self, state: AgentState) -> Union[Literal["tools"], Literal[END]]:
        """Route to the tool node when the last message has tool calls."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    def _prep_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a config copy with ``thread_id`` and callbacks ensured.

        Does not mutate the caller's config. Instance-level callbacks are merged
        with any per-call callbacks.
        """
        config = dict(config) if config else {}
        configurable = dict(config.get("configurable", {}))
        configurable.setdefault("thread_id", str(uuid.uuid4()))
        config["configurable"] = configurable

        if self.callbacks:
            existing = config.get("callbacks") or []
            if not isinstance(existing, list):
                existing = list(existing)
            merged = list(existing)
            for cb in self.callbacks:
                if cb not in merged:
                    merged.append(cb)
            config["callbacks"] = merged
        return config

    @staticmethod
    def _aggregate_usage(messages: Sequence[BaseMessage]) -> Optional[Dict[str, int]]:
        """Sum provider-agnostic token usage across all AI messages in a run."""
        input_tokens = output_tokens = total_tokens = 0
        found = False
        for message in messages:
            usage = getattr(message, "usage_metadata", None)
            if usage:
                found = True
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)
        if not found:
            return None
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _chunk_to_text(message: BaseMessage) -> str:
        """Extract plain text from a streamed message chunk (str or content blocks)."""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return ""

    # ------------------------------------------------------------------ #
    # Async API
    # ------------------------------------------------------------------ #
    async def ainvoke(
        self,
        input_data: AgentInput,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the agent asynchronously.

        Args:
            input_data: A string, message, list of those, or a state dict.
            config: Runtime config (e.g. ``{"configurable": {"thread_id": "..."}}``).

        Returns:
            ``{"content": str}``, plus ``"parsed"`` (the Pydantic instance) when
            an ``output_schema`` is configured, plus ``"usage"`` (token counts)
            when the provider reports them. Cost is tracked via observability
            backends (e.g. Langfuse).
        """
        config = self._prep_config(config)
        graph_input = _normalize_input(input_data)
        result = await self.compiled_graph.ainvoke(graph_input, config=config)

        content = result["messages"][-1].content
        output: Dict[str, Any] = {"content": content}
        if self.output_schema:
            output["parsed"] = self.output_schema.model_validate_json(content)
        usage = self._aggregate_usage(result["messages"])
        if usage is not None:
            output["usage"] = usage
        return output

    async def astream(
        self,
        input_data: AgentInput,
        config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """Stream the agent's answer token by token.

        Yields text deltas as they are produced. When ``output_schema`` is set
        (which cannot be token-streamed), the full content is yielded once.
        """
        if self.output_schema:
            result = await self.ainvoke(input_data, config)
            yield result["content"]
            return

        config = self._prep_config(config)
        graph_input = _normalize_input(input_data)
        async for msg, meta in self.compiled_graph.astream(
            graph_input, config=config, stream_mode="messages"
        ):
            if meta.get("langgraph_node") != "agent":
                continue
            if not isinstance(msg, AIMessageChunk):
                continue
            text = self._chunk_to_text(msg)
            if text:
                yield text

    async def astream_steps(
        self,
        input_data: AgentInput,
        config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream the agent's progress at the step level.

        Yields ``{"node": str, "update": Any}`` after each graph node runs -
        useful for surfacing tool calls and intermediate steps in a UI.
        """
        config = self._prep_config(config)
        graph_input = _normalize_input(input_data)
        async for chunk in self.compiled_graph.astream(
            graph_input, config=config, stream_mode="updates"
        ):
            for node, update in chunk.items():
                yield {"node": node, "update": update}

    # ------------------------------------------------------------------ #
    # Sync API
    # ------------------------------------------------------------------ #
    def invoke(
        self,
        input_data: AgentInput,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper around :meth:`ainvoke`."""
        return asyncio.run(self.ainvoke(input_data, config))

    def stream(
        self,
        input_data: AgentInput,
        config: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """Synchronous wrapper around :meth:`astream` (token streaming)."""
        loop = asyncio.new_event_loop()
        agen = self.astream(input_data, config)
        try:
            while True:
                try:
                    yield loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.run_until_complete(agen.aclose())
            loop.close()

    # ------------------------------------------------------------------ #
    # Mutation / introspection
    # ------------------------------------------------------------------ #
    def update_tools(self, new_tools: List[BaseTool]):
        """Replace the agent's tools and rebuild the workflow."""
        self.tools = new_tools
        self._build_llm()
        self._build_workflow()
        self._compile()

    def update_llm(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **llm_kwargs,
    ):
        """Update LLM settings and rebuild the workflow."""
        if model:
            self.model_name = self._resolve_model_string(model, provider)
        if base_url is not None:
            self.base_url = base_url
        if api_key is not None:
            self.api_key = api_key

        self.llm_core, self.provider_name = ModelRouter.get_llm(
            model=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            **llm_kwargs,
        )
        self._build_llm()
        self._build_workflow()
        self._compile()

    def serve(
        self,
        *,
        host: str = "localhost",
        port: int = 9999,
        name: str = "ErenAgents",
        description: str = "An ErenAgents agent.",
        **card_kwargs,
    ):
        """Serve this agent over the A2A protocol at ``http://host:port/``.

        A minimal AgentCard is built automatically. This call blocks (runs the
        HTTP server). Reach it from elsewhere with
        :func:`erenagents.a2a.call_a2a_agent`.
        """
        from .a2a import serve_a2a

        serve_a2a(
            self,
            host=host,
            port=port,
            name=name,
            description=description,
            **card_kwargs,
        )

    def get_graph(self):
        """Return the compiled graph."""
        return self.compiled_graph

    def get_workflow(self):
        """Return the StateGraph workflow."""
        return self.workflow
