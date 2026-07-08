"""Client helpers for reaching agents served over the A2A protocol."""

import logging
import re
from typing import Any, Dict, Optional
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient, create_text_message_object
from a2a.types import AgentCard, MessageSendParams, SendMessageRequest
from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)


async def get_agent_card(url: str, *, timeout: float = 30.0) -> AgentCard:
    """Resolve and return the ``AgentCard`` published at an A2A endpoint."""
    async with httpx.AsyncClient(timeout=timeout) as http:
        resolver = A2ACardResolver(httpx_client=http, base_url=url)
        return await resolver.get_agent_card()


def _text_from_parts(parts) -> str:
    texts = []
    for part in parts or []:
        root = getattr(part, "root", part)
        text = getattr(root, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts)


def _extract_text(response) -> str:
    """Pull the agent's reply text out of a SendMessageResponse (Task or Message)."""
    root = getattr(response, "root", response)
    result = getattr(root, "result", None)
    if result is None:
        error = getattr(root, "error", None)
        raise RuntimeError(f"A2A agent returned an error: {error}")

    # Direct Message result.
    parts = getattr(result, "parts", None)
    if parts:
        return _text_from_parts(parts)

    # Task result: prefer the status message, then artifacts.
    status = getattr(result, "status", None)
    message = getattr(status, "message", None) if status else None
    if message is not None:
        text = _text_from_parts(getattr(message, "parts", None))
        if text:
            return text

    artifact_parts = []
    for artifact in getattr(result, "artifacts", None) or []:
        artifact_parts.extend(getattr(artifact, "parts", None) or [])
    return _text_from_parts(artifact_parts)


async def call_a2a_agent(
    url: str,
    message: str,
    *,
    timeout: float = 60.0,
    agent_card: Optional[AgentCard] = None,
) -> str:
    """Send a message to an A2A agent at ``url`` and return its reply text.

    Args:
        url: Base URL of the served agent (e.g. ``"http://localhost:9999/"``).
        message: The user message to send.
        timeout: HTTP timeout in seconds.
        agent_card: Pre-resolved card (skips the discovery round-trip).

    Returns:
        The agent's reply as plain text.
    """
    async with httpx.AsyncClient(timeout=timeout) as http:
        if agent_card is None:
            resolver = A2ACardResolver(httpx_client=http, base_url=url)
            agent_card = await resolver.get_agent_card()

        client = A2AClient(httpx_client=http, agent_card=agent_card)
        request = SendMessageRequest(
            id=uuid4().hex,
            params=MessageSendParams(message=create_text_message_object(content=message)),
        )
        response = await client.send_message(request)
        return _extract_text(response)


def _extract_context_id(response) -> Optional[str]:
    """Pull the conversation context id from a SendMessageResponse, if any."""
    root = getattr(response, "root", response)
    result = getattr(root, "result", None)
    return getattr(result, "context_id", None) if result is not None else None


async def chat_a2a_agent(
    url: str,
    message: str,
    *,
    context_id: Optional[str] = None,
    timeout: float = 60.0,
    agent_card: Optional[AgentCard] = None,
) -> Dict[str, Any]:
    """Send a message as part of a multi-turn conversation.

    Reusing the returned ``context_id`` on the next call keeps the agent's
    memory (the server maps it to a LangGraph ``thread_id``).

    Returns:
        ``{"text": str, "context_id": str | None}``.
    """
    async with httpx.AsyncClient(timeout=timeout) as http:
        if agent_card is None:
            resolver = A2ACardResolver(httpx_client=http, base_url=url)
            agent_card = await resolver.get_agent_card()

        client = A2AClient(httpx_client=http, agent_card=agent_card)
        msg = create_text_message_object(content=message)
        if context_id:
            msg.context_id = context_id  # continue the same conversation
        request = SendMessageRequest(id=uuid4().hex, params=MessageSendParams(message=msg))
        response = await client.send_message(request)
        return {
            "text": _extract_text(response),
            "context_id": _extract_context_id(response) or context_id,
        }


def _sanitize_tool_name(name: str) -> str:
    """Make a valid tool name (letters, digits, underscores)."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return cleaned or "remote_agent"


def a2a_tool(
    url: str,
    *,
    name: str,
    description: str = "",
    timeout: float = 60.0,
) -> BaseTool:
    """Wrap a remote A2A agent as a LangChain tool.

    Lets one agent call another: give the returned tool to an ``Agent`` and its
    LLM can delegate a message to the remote agent at ``url``.

    Args:
        url: Base URL of the remote A2A agent.
        name: Tool name (sanitized to letters/digits/underscores).
        description: Tells the calling LLM when to use this agent.
        timeout: Per-call HTTP timeout.
    """
    tool_name = _sanitize_tool_name(name)
    tool_desc = description or f"Delegate a request to the '{name}' agent and return its reply."

    async def _delegate(message: str) -> str:
        result = await chat_a2a_agent(url, message, timeout=timeout)
        return result["text"]

    return StructuredTool.from_function(
        coroutine=_delegate,
        name=tool_name,
        description=tool_desc,
    )
