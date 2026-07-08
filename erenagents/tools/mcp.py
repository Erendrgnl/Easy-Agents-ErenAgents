"""MCP (Model Context Protocol) tool loading.

Thin wrapper over ``langchain_mcp_adapters`` to turn MCP server configs into
LangChain tools that can be passed to an :class:`~erenagents.agent.Agent`.
"""

from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool


# A server config maps a server name to its connection settings, e.g.
#   {"math": {"transport": "stdio", "command": "python", "args": ["server.py"]}}
#   {"serpapi": {"transport": "streamable_http", "url": "http://localhost:8000/mcp"}}
MCPServers = Dict[str, Dict[str, Any]]


async def load_mcp_tools(
    servers: MCPServers,
    *,
    server_name: Optional[str] = None,
) -> List[BaseTool]:
    """Load tools from one or more MCP servers.

    Args:
        servers: Mapping of server name to connection config (stdio, HTTP, SSE,
            or websocket), in ``langchain_mcp_adapters`` format.
        server_name: If given, only load tools from that single server.

    Returns:
        A list of LangChain ``BaseTool`` objects ready to pass to an Agent.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(servers)
    return await client.get_tools(server_name=server_name)
