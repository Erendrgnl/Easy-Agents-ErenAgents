import logging
from typing import List, Optional

import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from erenagents import Agent
from erenagents.a2a.executor import ErenAgentExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_agent_card(
    agent: Agent,
    *,
    name: str = "ErenAgents",
    description: str = "An ErenAgents agent.",
    version: str = "1.0.0",
    url: str = "http://localhost:9999/",
    skills: Optional[List[AgentSkill]] = None,
    streaming: bool = False,
    push_notifications: bool = True,
) -> AgentCard:
    """Build a minimal A2A ``AgentCard`` from an agent.

    Input/output content types are taken from the agent; skills default to an
    empty list. Override any field as needed for richer discovery metadata.
    """
    return AgentCard(
        name=name,
        description=description,
        version=version,
        url=url,
        default_input_modes=agent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=agent.SUPPORTED_CONTENT_TYPES,
        capabilities=AgentCapabilities(
            streaming=streaming, push_notifications=push_notifications
        ),
        skills=skills or [],
    )


def serve_a2a(
    agent: Agent,
    *,
    host: str = "localhost",
    port: int = 9999,
    agent_card: Optional[AgentCard] = None,
    name: str = "ErenAgents",
    description: str = "An ErenAgents agent.",
    version: str = "1.0.0",
    skills: Optional[List[AgentSkill]] = None,
):
    """Serve an agent over the A2A protocol at ``http://host:port/``.

    If ``agent_card`` is not provided, a minimal card is built automatically
    from ``name``/``description``/``skills``. This call blocks (runs uvicorn).
    """
    if agent_card is None:
        agent_card = build_agent_card(
            agent,
            name=name,
            description=description,
            version=version,
            url=f"http://{host}:{port}/",
            skills=skills,
        )

    try:
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client, config_store=push_config_store
        )
        request_handler = DefaultRequestHandler(
            agent_executor=ErenAgentExecutor(agent=agent),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender,
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        logger.info(f"Serving A2A agent '{agent_card.name}' at http://{host}:{port}/")
        uvicorn.run(server.build(), host=host, port=port)
    except Exception as e:
        logger.error(f"Error serving A2A: {e}")
        raise e
