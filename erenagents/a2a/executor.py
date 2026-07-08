import logging

from a2a.server.agent_execution import AgentExecutor as BaseAgentExecutor
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    TaskState,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from langchain_core.messages import HumanMessage

from erenagents import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErenAgentExecutor(BaseAgentExecutor):
    def __init__(self, agent: Agent):
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:

        query = context.get_user_input()

        # If the task is not found, create a new task
        task = context.current_task
        if not task:
            logger.info("No existing task found, creating new task")
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
            logger.info(f"Created new task with ID: {task.id}, context ID: {task.context_id}")
        else:
            logger.info(f"Using existing task with ID: {task.id}, context ID: {task.context_id}")

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            # (optional but correct lifecycle)
            await updater.update_status(TaskState.submitted)
            await updater.update_status(TaskState.working)

            logger.info(f"Invoking agent with thread_id: {task.context_id}")
            result = await self.agent.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": task.context_id}},
            )

            text = result["content"]

            logger.info(f"Updating task {task.id} status to completed")
            await updater.update_status(
                TaskState.completed,
                new_agent_text_message(text, task.context_id, task.id),
                final=True,
            )
            logger.info(f"Task {task.id} execution completed successfully")
        except Exception as e:
            logger.exception("Agent ainvoke failed")
            logger.error(f"Task {task.id} execution failed with error: {str(e)}")
            raise ServerError(error=InternalError()) from e

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.warning("Cancel operation requested but not supported")
        raise ServerError(error=UnsupportedOperationError())
