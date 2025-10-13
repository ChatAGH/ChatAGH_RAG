from typing import Dict, Optional

from chat_agh.agents import SupervisorAgent
from chat_agh.states import ChatState
from chat_agh.utils.utils import log_execution_time, logger, retry_on_exception


class SupervisorNode:
    def __init__(self) -> None:
        self.agent = SupervisorAgent()

    @retry_on_exception(attempts=2, delay=1, backoff=3)
    @log_execution_time
    def __call__(self, state: ChatState) -> Dict[str, Optional[Dict[str, str]] | bool]:
        agent_response = self.agent.invoke(
            agents_info=state["agents_info"],
            chat_history=state["chat_history"],
            context=state["context"],
        )

        logger.info(f"Retrieval decision: {agent_response.retrieval_decision}")
        if agent_response.retrieval_decision:
            logger.info(f"Queries: {agent_response.queries}")

        return {
            "retrieval_decision": agent_response.retrieval_decision,
            "agents_queries": agent_response.queries,
        }
