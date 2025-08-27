from src.states import ChatState
from src.agents import SupervisorAgent
from src.utils.utils import logger, log_execution_time, retry_on_exception


class SupervisorNode:
    def __init__(self):
        self.agent = SupervisorAgent()

    @retry_on_exception(attempts=2, delay=1, backoff=3)
    @log_execution_time
    def __call__(self, state: ChatState) -> dict:
        agent_response = self.agent.invoke(
            agents_info=state["agents_info"],
            chat_history=state["chat_history"],
        )
        logger.info(f"Retrieval decision: {agent_response.retrieval_decision}")
        if agent_response.retrieval_decision:
            logger.info(f"Queries: {agent_response.queries}")
        else:
            logger.info(f"Response: {agent_response.message}")

        return {
            "retrieval_decision": agent_response.retrieval_decision,
            "agents_queries": agent_response.queries,
        }
