from src.states import ChatState
from src.agents import SupervisorAgent
from src.utils.utils import logger


class SupervisorNode:
    def __init__(self):
        self.agent = SupervisorAgent()

    def __call__(self, state: ChatState) -> dict:
        agent_response = self.agent.invoke(
            agents_info=state["agents_info"],
            chat_history=state["chat_history"],
        )
        return {
            "retrieval_decision": agent_response.retrieval_decision,
            "agents_queries": agent_response.queries,
            "response": agent_response.message,
        }
