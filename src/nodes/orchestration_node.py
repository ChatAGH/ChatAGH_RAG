from src.state import ChatState
from src.agents import OrchestrationAgent


class OrchestrationNode:
    def __init__(self):
        self.agent = OrchestrationAgent()

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
