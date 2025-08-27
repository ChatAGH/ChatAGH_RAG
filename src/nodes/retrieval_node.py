from src.states import ChatState
from src.utils.agents_info import AgentsInfo, AgentDetails


class RetrievalNode:
    def __init__(self, retrieval_agents: dict):
        self.retrieval_agents = retrieval_agents

    def __call__(self, state: ChatState) -> dict:
        queries = state["agents_queries"]

        responses = []
        for agent_name, query in queries.items():
            if agent := self.retrieval_agents.get(agent_name):
                agent_response = agent.query(query)
                responses.append(
                    AgentDetails(
                        name=agent.name,
                        description=agent.description,
                        cached_history={
                            "query": query,
                            "response": agent_response,
                        }
                    )
                )

        return {"agents_info": AgentsInfo(agents_details=responses)}

