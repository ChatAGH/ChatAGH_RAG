from src.agents import retrieval_agent
from src.agents.retrieval_agent import RetrievalAgent
from src.states import ChatState
from src.utils.agents_info import AgentsInfo, AgentDetails


RETRIEVAL_AGENTS = (
    # (
    #     agent_name
    #     vector_store_index_name
    #     description
    # )
    (
        "recrutation_agent",
        "chunks"
        "Agent retrieving information about recrutation on AGH University"
    )
)


class RetrievalNode:
    def __init__(self):
        self.retrieval_agents = [
            RetrievalAgent(
                agent_name=agent_info[0],
                index_name=agent_info[1],
                description=agent_info[2],
            ) for agent_info in RETRIEVAL_AGENTS
        ]

    def __call__(self, state: ChatState) -> dict:
        queries = state["agents_queries"]

        responses = []
        for agent_name, query in queries.items():
            for agent in self.retrieval_agents:
                if agent.name == agent_name:
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

