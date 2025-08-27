from dataclasses import dataclass

from src.agents import retrieval_agent
from src.agents.retrieval_agent import RetrievalAgent
from src.states import ChatState
from src.utils.agents_info import AgentsInfo, AgentDetails


@dataclass
class RetrievalAgentInfo:
    name: str
    vector_store_index_name: str
    description: str

RETRIEVAL_AGENTS = [
    RetrievalAgentInfo(
        name="recrutation_agent",
        vector_store_index_name="chunks",
        description="Agent retrieving information about recrutation on AGH University"
    )
]


class RetrievalNode:
    def __init__(self):
        self.retrieval_agents = [
            RetrievalAgent(
                agent_name=agent_info.name,
                index_name=agent_info.vector_store_index_name,
                description=agent_info.description,
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

