from concurrent.futures import ThreadPoolExecutor, as_completed

from chat_agh.utils.utils import logger, log_execution_time
from chat_agh.agents.retrieval_agent import RetrievalAgent
from chat_agh.states import ChatState
from chat_agh.utils.agents_info import AgentsInfo, AgentDetails, RETRIEVAL_AGENTS


class RetrievalNode:
    def __init__(self):
        self.retrieval_agents = [
            RetrievalAgent(
                agent_name=agent_info.name,
                index_name=agent_info.vector_store_index_name,
                description=agent_info.description,
            ) for agent_info in RETRIEVAL_AGENTS
        ]

    @log_execution_time
    def __call__(self, state: ChatState) -> dict:
        queries = state["agents_queries"]
        responses = []

        def query_agent(agent, query):
            logger.info(f"Querying agent: {agent.name}. Query: {query}")
            agent_response = agent.query(query)
            logger.info(f"Agent response: {agent_response}")
            return AgentDetails(
                name=agent.name,
                description=agent.description,
                cached_history={
                    "query": query,
                    "response": agent_response,
                }
            )

        futures = []
        with ThreadPoolExecutor() as executor:
            for agent_name, query in queries.items():
                for agent in self.retrieval_agents:
                    if agent.name == agent_name:
                        futures.append(executor.submit(query_agent, agent, query))

            for future in as_completed(futures):
                responses.append(future.result())

        return {"agents_info": AgentsInfo(agents_details=responses)}
