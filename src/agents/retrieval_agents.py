from langgraph.graph.state import StateGraph

from src.states import RetrievalState
from src.agents.retrieval import (
    SimilaritySearch,
    ContextRetrieval,
    SummaryGeneration
)


class BaseRetrievalAgent:
    def __init__(self, agent_name: str, index_name: str):
        self.name = agent_name
        self.index_name = index_name
        self.graph = (
            StateGraph(RetrievalState)
            .add_node("similarity_search", SimilaritySearch(index_name=self.index_name))
            .add_node("context_retrieval", ContextRetrieval())
            .add_node("summary_generation", SummaryGeneration())
            .add_edge("similarity_search", "context_retrieval")
            .add_edge("context_retrieval", "summary_generation")
            .compile()
        )

    def query(self, query: str):
        pass