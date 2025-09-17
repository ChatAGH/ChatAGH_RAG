from langgraph.graph.state import StateGraph

from chat_agh.states import RetrievalState
from chat_agh.agents.retrieval import (
    SimilaritySearch,
    ContextRetrieval,
    SummaryGeneration
)


class RetrievalAgent:
    def __init__(
        self,
        agent_name: str,
        index_name: str,
        description: str,
        num_retrieved_chunks: int = 8,
        num_context_chunks: int = 3,
        window_size: int = 1
    ):
        self.name = agent_name
        self.index_name = index_name
        self.description = description
        self.graph = (
            StateGraph(RetrievalState)
            .add_node("similarity_search", SimilaritySearch(
                index_name=self.index_name,
                num_retrieved_chunks=num_retrieved_chunks,
                window_size=window_size
            ))
            .add_node("context_retrieval", ContextRetrieval(num_chunks=num_context_chunks))
            .add_node("summary_generation", SummaryGeneration())
            .add_edge("similarity_search", "context_retrieval")
            .add_edge("context_retrieval", "summary_generation")
            .set_entry_point("similarity_search")
            .compile()
        )

    def query(self, query: str):
        initial_state = RetrievalState(query=query)
        return self.graph.invoke(initial_state)["summary"] # type: ignore[arg-type]
