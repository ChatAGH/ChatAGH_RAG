from typing import Any

from langgraph.graph.state import StateGraph

from chat_agh.agents.retrieval import (
    SimilaritySearch,
    SummaryGeneration,
)
from chat_agh.states import RetrievalState


class RetrievalAgent:
    def __init__(
        self,
        agent_name: str,
        index_name: str,
        description: str,
        num_retrieved_chunks: int = 5,
        window_size: int = 0,
    ) -> None:
        self.name = agent_name
        self.index_name = index_name
        self.description = description
        self.graph = (
            StateGraph(RetrievalState)
            .add_node(
                "similarity_search",
                SimilaritySearch(
                    index_name=self.index_name,
                    num_retrieved_chunks=num_retrieved_chunks,
                    window_size=window_size,
                ),
            )
            .add_node("summary_generation", SummaryGeneration())
            .add_edge("similarity_search", "summary_generation")
            .set_entry_point("similarity_search")
            .compile()
        )

    def query(self, query: str) -> Any:
        initial_state = RetrievalState(query=query, retrieved_context=[])
        result = self.graph.invoke(initial_state)

        return {
            "summary": result.get("summary"),
            "retrieved_context": result.get("retrieved_context"),
        }
