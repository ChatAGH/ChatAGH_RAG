from __future__ import annotations

from langchain_core.documents import Document

from chat_agh.vector_store.mongodb import MongoDBVectorStore

from scripts.evaluation_tasks.search_parameters.base import CurrentSearchParameters


class NumCandidatesSearchParameters(CurrentSearchParameters):
    """Experiment with the number of dense candidates before fusion."""

    def __init__(self, *, num_candidates: int | None = None) -> None:
        super().__init__()
        if num_candidates is not None:
            self.num_candidates = num_candidates

    def run(self, vector_store: MongoDBVectorStore, query: str) -> list[Document]:
        return self._search_with_current_parameters(vector_store, query)
