from __future__ import annotations

from langchain_core.documents import Document

from chat_agh.vector_store.mongodb import MongoDBVectorStore

from .base import CurrentSearchParameters


class VectorWeightSearchParameters(CurrentSearchParameters):
    """Adjust dense vector contribution in hybrid scoring."""

    def __init__(self, *, vector_weight: float | None = None) -> None:
        super().__init__()
        if vector_weight is not None:
            self.vector_weight = vector_weight

    def run(self, vector_store: MongoDBVectorStore, query: str) -> list[Document]:
        return self._search_with_current_parameters(vector_store, query)
