from __future__ import annotations

from langchain_core.documents import Document

from chat_agh.vector_store.mongodb import MongoDBVectorStore
from scripts.evaluation_tasks.base import CurrentSearchParameters


class TextWeightSearchParameters(CurrentSearchParameters):
    """Adjust lexical contribution in hybrid scoring."""

    def __init__(self, *, text_weight: float | None = None) -> None:
        super().__init__()
        if text_weight is not None:
            self.text_weight = text_weight

    def run(self, vector_store: MongoDBVectorStore, query: str) -> list[Document]:
        return self._search_with_current_parameters(vector_store, query)
