from __future__ import annotations

from langchain_core.documents import Document

from chat_agh.vector_store.mongodb import MongoDBVectorStore
from scripts.evaluation_tasks.search_parameters.base import CurrentSearchParameters


class FuzzySearchParameters(CurrentSearchParameters):
    """Toggle fuzzy lexical search behaviour."""

    def __init__(self, *, fuzzy: bool | None = None) -> None:
        super().__init__()
        if fuzzy is not None:
            self.fuzzy = fuzzy

    def run(self, vector_store: MongoDBVectorStore, query: str) -> list[Document]:
        return self._search_with_current_parameters(vector_store, query)
