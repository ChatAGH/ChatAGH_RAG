from __future__ import annotations

from typing import Mapping

from langchain_core.documents import Document

from chat_agh.vector_store.mongodb import MongoDBVectorStore

from scripts.evaluation_tasks.search_parameters.base import CurrentSearchParameters


class FilterSearchParameters(CurrentSearchParameters):
    """Apply document-level filters in Atlas Search queries."""

    def __init__(self, *, filter: Mapping[str, object] | None = None) -> None:
        super().__init__()
        if filter is not None:
            self.filter = dict(filter)

    def run(self, vector_store: MongoDBVectorStore, query: str) -> list[Document]:
        return self._search_with_current_parameters(vector_store, query)
