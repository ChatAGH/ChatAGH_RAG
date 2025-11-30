from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

if TYPE_CHECKING:
    from chat_agh.vector_store.mongodb import MongoDBVectorStore
else:
    MongoDBVectorStore = Any  # type: ignore[assignment]

from scripts.evaluation_tasks.base import CurrentSearchParameters


class ExactSearchParameters(CurrentSearchParameters):
    """Toggle exact matching behaviour for lexical search."""

    def __init__(self, *, exact: bool | None = None) -> None:
        super().__init__()
        if exact is not None:
            self.exact = exact

    def run(self, vector_store: MongoDBVectorStore, query: str) -> list[Document]:
        return self._search_with_current_parameters(vector_store, query)
