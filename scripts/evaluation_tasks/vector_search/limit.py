from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

if TYPE_CHECKING:
    from chat_agh.vector_store.mongodb import MongoDBVectorStore
else:
    MongoDBVectorStore = Any  # type: ignore[assignment]

from scripts.evaluation_tasks.base import CurrentSearchParameters


class LimitSearchParameters(CurrentSearchParameters):
    """Control stage-level limits applied before fusion per pipeline."""

    def __init__(
        self,
        *,
        dense_limit: int | None = None,
        lexical_limit: int | None = None,
    ) -> None:
        super().__init__()
        if dense_limit is not None:
            self.dense_limit = dense_limit
        if lexical_limit is not None:
            self.lexical_limit = lexical_limit

    def run(self, vector_store: MongoDBVectorStore, query: str) -> list[Document]:
        return self._search_with_current_parameters(vector_store, query)
