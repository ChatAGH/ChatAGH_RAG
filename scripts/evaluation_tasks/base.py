from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

from langchain_core.documents import Document

from chat_agh.vector_store.mongodb import MongoDBVectorStore

SearchMode = Literal["dense", "lexical", "hybrid_rrf"]


class CurrentSearchParameters:
    """Current production parameters used as defaults for experiments."""

    DEFAULT_MODE: SearchMode = "hybrid_rrf"
    DEFAULT_K = 5
    DEFAULT_NUM_CANDIDATES = 60
    DEFAULT_FUZZY = True
    DEFAULT_VECTOR_WEIGHT = 1.0
    DEFAULT_TEXT_WEIGHT = 1.0

    def __init__(self) -> None:
        self.mode: SearchMode = self.DEFAULT_MODE
        self.k: int = self.DEFAULT_K
        self.num_candidates: int = self.DEFAULT_NUM_CANDIDATES
        self.fuzzy: bool = self.DEFAULT_FUZZY
        self.vector_weight: float = self.DEFAULT_VECTOR_WEIGHT
        self.text_weight: float = self.DEFAULT_TEXT_WEIGHT
        self.inner_limits: Optional[dict[str, int]] = None
        self.dense_limit: Optional[int] = None
        self.lexical_limit: Optional[int] = None
        self.filter: Optional[Mapping[str, Any]] = None
        self.exact: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "k": self.k,
            "num_candidates": self.num_candidates,
            "fuzzy": self.fuzzy,
            "vector_weight": self.vector_weight,
            "text_weight": self.text_weight,
        }
        if self.inner_limits is not None:
            payload["inner_limits"] = dict(self.inner_limits)
        if self.dense_limit is not None:
            payload["dense_limit"] = self.dense_limit
        if self.lexical_limit is not None:
            payload["lexical_limit"] = self.lexical_limit
        if self.filter is not None:
            payload["filter"] = dict(self.filter)
        if self.exact is not None:
            payload["exact"] = self.exact
        return payload

    def _search_with_current_parameters(
        self, vector_store: MongoDBVectorStore, query: str
    ) -> list[Document]:
        return vector_store.search(
            query=query,
            k=self.k,
            mode=self.mode,
            vector_weight=self.vector_weight,
            text_weight=self.text_weight,
            dense_limit=self.dense_limit,
            lexical_limit=self.lexical_limit,
        )

    def run(self, vector_store: MongoDBVectorStore, query: str) -> list[Document]:
        raise NotImplementedError("Subclasses must implement run().")


class SearchParametersOverride(CurrentSearchParameters):
    """Utility search parameters override used for experiment sweeps."""

    _OVERRIDABLE_FIELDS = {
        "mode",
        "k",
        "num_candidates",
        "fuzzy",
        "vector_weight",
        "text_weight",
        "inner_limits",
        "dense_limit",
        "lexical_limit",
        "filter",
        "exact",
    }

    def __init__(self, **overrides: Any) -> None:
        super().__init__()
        for field_name, value in overrides.items():
            if field_name not in self._OVERRIDABLE_FIELDS:
                msg = f"Unsupported search parameter override: {field_name}"
                raise ValueError(msg)
            setattr(self, field_name, value)

    def run(self, vector_store: MongoDBVectorStore, query: str) -> list[Document]:
        return self._search_with_current_parameters(vector_store, query)
