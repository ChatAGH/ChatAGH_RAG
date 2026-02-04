from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, cast

from langchain_core.documents import Document

from chat_agh.utils import log_execution_time, RetrievedContext
from chat_agh.vector_store.mongodb import MongoDBVectorStore


SearchOut = Union[
    List[Document],
    List[List[Document]],
    Dict[str, Any],
]


class SimilaritySearch:
    def __init__(
        self, index_name: str, num_retrieved_chunks: int = 5, window_size: int = 1
    ) -> None:
        self.num_retrieved_chunks = num_retrieved_chunks
        self.window_size = window_size
        self.vector_store = MongoDBVectorStore(index_name)

    @log_execution_time
    def __call__(self, state: Any) -> Dict[str, Any]:
        out: SearchOut = self.vector_store.search(
            state["query"],
            mode="hybrid",
            final_limit=10,
            # NEW hybrid knobs
            hybrid_candidate_limit=250,  # how many docs to pull from vector stage before rerank
            hybrid_num_candidates=2500,  # internal ANN candidates
            dense_weight=1.0,  # keep dense signal
            bm25_weight=1.0,  # keep lexical rerank signal
            rrf_k=60,
            window={
                "enabled": True,
                "window_before": 2,
                "window_after": 2,
                "group_fields": ["url"],
                "max_total_results": 100,
            },
            augment=True,
            augment_kwargs={
                "per_doc_max_entities": 20,
                "per_doc_max_relationships": 20,
                "agg_max_entities": 40,
                "agg_max_relationships": 40,
                "candidate_k": 1600,
                "bm25_min_score": 0,
                "min_match_tokens": 1,
                "min_match_ratio": 0.9,
                "enable_subsequence": True,
            },
        )

        chunks_windows = self._to_chunks_windows(out)

        if not chunks_windows:
            raise Exception("No chunks windows found")

        retrieved_context = [
            RetrievedContext(source_url=url, chunks=chunks, related_chunks={})
            for url, chunks in chunks_windows.items()
        ]

        return {
            # "retrieved_chunks": chunks_windows,
            "retrieved_context": chunks_windows,
        }

    # -------------------------
    # Normalization helpers
    # -------------------------
    @staticmethod
    def _pick_source_url(md: Dict[str, Any]) -> str:
        for k in (
            "url",
            "page_url",
            "source_url",
            "source",
            "doc_id",
            "document_id",
            "page_id",
            "id",
        ):
            v = md.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
        return "unknown"

    def _to_chunks_windows(self, out: SearchOut) -> Dict[str, List[Document]]:
        """
        Convert `MongoDBVectorStore.search()` output into:
            dict[source_url, list[Document]]

        Rules:
          - if windows output -> each window becomes one (source_url -> chunks) entry
          - if flat docs output -> group docs by url
          - if augmented dict output -> unwrap; attach augmentation to first chunk of the first window/doc group
        """
        augmentation: Optional[Dict[str, Any]] = None
        results_obj: Any = out

        if isinstance(out, dict) and "results" in out:
            results_obj = out.get("results")
            aug = out.get("augmentation")
            if isinstance(aug, dict):
                augmentation = aug

        chunks_windows: Dict[str, List[Document]] = {}

        if (
            isinstance(results_obj, list)
            and results_obj
            and isinstance(results_obj[0], list)
        ):
            windows = cast(List[List[Document]], results_obj)
            for w in windows:
                if not w:
                    continue
                url = self._pick_source_url(dict(w[0].metadata or {}))
                chunks_windows[url] = [
                    Document(
                        page_content=d.page_content, metadata=dict(d.metadata or {})
                    )
                    for d in w
                ]

        elif isinstance(results_obj, list):
            docs = cast(List[Document], results_obj)
            for d in docs:
                md = dict(d.metadata or {})
                url = self._pick_source_url(md)
                chunks_windows.setdefault(url, []).append(
                    Document(page_content=d.page_content, metadata=md)
                )

        if augmentation and chunks_windows:
            first_url = next(iter(chunks_windows.keys()))
            if chunks_windows[first_url]:
                md0 = dict(chunks_windows[first_url][0].metadata or {})
                md0["augmentation"] = augmentation
                chunks_windows[first_url][0] = Document(
                    page_content=chunks_windows[first_url][0].page_content,
                    metadata=md0,
                )

        if self.num_retrieved_chunks and self.num_retrieved_chunks > 0:
            for url in list(chunks_windows.keys()):
                chunks_windows[url] = chunks_windows[url][: self.num_retrieved_chunks]

        return chunks_windows
