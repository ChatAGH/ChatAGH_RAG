from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union, cast

from langchain_core.documents import Document

from chat_agh.states import ChatState
from chat_agh.utils import log_execution_time, logger
from chat_agh.vector_store.mongodb import MongoDBVectorStore


SearchOut = Union[
    List[Document],
    List[List[Document]],
    Dict[str, Any],
]


@dataclass(frozen=True)
class WindowCandidate:
    """
    One rerankable candidate unit.

    - In non-window mode: docs has length 1.
    - In window mode: docs has length window_size (before+after+1), ordered by sequence_number.
    """

    docs: List[Document]
    score: float
    # Optional: used for dedup/diversity, debugging, etc.
    group_key: Optional[str] = None


class InitialRetrievalNode:
    def __init__(
        self,
        collections: list[str],
        num_chunks: int = 5,
        search_config: dict = None,
    ):
        # NOTE: In window mode, num_chunks means "top windows", not "top documents".
        self.num_chunks = num_chunks
        self.vector_stores = [
            MongoDBVectorStore(collection) for collection in collections
        ]

        if search_config is None:
            self.search_config = {
                "mode": "dense",
                "final_limit": 5,
                "dense_limit": 50,
                "num_dense_candidates": 400,
                "window": {"enabled": False},
                "augment": False,
            }
        else:
            self.search_config = search_config

    @log_execution_time
    def __call__(self, state: ChatState) -> dict[str, list[Document]]:
        chat_history = state["chat_history"]
        chat_history_text = "".join(
            [cast(str, m.content) for m in chat_history.messages]
        )

        if len(chat_history_text) <= 20:
            logger.info(f"Initial retrieval skipped: {chat_history_text}")
            return {"context": []}

        logger.info(f"Initial retrieval, query: {chat_history_text}")

        # Run search on all collections
        with ThreadPoolExecutor() as executor:
            per_store_outputs: List[SearchOut] = list(
                executor.map(
                    lambda vs: vs.search(chat_history_text, **self.search_config),
                    self.vector_stores,
                )
            )

        # Normalize outputs -> window candidates (each window is one unit)
        all_candidates: List[WindowCandidate] = []
        augmentation: Optional[Dict[str, Any]] = None

        for out in per_store_outputs:
            cand, aug = self._extract_candidates(out)
            all_candidates.extend(cand)
            # Keep augmentation only once (first one wins), same idea as your previous code
            if augmentation is None and aug:
                augmentation = aug

        selected = self._rerank_candidates(all_candidates)

        # Flatten only after selecting top windows
        final_docs = self._flatten_selected(selected)

        # Attach augmentation ONCE (don’t duplicate big payloads)
        if augmentation and final_docs:
            md0 = dict(final_docs[0].metadata or {})
            md0["augmentation"] = augmentation
            final_docs[0] = Document(
                page_content=final_docs[0].page_content, metadata=md0
            )

        logger.info(
            f"Found {len(all_candidates)} candidates in {len(self.vector_stores)} collections, "
            f"selected windows: {len(selected)}, final docs: {len(final_docs)}"
        )

        return {
            "context": final_docs,
            "retrieved_chunks": final_docs,
        }

    # -----------------------------
    # Normalization (SearchOut -> WindowCandidate[])
    # -----------------------------
    def _extract_candidates(
        self, out: SearchOut
    ) -> tuple[List[WindowCandidate], Optional[Dict[str, Any]]]:
        """
        Converts MongoDBVectorStore.search() output into window-level candidates.

        Supported:
          - List[Document] -> each doc becomes a single-doc window candidate
          - List[List[Document]] -> each inner list is one window candidate
          - Dict {"results": ..., "augmentation": ...} -> unwrap + return augmentation separately
        """
        augmentation: Optional[Dict[str, Any]] = None
        results_obj: Any = out

        if isinstance(out, dict) and "results" in out:
            results_obj = out.get("results")
            aug = out.get("augmentation")
            if isinstance(aug, dict):
                augmentation = aug

        candidates: List[WindowCandidate] = []

        # window mode: List[List[Document]]
        if (
            isinstance(results_obj, list)
            and results_obj
            and isinstance(results_obj[0], list)
        ):
            windows = cast(List[List[Document]], results_obj)
            for wi, w in enumerate(windows, start=1):
                if not w:
                    continue

                # clone docs + add window markers
                cloned: List[Document] = []
                for ci, d in enumerate(w, start=1):
                    md = dict(d.metadata or {})
                    md["_window"] = {
                        "index": wi,
                        "chunk_index": ci,
                        "window_size": len(w),
                    }
                    cloned.append(Document(page_content=d.page_content, metadata=md))

                score = self._aggregate_window_score(cloned)

                group_key = self._window_group_key(cloned)
                candidates.append(
                    WindowCandidate(docs=cloned, score=score, group_key=group_key)
                )

        # non-window: List[Document]
        elif isinstance(results_obj, list):
            docs = cast(List[Document], results_obj)
            for d in docs:
                md = dict(d.metadata or {})
                cloned = [Document(page_content=d.page_content, metadata=md)]
                score = self._aggregate_window_score(cloned)
                group_key = self._window_group_key(cloned)
                candidates.append(
                    WindowCandidate(docs=cloned, score=score, group_key=group_key)
                )

        return candidates, augmentation

    def _aggregate_window_score(self, docs: Sequence[Document]) -> float:
        """
        Option A: window-level score aggregation.

        NOTE: In your current MongoDBVectorStore window implementation, all docs in a window
        typically inherit the same 'score' (base hit score), so max/mean will be equivalent.
        """
        scores: List[float] = []
        for d in docs:
            try:
                scores.append(float((d.metadata or {}).get("score", 0.0) or 0.0))
            except Exception:
                scores.append(0.0)
        return max(scores) if scores else 0.0

    def _window_group_key(self, docs: Sequence[Document]) -> Optional[str]:
        """
        Useful for debugging / optional diversity later.
        Tries to derive a stable grouping key similar to your window group_fields.
        """
        if not docs:
            return None
        md = docs[0].metadata or {}
        for k in ("url", "page_url", "source_url", "doc_id", "document_id", "page_id"):
            v = md.get(k)
            if v:
                return f"{k}:{v}"
        return None

    # -----------------------------
    # Reranking (WindowCandidate -> top K candidates)
    # -----------------------------
    def _rerank_candidates(self, cands: List[WindowCandidate]) -> List[WindowCandidate]:
        if not cands:
            return []

        # Simple: sort by aggregated window score desc
        # (Optional tie-breakers: prefer smaller windows, prefer higher seq? kept simple)
        c_sorted = sorted(cands, key=lambda c: float(c.score), reverse=True)
        return c_sorted[: self.num_chunks]

    # -----------------------------
    # Finalization (selected windows -> flat docs for context)
    # -----------------------------
    def _flatten_selected(self, selected: List[WindowCandidate]) -> List[Document]:
        """
        Flattens selected windows into the final context list, while deduplicating overlaps.
        Dedup key uses (id) if available, else (url/page_url/doc_id + sequence_number), else fallback to content hash.
        """
        out: List[Document] = []
        seen: set[str] = set()

        def key_for(d: Document) -> str:
            md = d.metadata or {}
            _id = md.get("id")
            if _id is not None:
                return f"id:{_id}"

            seq = md.get("sequence_number")
            group = (
                md.get("url")
                or md.get("page_url")
                or md.get("doc_id")
                or md.get("document_id")
                or md.get("page_id")
            )
            if group is not None and seq is not None:
                return f"g:{group}|s:{seq}"

            # last resort
            return f"txt:{hash(d.page_content or '')}"

        # preserve window internal order (you already sort by sequence in vector store)
        for cand in selected:
            for d in cand.docs:
                k = key_for(d)
                if k in seen:
                    continue
                seen.add(k)

                # annotate final per-window score (handy for debugging)
                md = dict(d.metadata or {})
                md["_window_score"] = float(cand.score)
                out.append(Document(page_content=d.page_content, metadata=md))

        return out
