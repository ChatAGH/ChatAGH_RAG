from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

from langchain_core.documents import Document
from pymongo.collection import Collection
from pymongo.results import InsertManyResult

from chat_agh.utils import embedding_model, mongo_client
from chat_agh.vector_store.graph_augmentor import GraphAugmentor


# ======================
# Configs
# ======================


@dataclass(frozen=True)
class WindowRetrievalConfig:
    enabled: bool = False
    window_before: int = 0
    window_after: int = 0

    group_fields: Tuple[str, ...] = (
        "url",
        "page_url",
        "source_url",
        "source",
        "doc_id",
        "document_id",
        "page_id",
    )

    max_total_results: Optional[int] = None


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    # basic tokenizer; keeps letters+digits (unicode), splits on non-word
    # you can tune it if needed
    min_token_len: int = 2


SearchMode = Literal["dense", "lexical", "hybrid", "hybrid_rrf_legacy"]


# ======================
# Vector Store
# ======================


class MongoDBVectorStore:
    """
    MongoDB Atlas retrieval:
      - dense: $vectorSearch
      - lexical: $search (Atlas Search)
      - hybrid: dense candidate generation -> BM25 rerank (Python)
               (optional rank-fusion inside the candidate pool)
      - hybrid_rrf_legacy: your old "dense+lexical from DB then RRF" (kept for compatibility)

    Optional window retrieval:
      - returns List[List[Document]] if enabled

    Optional graph augmentation:
      - search(..., augment=True) returns {"results": ..., "augmentation": ...}
    """

    def __init__(
        self,
        collection_name: str,
        db_name: str = "chat_agh",
        vector_index_name: str = "vector_index",
        search_index_name: str = "default",
        text_field: Union[str, List[str]] = "text",
        vector_field: str = "embedding",
        similarity: str = "cosine",  # 'cosine' | 'dotProduct' | 'euclidean'
        create_indexes: bool = False,
        logger: Optional[Any] = None,
        window_config: Optional[WindowRetrievalConfig] = None,
        # graph augmentor
        enable_graph_augmentor: bool = True,
        graph_augmentor_debug: bool = False,
    ):
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection: Collection = mongo_client[db_name][collection_name]
        self.logger = logger

        self.dense_model = embedding_model
        self.num_dimensions = (
            getattr(
                self.dense_model, "get_sentence_embedding_dimension", lambda: None
            )()
            or 1024
        )

        self.text_field = text_field
        self.vector_field = vector_field
        self.vector_index_name = vector_index_name
        self.search_index_name = search_index_name
        self.similarity = similarity

        self.window_config = window_config or WindowRetrievalConfig()

        self._enable_graph_augmentor = bool(enable_graph_augmentor)
        self._graph_augmentor_debug = bool(graph_augmentor_debug)
        self._graph_augmentor: Optional[GraphAugmentor] = None

        # if create_indexes:
        #     self._ensure_search_indexes()

        if self.logger:
            self.logger.info(
                "MongoDBVectorStore init (db=%s, collection=%s, vector_index=%s, search_index=%s)",
                db_name,
                collection_name,
                vector_index_name,
                search_index_name,
            )

    # ----------------------------
    # Index management
    # ----------------------------
    def _ensure_search_indexes(self) -> None:
        try:
            existing = [
                idx.get("name") for idx in self.collection.list_search_indexes()
            ]
        except Exception:
            existing = []

        if self.search_index_name not in existing:
            if isinstance(self.text_field, list):
                fields_mapping = {f: {"type": "string"} for f in self.text_field}
            else:
                fields_mapping = {self.text_field: {"type": "string"}}

            search_index_model = {
                "name": self.search_index_name,
                "definition": {
                    "mappings": {"dynamic": False, "fields": fields_mapping}
                },
            }
            self.collection.create_search_index(search_index_model)

        if self.vector_index_name not in existing:
            vector_index_model = {
                "name": self.vector_index_name,
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": self.vector_field,
                            "numDimensions": self.num_dimensions,
                            "similarity": self.similarity,
                        }
                    ]
                },
            }
            self.collection.create_search_index(vector_index_model)

    # ----------------------------
    # Indexing
    # ----------------------------
    def indexing(
        self,
        documents: List[Document],
        batch_size: int = 500,
        ordered: bool = False,  # faster on duplicates/errors
    ) -> List[InsertManyResult]:
        results: List[InsertManyResult] = []
        total = len(documents)

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc.page_content or "" for doc in batch]

            embeddings = self.dense_model.encode(
                texts,
                normalize_embeddings=(self.similarity == "cosine"),
            ).tolist()

            records = [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    self.vector_field: emb,
                }
                for doc, emb in zip(batch, embeddings)
            ]

            res = self.collection.insert_many(records, ordered=ordered)
            results.append(res)

            if self.logger:
                self.logger.info(
                    "Inserted batch %d (%d docs)", (i // batch_size + 1), len(records)
                )
            else:
                print(f"Inserted batch {i // batch_size + 1}: {len(records)} documents")

        return results

    # ----------------------------
    # Graph augmentation
    # ----------------------------
    def _get_graph_augmentor(self) -> GraphAugmentor:
        if not self._enable_graph_augmentor:
            raise RuntimeError("Graph augmentor disabled in MongoDBVectorStore.")
        if self._graph_augmentor is not None:
            return self._graph_augmentor

        aug = GraphAugmentor(
            mongo_uri=os.environ.get("MONGODB_URI"),
            debug=self._graph_augmentor_debug,
            logger=getattr(self.logger, "getChild", lambda _n: None)("GraphAugmentor")
            if self.logger
            else None,
        )
        aug.load_entities()  # build BM25 once
        self._graph_augmentor = aug
        return aug

    def _run_graph_augmentation(
        self,
        results: Union[List[Document], List[List[Document]]],
        augment_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        aug = self._get_graph_augmentor()
        kw = dict(augment_kwargs or {})

        if results and isinstance(results[0], list):
            windows = cast(List[List[Document]], results)
            texts = ["\n".join((d.page_content or "") for d in w) for w in windows]
            ids: List[str] = []
            for i, w in enumerate(windows):
                md = (w[0].metadata or {}) if w else {}
                ids.append(
                    str(
                        md.get("url")
                        or md.get("page_url")
                        or md.get("doc_id")
                        or f"window_{i}"
                    )
                )
            return aug.run_many(texts, doc_ids=ids, include_per_doc=False, **kw)

        hits = cast(List[Document], results)
        texts = [d.page_content or "" for d in hits]
        ids = []
        for i, d in enumerate(hits):
            md = d.metadata or {}
            ids.append(
                str(
                    md.get("url")
                    or md.get("page_url")
                    or md.get("doc_id")
                    or md.get("id")
                    or f"hit_{i}"
                )
            )
        return aug.run_many(texts, doc_ids=ids, include_per_doc=False, **kw)

    # ----------------------------
    # Low-level searches
    # ----------------------------
    @lru_cache(maxsize=1024)
    def _encode_query(self, query: str) -> List[float]:
        # caching helps if you benchmark repeated queries
        return self.dense_model.encode(
            query,
            normalize_embeddings=(self.similarity == "cosine"),
        ).tolist()

    def _vector_search(
        self,
        query: str,
        *,
        limit: int,
        num_candidates: int,
        projection: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        qv = self._encode_query(query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": self.vector_field,
                    "queryVector": qv,
                    "limit": int(limit),
                    "numCandidates": int(num_candidates),
                }
            },
            {
                "$project": projection
                or {"text": 1, "metadata": 1, "score": {"$meta": "vectorSearchScore"}}
            },
        ]

        return cast(List[Dict[str, Any]], list(self.collection.aggregate(pipeline)))

    def _lexical_search(
        self,
        query: str,
        *,
        limit: int,
        fuzzy: bool,
        fuzzy_max_edits: int,
        fuzzy_prefix_length: int,
        projection: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        text_stage: Dict[str, Any] = {
            "index": self.search_index_name,
            "text": {"query": query, "path": self.text_field},
        }
        if fuzzy:
            text_stage["text"]["fuzzy"] = {
                "maxEdits": int(fuzzy_max_edits),
                "prefixLength": int(fuzzy_prefix_length),
            }

        pipeline: List[Dict[str, Any]] = [
            {"$search": text_stage},
            {"$limit": int(limit)},
            {
                "$project": projection
                or {"text": 1, "metadata": 1, "score": {"$meta": "searchScore"}}
            },
        ]
        return cast(List[Dict[str, Any]], list(self.collection.aggregate(pipeline)))

    # ----------------------------
    # Doc building
    # ----------------------------
    @staticmethod
    def _to_docs(
        rows: List[Dict[str, Any]],
        *,
        retrieval_meta: Dict[str, Any],
        score_key: str = "score",
    ) -> List[Document]:
        docs: List[Document] = []
        for r in rows:
            md = dict(r.get("metadata", {}) or {})
            md["id"] = r.get("_id")
            md["score"] = float(r.get(score_key, 0.0) or 0.0)
            md["_retrieval"] = dict(retrieval_meta)
            docs.append(Document(page_content=r.get("text", "") or "", metadata=md))
        return docs

    # ----------------------------
    # Window retrieval (batched for speed)
    # ----------------------------
    @staticmethod
    def _pick_group_field(
        md: Dict[str, Any], group_fields: Sequence[str]
    ) -> Optional[Tuple[str, Any]]:
        for f in group_fields:
            if f in {"sequence_number", "metadata.sequence_number"}:
                continue
            if f in md and md[f] is not None:
                return f, md[f]
        return None

    def _resolve_window_config(
        self,
        window: Optional[Union[WindowRetrievalConfig, Dict[str, Any]]],
    ) -> WindowRetrievalConfig:
        if window is None:
            return self.window_config
        if isinstance(window, WindowRetrievalConfig):
            return window

        base = self.window_config
        data = dict(window)

        if "group_fields" in data and isinstance(data["group_fields"], list):
            data["group_fields"] = tuple(data["group_fields"])

        if "before" in data and "window_before" not in data:
            data["window_before"] = data.pop("before")
        if "after" in data and "window_after" not in data:
            data["window_after"] = data.pop("after")

        allowed = set(base.__dataclass_fields__.keys())
        cleaned = {k: v for k, v in data.items() if k in allowed}
        return replace(base, **cleaned)

    def _apply_window_retrieval(
        self, base_docs: List[Document], cfg: WindowRetrievalConfig
    ) -> List[List[Document]]:
        if not cfg.enabled or not base_docs:
            return []

        # Prepare window requests (batched)
        reqs: List[Tuple[int, str, Any, int, int, float, Dict[str, Any]]] = []
        for i, base in enumerate(base_docs):
            md = dict(base.metadata or {})
            seq = md.get("sequence_number")
            if seq is None:
                continue
            try:
                seq_i = int(seq)
            except Exception:
                continue

            pick = self._pick_group_field(md, cfg.group_fields)
            if pick is None:
                continue
            group_field, group_value = pick

            seq_min = seq_i - int(cfg.window_before)
            seq_max = seq_i + int(cfg.window_after)

            base_score = float(md.get("score", 0.0) or 0.0)
            base_retr = md.get("_retrieval", {}) or {}
            reqs.append(
                (i, group_field, group_value, seq_min, seq_max, base_score, base_retr)
            )

            if cfg.max_total_results is not None and len(reqs) >= int(
                cfg.max_total_results
            ):
                break

        if not reqs:
            return []

        # If no window needed, just wrap base docs
        if cfg.window_before <= 0 and cfg.window_after <= 0:
            return [[d] for d in base_docs[: (cfg.max_total_results or len(base_docs))]]

        # One DB query for all windows
        ors: List[Dict[str, Any]] = []
        for _, group_field, group_value, seq_min, seq_max, *_ in reqs:
            ors.append(
                {
                    f"metadata.{group_field}": group_value,
                    "metadata.sequence_number": {
                        "$gte": int(seq_min),
                        "$lte": int(seq_max),
                    },
                }
            )

        q = {"$or": ors} if len(ors) > 1 else ors[0]
        cursor = self.collection.find(q, {"text": 1, "metadata": 1}).sort(
            [("metadata.sequence_number", 1), ("_id", 1)]
        )
        fetched = list(cursor)

        # Bucket fetched docs by (group_field, group_value)
        buckets: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
        for r in fetched:
            md = dict(r.get("metadata", {}) or {})
            # identify bucket based on any possible group field present in metadata
            # (we’ll match later with the exact req tuple)
            for gf in cfg.group_fields:
                if gf in md:
                    key = (gf, md[gf])
                    buckets.setdefault(key, []).append(r)

        windows: List[List[Document]] = []
        for (
            _,
            group_field,
            group_value,
            seq_min,
            seq_max,
            base_score,
            base_retr,
        ) in reqs:
            recs = buckets.get((group_field, group_value), [])

            window_docs: List[Document] = []
            for rec in recs:
                rec_md = dict(rec.get("metadata", {}) or {})
                try:
                    s = int(rec_md.get("sequence_number"))
                except Exception:
                    continue
                if s < seq_min or s > seq_max:
                    continue

                rec_md["id"] = rec.get("_id")
                rec_md["score"] = base_score
                rec_md["_retrieval"] = base_retr
                window_docs.append(
                    Document(page_content=rec.get("text", "") or "", metadata=rec_md)
                )

            window_docs.sort(
                key=lambda d: (
                    int(d.metadata.get("sequence_number", 10**18)),
                    str(d.metadata.get("id")),
                )
            )
            if window_docs:
                windows.append(window_docs)
            else:
                # fallback: include base doc if nothing fetched
                windows.append([base_docs[0]])

        return windows

    # ----------------------------
    # Hybrid: dense candidates -> BM25 rerank (fast)
    # ----------------------------
    @staticmethod
    def _rrf(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    @staticmethod
    @lru_cache(maxsize=20000)
    def _tokenize_cached(text: str) -> Tuple[str, ...]:
        # cached tokenization can help in repeated reranks/benchmarks
        tokens = re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
        return tuple(tokens)

    def _tokenize(self, text: str, bm25_cfg: BM25Config) -> List[str]:
        toks = list(self._tokenize_cached(text))
        if bm25_cfg.min_token_len > 1:
            toks = [t for t in toks if len(t) >= bm25_cfg.min_token_len]
        return toks

    def _bm25_scores(
        self,
        query: str,
        docs: List[Document],
        bm25_cfg: BM25Config,
    ) -> List[float]:
        q_toks = self._tokenize(query, bm25_cfg)
        if not q_toks or not docs:
            return [0.0 for _ in docs]

        doc_toks_list: List[List[str]] = [
            self._tokenize(d.page_content or "", bm25_cfg) for d in docs
        ]
        N = len(doc_toks_list)
        doc_lens = [len(toks) for toks in doc_toks_list]
        avgdl = (sum(doc_lens) / N) if N else 0.0

        # df for query terms only (fast)
        df: Dict[str, int] = {t: 0 for t in set(q_toks)}
        for toks in doc_toks_list:
            seen = set(toks)
            for t in df.keys():
                if t in seen:
                    df[t] += 1

        def idf(n: int) -> float:
            # BM25+ style idf smoothing
            return math.log(1.0 + (N - n + 0.5) / (n + 0.5))

        idf_map = {t: idf(df[t]) for t in df.keys()}

        k1 = float(bm25_cfg.k1)
        b = float(bm25_cfg.b)

        scores: List[float] = []
        for toks, dl in zip(doc_toks_list, doc_lens):
            if not toks:
                scores.append(0.0)
                continue
            tf: Dict[str, int] = {}
            for t in toks:
                if t in idf_map:
                    tf[t] = tf.get(t, 0) + 1

            s = 0.0
            denom_norm = k1 * (1.0 - b + b * (dl / avgdl)) if avgdl > 0 else k1
            for t in set(q_toks):
                f = tf.get(t, 0)
                if f <= 0:
                    continue
                s += idf_map[t] * (f * (k1 + 1.0)) / (f + denom_norm)
            scores.append(float(s))

        return scores

    def _hybrid_dense_then_bm25(
        self,
        query: str,
        *,
        candidate_limit: int,
        num_candidates: int,
        final_limit: int,
        bm25_cfg: BM25Config,
        # rank fusion inside pool (recommended)
        dense_weight: float,
        bm25_weight: float,
        rrf_k: int,
    ) -> List[Document]:
        # 1) dense candidate pool (keep projection minimal)
        rows = self._vector_search(
            query,
            limit=int(candidate_limit),
            num_candidates=int(num_candidates),
            projection={
                "text": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"},
            },
        )
        candidates = self._to_docs(
            rows, retrieval_meta={"mode": "hybrid", "stage": "dense_pool"}
        )

        if not candidates:
            return []

        # 2) BM25 rerank on candidates (Python)
        bm25_scores = self._bm25_scores(query, candidates, bm25_cfg=bm25_cfg)

        # 3) robust fusion: RRF(dense_rank) + RRF(bm25_rank)
        # (works well without score calibration)
        dense_rank = {d.metadata["id"]: i for i, d in enumerate(candidates, start=1)}

        bm25_order = sorted(
            range(len(candidates)), key=lambda i: bm25_scores[i], reverse=True
        )
        bm25_rank: Dict[Any, int] = {}
        for r, idx in enumerate(bm25_order, start=1):
            bm25_rank[candidates[idx].metadata["id"]] = r

        fused: List[Tuple[float, int]] = []
        for i, d in enumerate(candidates):
            _id = d.metadata["id"]
            dr = dense_rank.get(_id, len(candidates) + 1)
            br = bm25_rank.get(_id, len(candidates) + 1)

            score = 0.0
            score += float(dense_weight) * self._rrf(int(dr), k=int(rrf_k))
            score += float(bm25_weight) * self._rrf(int(br), k=int(rrf_k))
            fused.append((score, i))

        fused.sort(key=lambda x: x[0], reverse=True)
        top = [candidates[i] for _, i in fused[: int(final_limit)]]

        # enrich metadata
        for d in top:
            _id = d.metadata["id"]
            d.metadata["_retrieval"] = {
                "mode": "hybrid",
                "method": "dense_then_bm25",
                "candidate_limit": int(candidate_limit),
                "dense_rank": dense_rank.get(_id),
                "bm25_rank": bm25_rank.get(_id),
                "fused_rrf_k": int(rrf_k),
                "dense_weight": float(dense_weight),
                "bm25_weight": float(bm25_weight),
            }

        return top

    # ----------------------------
    # Public search
    # ----------------------------
    def search(
        self,
        query: str,
        mode: SearchMode = "dense",
        # lexical params
        lexical_limit: int = 20,
        fuzzy: bool = False,
        fuzzy_max_edits: int = 1,
        fuzzy_prefix_length: int = 1,
        # dense params
        dense_limit: int = 20,
        num_dense_candidates: int = 200,
        final_limit: int = 5,
        # hybrid dense->bm25 params
        hybrid_candidate_limit: int = 200,  # how many docs to pull from vector stage
        hybrid_num_candidates: int = 2000,  # how deep vector stage considers internally
        bm25_cfg: Optional[BM25Config] = None,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rrf_k: int = 60,
        # legacy hybrid_rrf params (kept)
        window: Optional[Union[WindowRetrievalConfig, Dict[str, Any]]] = None,
        augment: bool = False,
        augment_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[List[Document], List[List[Document]], Dict[str, Any]]:
        if final_limit <= 0:
            return []

        cfg = self._resolve_window_config(window)
        bm25_cfg = bm25_cfg or BM25Config()

        if mode == "dense":
            rows = self._vector_search(
                query,
                limit=int(dense_limit),
                num_candidates=int(num_dense_candidates),
            )
            hits = self._to_docs(
                rows[: int(final_limit)], retrieval_meta={"mode": "dense"}
            )
            final_hits: Union[List[Document], List[List[Document]]] = (
                self._apply_window_retrieval(hits, cfg) if cfg.enabled else hits
            )
            return self._maybe_augment(
                final_hits, augment=augment, augment_kwargs=augment_kwargs
            )

        if mode == "lexical":
            rows = self._lexical_search(
                query,
                limit=int(lexical_limit),
                fuzzy=bool(fuzzy),
                fuzzy_max_edits=int(fuzzy_max_edits),
                fuzzy_prefix_length=int(fuzzy_prefix_length),
            )
            hits = self._to_docs(
                rows[: int(final_limit)], retrieval_meta={"mode": "lexical"}
            )
            final_hits = (
                self._apply_window_retrieval(hits, cfg) if cfg.enabled else hits
            )
            return self._maybe_augment(
                final_hits, augment=augment, augment_kwargs=augment_kwargs
            )

        if mode == "hybrid":
            hits = self._hybrid_dense_then_bm25(
                query,
                candidate_limit=int(hybrid_candidate_limit),
                num_candidates=int(hybrid_num_candidates),
                final_limit=int(final_limit),
                bm25_cfg=bm25_cfg,
                dense_weight=float(dense_weight),
                bm25_weight=float(bm25_weight),
                rrf_k=int(rrf_k),
            )
            final_hits = (
                self._apply_window_retrieval(hits, cfg) if cfg.enabled else hits
            )
            return self._maybe_augment(
                final_hits, augment=augment, augment_kwargs=augment_kwargs
            )

        raise ValueError("mode must be one of: dense | lexical | hybrid ")

    def _maybe_augment(
        self,
        results: Union[List[Document], List[List[Document]]],
        *,
        augment: bool,
        augment_kwargs: Optional[Dict[str, Any]],
    ) -> Union[List[Document], List[List[Document]], Dict[str, Any]]:
        if not augment:
            return results
        return {
            "results": results,
            "augmentation": self._run_graph_augmentation(
                results, augment_kwargs=augment_kwargs
            ),
        }


if __name__ == "__main__":
    import logging
    import os
    from typing import Any, Dict, List, Union, cast

    from dotenv import load_dotenv
    from langchain_core.documents import Document

    load_dotenv("/Users/wnowogorski/PycharmProjects/ChatAGH_RAG/.env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    assert os.getenv("MONGODB_URI"), "Set MONGODB_URI first."

    store = MongoDBVectorStore(
        collection_name="cluster_9_v3",
        create_indexes=True,
        logger=logging.getLogger("MongoDBVectorStore"),
    )

    query = """
    Jakie dane osobowe pracodawca AGH ma prawo żądać od pracownika oraz od osoby ubiegającej się o zatrudnienie, a
    jakie dodatkowe dane mogą być wymagane w szczególnych sytuacjach?
    """

    out = store.search(
        query,
        mode="hybrid",
        final_limit=3,
        hybrid_candidate_limit=250,
        hybrid_num_candidates=2500,
        bm25_cfg=BM25Config(k1=1.5, b=0.75, min_token_len=2),
        dense_weight=1.0,
        bm25_weight=1.0,
        rrf_k=60,
        window={
            "enabled": True,
            "window_before": 1,
            "window_after": 1,
            "group_fields": ["url"],
            "max_total_results": 50,
        },
        augment=True,
        augment_kwargs={
            "per_doc_max_entities": 30,
            "per_doc_max_relationships": 30,
            "agg_max_entities": 20,
            "agg_max_relationships": 20,
            "candidate_k": 800,
            "bm25_min_score": 0.0,
            "min_match_tokens": 1,
            "min_match_ratio": 0.9,
            "enable_subsequence": True,
        },
    )

    results: Union[List[Document], List[List[Document]]]
    augmentation: Dict[str, Any] = {}

    if isinstance(out, dict):
        results = cast(
            Union[List[Document], List[List[Document]]], out.get("results", [])
        )
        augmentation = cast(Dict[str, Any], out.get("augmentation", {}))
    else:
        results = cast(Union[List[Document], List[List[Document]]], out)

    if augmentation:
        print("\nAugmentation stats:", augmentation.get("stats"))
        print("Top entities:", (augmentation.get("entities") or [])[:5])
        print("Top relationships:", (augmentation.get("relationships") or [])[:5])

    print(out)
