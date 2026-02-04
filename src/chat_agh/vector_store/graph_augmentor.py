from __future__ import annotations

import heapq
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, DefaultDict, Set

from pymongo import MongoClient


# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class EntityMatch:
    """Represents an entity detected in input text."""

    url: str
    name: str
    type: Optional[str]
    bm25_score: float
    matched_on: str  # "exact" | "partial" | "subsequence"
    matched_tokens: Tuple[str, ...] = ()
    llm_description: Optional[str] = None
    sources: Optional[Any] = None


@dataclass(frozen=True)
class RelationshipHit:
    """Represents a relationship where both endpoints were detected in input text."""

    source_entity_url: str
    source_entity_name: str
    target_entity_url: str
    target_entity_name: str
    description: Optional[str]


class GraphAugmentor:
    """
    Fast entity detection + relationship extraction.

    Key changes vs your original:
    - Candidate generation via token->entity inverted index (no O(N) scoring per doc)
    - Candidate-only BM25 scoring (fast)
    - Heavy fields (relationships, sources, llm_description) are fetched only for selected URLs
    - run_many hydrates heavy fields once per batch (1 Mongo query)
    """

    _WORD_RE = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŻŹąćęłńóśżź]+", re.UNICODE)

    def __init__(
        self,
        mongo_uri: str,
        *,
        db_name: str = "chat_agh",
        collection_name: str = "entities",
        url_field: str = "url",
        name_field: str = "name",
        type_field: str = "type",
        llm_description_field: str = "llm_description",
        sources_field: str = "sources",
        relationships_field: str = "relationships",
        min_token_len: int = 4,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        # Mongo reliability defaults
        server_selection_timeout_ms: int = 5000,
        connect_timeout_ms: int = 5000,
        socket_timeout_ms: int = 20000,
        max_pool_size: int = 50,
    ):
        self.client = MongoClient(
            mongo_uri,
            retryReads=True,
            retryWrites=True,
            serverSelectionTimeoutMS=server_selection_timeout_ms,
            connectTimeoutMS=connect_timeout_ms,
            socketTimeoutMS=socket_timeout_ms,
            maxPoolSize=max_pool_size,
        )
        self.col = self.client[db_name][collection_name]

        self.url_field = url_field
        self.name_field = name_field
        self.type_field = type_field
        self.llm_description_field = llm_description_field
        self.sources_field = sources_field
        self.relationships_field = relationships_field
        self.min_token_len = int(min_token_len)

        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.set_debug(debug)

        # Light index storage
        self._urls: List[str] = []
        self._names: List[str] = []
        self._types: List[Optional[str]] = []
        self._name_norms: List[str] = []
        self._name_tokens: List[List[str]] = []
        self._required_tokens: List[Tuple[str, ...]] = []
        self._required_sets: List[Set[str]] = []
        self._tf: List[Dict[str, int]] = []
        self._dl: List[int] = []

        # BM25 stats
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 1.0
        self._N: int = 0
        self._k1: float = 1.5
        self._b: float = 0.75
        self._df: Dict[str, int] = {}

        # Inverted index (token -> entity ids)
        self._postings: DefaultDict[str, List[int]] = defaultdict(list)

        # Heavy doc cache by url (relationships/llm_description/sources)
        self._heavy_by_url: Dict[str, Dict[str, Any]] = {}

        self._loaded = False

    # -----------------------------
    # Context manager / cleanup
    # -----------------------------
    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def __enter__(self) -> "GraphAugmentor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -----------------------------
    # Logging
    # -----------------------------
    def set_debug(self, enabled: bool) -> None:
        self.debug = bool(enabled)
        # Avoid clobbering global logger config; just set our logger level
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

    # -----------------------------
    # Text utils
    # -----------------------------
    @staticmethod
    def _normalize_ws(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower()).strip()

    def _tokenize(self, s: str) -> List[str]:
        return [t.lower() for t in self._WORD_RE.findall(s)]

    @staticmethod
    def _is_char_subsequence(needle: str, haystack: str) -> bool:
        # Both assumed lower already
        needle = re.sub(r"\W+", "", needle)
        haystack = re.sub(r"\W+", "", haystack)
        if not needle:
            return False
        j = 0
        for ch in haystack:
            if ch == needle[j]:
                j += 1
                if j == len(needle):
                    return True
        return False

    # -----------------------------
    # Load + build indexes (LIGHT only)
    # -----------------------------
    def load_entities(
        self, *, query: Optional[Dict[str, Any]] = None, limit: Optional[int] = None
    ) -> int:
        """
        Loads ONLY light fields needed for candidate generation:
        url, name, type

        Heavy fields (relationships/llm_description/sources) are hydrated on demand for selected URLs.
        """
        q = query or {}

        # Fail fast if mongo unreachable
        self.client.admin.command("ping")

        projection = {
            self.url_field: 1,
            self.name_field: 1,
            self.type_field: 1,
        }

        self.logger.info("Loading LIGHT entity index from MongoDB...")
        cursor = self.col.find(q, projection=projection)
        if limit is not None:
            cursor = cursor.limit(int(limit))

        urls: List[str] = []
        names: List[str] = []
        types: List[Optional[str]] = []
        name_norms: List[str] = []
        name_tokens: List[List[str]] = []
        required_tokens: List[Tuple[str, ...]] = []
        required_sets: List[Set[str]] = []
        tf_list: List[Dict[str, int]] = []
        dl_list: List[int] = []

        df: Counter[str] = Counter()
        postings: DefaultDict[str, List[int]] = defaultdict(list)

        skipped = 0
        for doc in cursor:
            url = doc.get(self.url_field)
            name = doc.get(self.name_field)
            if not url or not name:
                skipped += 1
                continue

            url_s = str(url)
            name_s = str(name)
            typ = doc.get(self.type_field)

            toks = self._tokenize(name_s)
            if not toks:
                skipped += 1
                continue

            # required tokens used for verification precision
            req = tuple(t for t in toks if len(t) >= self.min_token_len)
            req_set = set(req)

            # BM25 doc representation (use all tokens; short tokens help recall in candidate gen)
            tf = dict(Counter(toks))
            dl = len(toks)

            idx = len(urls)
            urls.append(url_s)
            names.append(name_s)
            types.append(typ)
            name_norms.append(self._normalize_ws(name_s))
            name_tokens.append(toks)
            required_tokens.append(req)
            required_sets.append(req_set)
            tf_list.append(tf)
            dl_list.append(dl)

            # df for BM25 (unique tokens per doc)
            for t in set(toks):
                df[t] += 1

            # postings for candidate generation:
            # include all tokens (even short) to find candidates, but verification will enforce min_token_len
            for t in set(toks):
                postings[t].append(idx)

        self._urls = urls
        self._names = names
        self._types = types
        self._name_norms = name_norms
        self._name_tokens = name_tokens
        self._required_tokens = required_tokens
        self._required_sets = required_sets
        self._tf = tf_list
        self._dl = dl_list
        self._postings = postings

        self._N = len(urls)
        self._avgdl = (sum(dl_list) / self._N) if self._N else 1.0
        self._df = dict(df)

        # BM25 IDF
        # idf = log(1 + (N - df + 0.5)/(df + 0.5))
        idf: Dict[str, float] = {}
        N = self._N
        for t, dft in df.items():
            idf[t] = math.log(1.0 + (N - dft + 0.5) / (dft + 0.5))
        self._idf = idf

        self._loaded = True
        self.logger.info(
            "Loaded %d entities (skipped %d), built inverted index + BM25 stats.",
            self._N,
            skipped,
        )
        return self._N

    # -----------------------------
    # Candidate generation + BM25 scoring (FAST)
    # -----------------------------
    def _bm25_score_candidate(self, ent_id: int, q_counts: Counter[str]) -> float:
        tf = self._tf[ent_id]
        dl = self._dl[ent_id]
        denom_const = self._k1 * (1.0 - self._b + self._b * (dl / self._avgdl))

        score = 0.0
        for t, qf in q_counts.items():
            f = tf.get(t)
            if not f:
                continue
            idf = self._idf.get(t, 0.0)
            # standard BM25 term
            term = idf * (f * (self._k1 + 1.0)) / (f + denom_const)
            # lightly account for repeated query tokens
            score += term * (1.0 + 0.1 * (qf - 1))
        return score

    def _generate_candidates(
        self,
        text_tokens: List[str],
        *,
        max_seed_tokens: int,
        max_candidates_cap: int,
        allow_full_scan_fallback: bool,
    ) -> List[int]:
        token_set = set(text_tokens)
        if not token_set:
            return []

        # Sort tokens by rarity (df asc) => fewer postings => fewer candidates
        # Missing tokens treated as very common (skip)
        def rarity_key(t: str) -> int:
            return self._df.get(t, 10**9)

        seed_tokens = sorted(token_set, key=rarity_key)
        candidates: Set[int] = set()

        for t in seed_tokens[:max_seed_tokens]:
            plist = self._postings.get(t)
            if not plist:
                continue
            candidates.update(plist)
            if len(candidates) >= max_candidates_cap:
                break

        if not candidates and allow_full_scan_fallback:
            # rare fallback: no overlap tokens at all
            return list(range(self._N))

        return list(candidates)

    # -----------------------------
    # Verification (exact/partial/subsequence) using precomputed entity data
    # -----------------------------
    def _match_name_in_text(
        self,
        text_norm: str,
        text_tokens: List[str],
        text_token_set: Set[str],
        *,
        ent_id: int,
        min_match_tokens: int,
        min_match_ratio: float,
        enable_subsequence: bool,
        max_subseq_checks: int,
    ) -> Tuple[bool, str, Tuple[str, ...]]:
        name_norm = self._name_norms[ent_id]
        name = self._names[ent_id]

        # --- 1) exact match with token-boundary protection for single-token names ---
        if name_norm:
            if " " in name_norm:
                if name_norm in text_norm:
                    req = tuple(
                        t
                        for t in self._name_tokens[ent_id]
                        if len(t) >= self.min_token_len
                    )
                    return True, "exact", req
            else:
                # boundary-safe check on normalized whitespace string
                padded = f" {text_norm} "
                needle = f" {name_norm} "
                if needle in padded:
                    req = tuple(
                        t
                        for t in self._name_tokens[ent_id]
                        if len(t) >= self.min_token_len
                    )
                    return True, "exact", req

        # --- 2) partial token coverage ---
        req_set = self._required_sets[ent_id]
        if not req_set:
            return False, "none", ()

        matched: Set[str] = set()
        used_subseq = False

        # fast hits
        matched |= req_set & text_token_set

        # subsequence (bounded)
        if enable_subsequence and len(matched) < len(req_set):
            # bucket doc tokens by first char to cut comparisons
            buckets: DefaultDict[str, List[str]] = defaultdict(list)
            for tt in text_tokens:
                if tt:
                    buckets[tt[0]].append(tt)

            checks = 0
            for tok in req_set:
                if tok in matched:
                    continue
                if checks >= max_subseq_checks:
                    break
                # only compare against tokens starting with same first char (heuristic)
                cand_toks = buckets.get(tok[0], [])
                for tt in cand_toks:
                    checks += 1
                    if checks >= max_subseq_checks:
                        break
                    if self._is_char_subsequence(tok, tt):
                        matched.add(tok)
                        used_subseq = True
                        break

        coverage = len(matched) / len(req_set) if req_set else 0.0
        ok = (len(matched) >= min_match_tokens) and (coverage >= min_match_ratio)

        if self.debug:
            self.logger.debug(
                "verify ent=%r req=%d matched=%d cov=%.2f ok=%s",
                name,
                len(req_set),
                len(matched),
                coverage,
                ok,
            )

        if not ok:
            return False, "none", tuple(sorted(matched))

        return (
            True,
            ("subsequence" if used_subseq else "partial"),
            tuple(sorted(matched)),
        )

    # -----------------------------
    # Public: entity detection
    # -----------------------------
    def find_entities(
        self,
        text: str,
        *,
        candidate_k: int = 1000,
        bm25_min_score: float = 0.0,
        min_match_tokens: int = 1,
        min_match_ratio: float = 0.6,
        enable_subsequence: bool = True,
        # new knobs (safe defaults)
        max_seed_tokens: int = 12,
        max_candidates_cap: int = 50_000,
        allow_full_scan_fallback: bool = True,
        max_subseq_checks: int = 20_000,
    ) -> List[EntityMatch]:
        if not self._loaded:
            raise RuntimeError("Index not built. Call load_entities() first.")

        text_norm = self._normalize_ws(text)
        text_tokens = self._tokenize(text)
        text_token_set = set(text_tokens)
        if not text_tokens:
            return []

        candidates = self._generate_candidates(
            text_tokens,
            max_seed_tokens=max_seed_tokens,
            max_candidates_cap=max_candidates_cap,
            allow_full_scan_fallback=allow_full_scan_fallback,
        )

        q_counts = Counter(text_tokens)

        # score only candidates
        scored: List[Tuple[float, int]] = []
        for ent_id in candidates:
            s = self._bm25_score_candidate(ent_id, q_counts)
            if s >= bm25_min_score:
                scored.append((s, ent_id))

        if not scored:
            return []

        # take top-k by score without sorting everything
        top = heapq.nlargest(max(1, int(candidate_k)), scored, key=lambda x: x[0])

        matches: List[EntityMatch] = []
        for score, ent_id in top:
            found, matched_on, matched_tokens = self._match_name_in_text(
                text_norm,
                text_tokens,
                text_token_set,
                ent_id=ent_id,
                min_match_tokens=min_match_tokens,
                min_match_ratio=min_match_ratio,
                enable_subsequence=enable_subsequence,
                max_subseq_checks=max_subseq_checks,
            )
            if not found:
                continue

            url = self._urls[ent_id]
            matches.append(
                EntityMatch(
                    url=url,
                    name=self._names[ent_id],
                    type=self._types[ent_id],
                    bm25_score=float(score),
                    matched_on=matched_on,
                    matched_tokens=matched_tokens,
                    # hydrated later (run/run_many)
                    llm_description=None,
                    sources=None,
                )
            )

        matches.sort(key=lambda x: x.bm25_score, reverse=True)
        return matches

    # -----------------------------
    # Heavy hydration (relationships, llm_description, sources)
    # -----------------------------
    def _ensure_hydrated(self, urls: Iterable[str]) -> None:
        missing = [u for u in urls if u and (u not in self._heavy_by_url)]
        if not missing:
            return

        projection = {
            self.url_field: 1,
            self.llm_description_field: 1,
            self.sources_field: 1,
            self.relationships_field: 1,
            self.name_field: 1,
        }

        cursor = self.col.find(
            {self.url_field: {"$in": missing}}, projection=projection
        )
        for doc in cursor:
            u = doc.get(self.url_field)
            if not u:
                continue
            self._heavy_by_url[str(u)] = doc

        # ensure cache placeholders to avoid repeated DB hits for not-found urls
        for u in missing:
            self._heavy_by_url.setdefault(u, {})

    # -----------------------------
    # Relationships (uses hydrated cache)
    # -----------------------------
    def extract_relationships(
        self, entities: List[EntityMatch]
    ) -> List[RelationshipHit]:
        if not entities:
            return []

        # hydrate relationship lists for these entities only
        self._ensure_hydrated(e.url for e in entities)

        # resolve duplicates by name safely: name_norm -> best-scoring entity
        # (don’t overwrite silently)
        by_url = {e.url: e for e in entities}
        by_name: Dict[str, List[EntityMatch]] = defaultdict(list)
        for e in sorted(entities, key=lambda x: x.bm25_score, reverse=True):
            by_name[self._normalize_ws(e.name)].append(e)

        hits: List[RelationshipHit] = []
        seen: Set[Tuple[str, str, str]] = set()  # dedupe within a doc

        for src in entities:
            src_doc = self._heavy_by_url.get(src.url) or {}
            rels = src_doc.get(self.relationships_field) or []
            if not isinstance(rels, list):
                continue

            for rel in rels:
                if not isinstance(rel, dict):
                    continue

                tgt_url = rel.get("target_entity_url")
                tgt_name = rel.get("target_entity_name")
                desc = rel.get("description")
                desc_norm = self._normalize_ws(desc or "")

                tgt: Optional[EntityMatch] = None
                if tgt_url:
                    tgt = by_url.get(str(tgt_url))
                if tgt is None and tgt_name:
                    lst = by_name.get(self._normalize_ws(str(tgt_name)), [])
                    tgt = lst[0] if lst else None

                if not tgt:
                    continue

                key = (src.url, tgt.url, desc_norm)
                if key in seen:
                    continue
                seen.add(key)

                hits.append(
                    RelationshipHit(
                        source_entity_url=src.url,
                        source_entity_name=src.name,
                        target_entity_url=tgt.url,
                        target_entity_name=tgt.name,
                        description=desc,
                    )
                )
        return hits

    # -----------------------------
    # Filtering / selection helpers
    # -----------------------------
    def _filter_entities(
        self,
        entities: List[EntityMatch],
        *,
        include_types: Optional[Iterable[str]] = None,
        exclude_types: Optional[Iterable[str]] = None,
        name_regex: Optional[str] = None,
        url_regex: Optional[str] = None,
        matched_on_in: Optional[Iterable[str]] = None,
    ) -> List[EntityMatch]:
        inc_types = set(include_types) if include_types else None
        exc_types = set(exclude_types) if exclude_types else None
        name_re = re.compile(name_regex, re.IGNORECASE) if name_regex else None
        url_re = re.compile(url_regex, re.IGNORECASE) if url_regex else None
        on_set = set(matched_on_in) if matched_on_in else None

        out: List[EntityMatch] = []
        for e in entities:
            if inc_types is not None and (e.type not in inc_types):
                continue
            if exc_types is not None and (e.type in exc_types):
                continue
            if on_set is not None and (e.matched_on not in on_set):
                continue
            if name_re is not None and not name_re.search(e.name):
                continue
            if url_re is not None and not url_re.search(e.url):
                continue
            out.append(e)
        return out

    def _serialize(
        self, objs: List[Any], *, fields: Optional[Iterable[str]]
    ) -> List[Dict[str, Any]]:
        if fields is None:
            return [asdict(o) for o in objs]
        fields_set = set(fields)
        return [{k: v for k, v in asdict(o).items() if k in fields_set} for o in objs]

    def _attach_heavy_fields(self, entities: List[EntityMatch]) -> List[EntityMatch]:
        """Return new EntityMatch objects with llm_description/sources filled from cache."""
        if not entities:
            return []
        self._ensure_hydrated(e.url for e in entities)

        out: List[EntityMatch] = []
        for e in entities:
            doc = self._heavy_by_url.get(e.url) or {}
            out.append(
                EntityMatch(
                    url=e.url,
                    name=e.name,
                    type=e.type,
                    bm25_score=e.bm25_score,
                    matched_on=e.matched_on,
                    matched_tokens=e.matched_tokens,
                    llm_description=doc.get(self.llm_description_field),
                    sources=doc.get(self.sources_field),
                )
            )
        return out

    # -----------------------------
    # One-shot pipeline
    # -----------------------------
    def run(
        self,
        text: str,
        *,
        candidate_k: int = 1000,
        bm25_min_score: float = 0.0,
        min_match_tokens: int = 1,
        min_match_ratio: float = 0.6,
        enable_subsequence: bool = True,
        max_entities: int = 10,
        max_relationships: int = 20,
        entity_include_types: Optional[Iterable[str]] = None,
        entity_exclude_types: Optional[Iterable[str]] = None,
        entity_name_regex: Optional[str] = None,
        entity_url_regex: Optional[str] = None,
        entity_matched_on_in: Optional[Iterable[str]] = None,
        relationship_description_regex: Optional[str] = None,
        as_dict: bool = True,
        return_entity_fields: Optional[Iterable[str]] = None,
        return_relationship_fields: Optional[Iterable[str]] = None,
        # forwarded to find_entities (optional)
        max_seed_tokens: int = 12,
        max_candidates_cap: int = 50_000,
        allow_full_scan_fallback: bool = True,
        max_subseq_checks: int = 20_000,
    ) -> Dict[str, Any]:
        detected = self.find_entities(
            text,
            candidate_k=candidate_k,
            bm25_min_score=bm25_min_score,
            min_match_tokens=min_match_tokens,
            min_match_ratio=min_match_ratio,
            enable_subsequence=enable_subsequence,
            max_seed_tokens=max_seed_tokens,
            max_candidates_cap=max_candidates_cap,
            allow_full_scan_fallback=allow_full_scan_fallback,
            max_subseq_checks=max_subseq_checks,
        )

        filtered = self._filter_entities(
            detected,
            include_types=entity_include_types,
            exclude_types=entity_exclude_types,
            name_regex=entity_name_regex,
            url_regex=entity_url_regex,
            matched_on_in=entity_matched_on_in,
        )

        selected = sorted(filtered, key=lambda e: e.bm25_score, reverse=True)[
            :max_entities
        ]

        # hydrate only selected for outputs + relationship extraction
        selected_h = self._attach_heavy_fields(selected)

        rels = self.extract_relationships(selected_h)

        if relationship_description_regex:
            desc_re = re.compile(relationship_description_regex, re.IGNORECASE)
            rels = [r for r in rels if desc_re.search(r.description or "")]

        rels = rels[:max_relationships]

        if not as_dict:
            return {"entities": selected_h, "relationships": rels}

        return {
            "entities": self._serialize(selected_h, fields=return_entity_fields),
            "relationships": self._serialize(rels, fields=return_relationship_fields),
        }

    # -----------------------------
    # Batch pipeline (1 heavy Mongo query total)
    # -----------------------------
    def run_many(
        self,
        documents: List[str],
        *,
        doc_ids: Optional[List[str]] = None,
        include_per_doc: bool = False,
        agg_max_entities: int = 200,
        agg_max_relationships: int = 500,
        per_doc_max_entities: int = 50,
        per_doc_max_relationships: int = 50,
        **run_kwargs: Any,
    ) -> Dict[str, Any]:
        if doc_ids is not None and len(doc_ids) != len(documents):
            raise ValueError("doc_ids must be the same length as documents")

        ids = doc_ids or [str(i) for i in range(len(documents))]

        # ---------- Stage 1: detect + select entities per doc (NO heavy fetch) ----------
        per_doc_selected: List[Tuple[str, List[EntityMatch]]] = []
        all_selected_urls: Set[str] = set()

        total_docs = 0
        for doc_id, text in zip(ids, documents):
            total_docs += 1

            out = self.run(
                text,
                as_dict=False,
                max_entities=per_doc_max_entities,
                max_relationships=0,  # defer relationships to stage 2
                **run_kwargs,
            )
            selected = out["entities"]
            # out["entities"] are hydrated in run(); we want to avoid per-doc hydration in batch
            # so re-run selection without hydration:
            detected = self.find_entities(
                text,
                **{
                    k: v
                    for k, v in run_kwargs.items()
                    if k
                    in {
                        "candidate_k",
                        "bm25_min_score",
                        "min_match_tokens",
                        "min_match_ratio",
                        "enable_subsequence",
                        "max_seed_tokens",
                        "max_candidates_cap",
                        "allow_full_scan_fallback",
                        "max_subseq_checks",
                    }
                },
            )
            filtered = self._filter_entities(
                detected,
                include_types=run_kwargs.get("entity_include_types"),
                exclude_types=run_kwargs.get("entity_exclude_types"),
                name_regex=run_kwargs.get("entity_name_regex"),
                url_regex=run_kwargs.get("entity_url_regex"),
                matched_on_in=run_kwargs.get("entity_matched_on_in"),
            )
            selected_light = sorted(filtered, key=lambda e: e.bm25_score, reverse=True)[
                :per_doc_max_entities
            ]

            per_doc_selected.append((doc_id, selected_light))
            for e in selected_light:
                all_selected_urls.add(e.url)

        # ---------- Stage 2: one heavy hydration for all selected URLs ----------
        self._ensure_hydrated(all_selected_urls)

        # aggregation stores
        ent_aggr: Dict[str, Dict[str, Any]] = {}
        rel_aggr: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        per_doc_out: List[Dict[str, Any]] = []

        for doc_id, selected_light in per_doc_selected:
            # attach heavy fields now from cache
            selected_h = self._attach_heavy_fields(selected_light)[
                :per_doc_max_entities
            ]

            rels = self.extract_relationships(selected_h)[:per_doc_max_relationships]

            # aggregate entities by url
            for e in selected_h:
                url = e.url
                score = float(e.bm25_score)
                prev = ent_aggr.get(url)

                if prev is None:
                    ent_aggr[url] = {
                        **asdict(e),
                        "occurrences": 1,
                        "doc_ids": {doc_id},
                        "best_bm25_score": score,
                    }
                else:
                    prev["occurrences"] += 1
                    prev["doc_ids"].add(doc_id)
                    if score > float(prev.get("best_bm25_score", 0.0) or 0.0):
                        prev["best_bm25_score"] = score
                        # keep best snapshot
                        for k, v in asdict(e).items():
                            prev[k] = v

            # aggregate relationships by (src, tgt, normalized_desc)
            for r in rels:
                su = r.source_entity_url
                tu = r.target_entity_url
                desc_norm = self._normalize_ws(r.description or "")
                key = (su, tu, desc_norm)
                prev = rel_aggr.get(key)
                if prev is None:
                    rel_aggr[key] = {**asdict(r), "occurrences": 1, "doc_ids": {doc_id}}
                else:
                    prev["occurrences"] += 1
                    prev["doc_ids"].add(doc_id)

            if include_per_doc:
                per_doc_out.append(
                    {
                        "doc_id": doc_id,
                        "entities": [asdict(e) for e in selected_h],
                        "relationships": [asdict(r) for r in rels],
                    }
                )

        # finalize sets -> lists
        ent_list = list(ent_aggr.values())
        for e in ent_list:
            e["doc_ids"] = sorted(e["doc_ids"])

        rel_list = list(rel_aggr.values())
        for r in rel_list:
            r["doc_ids"] = sorted(r["doc_ids"])

        # priority: occurrences desc, then best_bm25_score desc
        ent_list.sort(
            key=lambda x: (
                int(x.get("occurrences", 0)),
                float(x.get("best_bm25_score", 0.0)),
            ),
            reverse=True,
        )
        rel_list.sort(key=lambda x: int(x.get("occurrences", 0)), reverse=True)

        ent_list = ent_list[:agg_max_entities]
        rel_list = rel_list[:agg_max_relationships]

        result: Dict[str, Any] = {
            "entities": ent_list,
            "relationships": rel_list,
            "stats": {
                "documents": total_docs,
                "unique_entities": len(ent_aggr),
                "unique_relationships": len(rel_aggr),
                "returned_entities": len(ent_list),
                "returned_relationships": len(rel_list),
                "hydrated_urls": len(all_selected_urls),
            },
        }
        if include_per_doc:
            result["per_doc"] = per_doc_out

        return result
