from typing import List, Union, Optional, Dict, Any

from langchain_core.documents import Document
from chat_agh.utils.utils import mongo_client, embedding_model


class MongoDBVectorStore:
    """
    Parametrizable vector + lexical (BM25) + hybrid (RRF) search for MongoDB Atlas.

    Requires:
      - Atlas Search index on `text` (type: search)
      - Atlas Vector Search index on `embedding` (type: vectorSearch)
      - MongoDB/Atlas version that supports $rankFusion for hybrid RRF

    Modes:
      - 'dense'        -> $vectorSearch only
      - 'lexical'      -> $search (BM25) only
      - 'hybrid_rrf'   -> $rankFusion over both pipelines
    """

    def __init__(
        self,
        collection_name: str,
        db_name: str = "chat_agh",
        vector_index_name: str = "vector_index",
        search_index_name: str = "default",
        text_field: Union[str, List[str]] = "text",
        vector_field: str = "embedding",
        similarity: str = "cosine",           # 'cosine' | 'dotProduct' | 'euclidean'
        create_indexes: bool = True,
    ):
        self.collection = mongo_client[db_name][collection_name]

        # Embeddings
        self.dense_model = embedding_model
        self.num_dimensions = getattr(self.dense_model, "get_sentence_embedding_dimension", lambda: None)() or 1024

        # Config
        self.text_field = text_field
        self.vector_field = vector_field
        self.vector_index_name = vector_index_name
        self.search_index_name = search_index_name
        self.similarity = similarity

        if create_indexes:
            self._ensure_search_indexes()

    def _ensure_search_indexes(self) -> None:
        """
        Creates Atlas Search + Vector Search indexes if they don't exist yet.
        Uses PyMongo's create_search_index / list_search_indexes.
        """
        try:
            existing = [idx.get("name") for idx in self.collection.list_search_indexes()]  # PyMongo 4.8+
        except Exception:
            existing = []

        if self.search_index_name not in existing:
            try:
                if isinstance(self.text_field, list):
                    fields_mapping = {f: {"type": "string"} for f in self.text_field}
                else:
                    fields_mapping = {self.text_field: {"type": "string"}}
                search_index_model = {
                    "name": self.search_index_name,
                    "definition": {
                        "mappings": {
                            "dynamic": False,
                            "fields": fields_mapping,
                        }
                    },
                }
                self.collection.create_search_index(search_index_model)
            except Exception:
                pass

        if self.vector_index_name not in existing:
            try:
                vector_index_model = {
                    "name": self.vector_index_name,
                    "type": "vectorSearch",
                    "definition": {
                        "fields": [{
                            "type": "vector",
                            "path": self.vector_field,
                            "numDimensions": self.num_dimensions,
                            "similarity": self.similarity
                        }]
                    }
                }
                self.collection.create_search_index(vector_index_model)
            except Exception:
                pass

    def indexing(self, documents: List[Document], batch_size: int = 500):
        """
        Insert docs with precomputed embeddings.
        Each record: { text, metadata, embedding }
        """
        results = []
        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc.page_content for doc in batch]
            embeddings = self.dense_model.encode(texts, normalize_embeddings=(self.similarity == "cosine")).tolist()
            records = [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    self.vector_field: emb
                }
                for doc, emb in zip(batch, embeddings)
            ]
            res = self.collection.insert_many(records)
            results.append(res)
            print(f"Inserted batch {i // batch_size + 1}: {len(records)} documents")
        return results

    def _dense_pipeline(self, query_vector, limit: int, num_candidates: int):
        return [{
            "$vectorSearch": {
                "index": self.vector_index_name,
                "path": self.vector_field,
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": limit
            }
        }]

    def _lexical_pipeline(self, query: str, limit: int, fuzzy: bool, min_should: Optional[int] = None):
        text_stage: Dict[str, Any] = {
            "index": self.search_index_name,
            "text": {
                "query": query,
                "path": self.text_field
            }
        }
        if fuzzy:
            text_stage["text"]["fuzzy"] = {"maxEdits": 1, "prefixLength": 2}
        pipeline = [{"$search": text_stage}, {"$limit": limit}]
        return pipeline

    def search(
            self,
            query: str,
            k: int = 5,
            mode: str = "hybrid_rrf",  # 'dense' | 'lexical' | 'hybrid_rrf'
            num_candidates: int = 40,
            fuzzy: bool = True,
            vector_weight: float = 1.0,
            text_weight: float = 1.0,
            inner_limits: Optional[dict] = None,
    ) -> List[Document]:
        """
        Execute a search query using dense, lexical, or hybrid RRF fusion.
        Python-based RRF is always used for hybrid.

        Parameters
        ----------
        query : str
            User query string.
        k : int
            Number of top results to return.
        mode : str
            'dense', 'lexical', or 'hybrid_rrf'.
        num_candidates : int
            Candidates to fetch for dense search before fusion.
        fuzzy : bool
            Enable fuzzy matching for lexical search.
        vector_weight : float
            Weight for dense scores in hybrid fusion.
        text_weight : float
            Weight for lexical scores in hybrid fusion.
        inner_limits : dict, optional
            Limit of results per pipeline before fusion. Example: {"dense": 25, "lexical": 25}.
        """
        inner_limits = inner_limits or {"dense": k, "lexical": k}

        if mode not in {"dense", "lexical", "hybrid_rrf"}:
            raise ValueError("mode must be one of: 'dense', 'lexical', 'hybrid_rrf'")

        results_dense = []
        results_lexical = []

        if mode in {"dense", "hybrid_rrf"}:
            query_vector = self.dense_model.encode(
                query,
                normalize_embeddings=(self.similarity == "cosine")
            ).tolist()
            pipeline_dense = self._dense_pipeline(query_vector, inner_limits["dense"], num_candidates) + [{
                "$project": {
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }]
            results_dense = list(self.collection.aggregate(pipeline_dense))

        if mode in {"lexical", "hybrid_rrf"}:
            pipeline_lex = self._lexical_pipeline(query, inner_limits["lexical"], fuzzy) + [{
                "$project": {
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "searchScore"}
                }
            }]
            results_lexical = list(self.collection.aggregate(pipeline_lex))

        if mode == "dense":
            docs = results_dense
        elif mode == "lexical":
            docs = results_lexical
        else:
            def rrf_score(rank, k_rrf=60):
                return 1.0 / (k_rrf + rank)

            scores: Dict[str, float] = {}
            all_docs = {}

            for idx, doc in enumerate(results_dense):
                doc_id = doc["_id"]
                scores[doc_id] = scores.get(doc_id, 0) + rrf_score(idx) * vector_weight
                all_docs[doc_id] = doc

            for idx, doc in enumerate(results_lexical):
                doc_id = doc["_id"]
                scores[doc_id] = scores.get(doc_id, 0) + rrf_score(idx) * text_weight
                all_docs[doc_id] = doc

            top_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
            docs = [all_docs[_id] for _id in top_ids]

        return [
            Document(
                page_content=doc.get("text", ""),
                metadata={**doc.get("metadata", {}), "id": doc.get("_id"), "score": doc.get("score", 0)}
            )
            for doc in docs
        ]

