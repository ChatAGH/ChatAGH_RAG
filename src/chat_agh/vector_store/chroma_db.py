from __future__ import annotations

from typing import List, Optional

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

from chat_agh.utils import embedding_model


class ChromaVectorStore:
    """
    Minimal ChromaDB-based vector store.

    Supports:
      - indexing(documents)
      - search(query, k)

    No window retrieval, no hybrid search, no augmentation.
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./chroma",
    ):
        self.embedding_model = embedding_model

        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False,
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_documents,
        )

    # ----------------------------
    # Embeddings
    # ----------------------------
    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
        ).tolist()

    def _embed_query(self, query: str) -> List[float]:
        return self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

    # ----------------------------
    # Indexing
    # ----------------------------
    def indexing(self, documents: List[Document]) -> None:
        if not documents:
            return

        ids = []
        texts = []
        metadatas = []

        for i, doc in enumerate(documents):
            ids.append(
                str(doc.metadata.get("id") or f"doc_{i}")
            )
            texts.append(doc.page_content or "")
            metadatas.append(doc.metadata or {})

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

        self.client.persist()

    # ----------------------------
    # Search
    # ----------------------------
    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        result = self.collection.query(
            query_embeddings=[self._embed_query(query)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs: List[Document] = []

        for text, md, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            metadata = dict(md or {})
            metadata["score"] = float(1.0 - dist)  # cosine similarity proxy
            metadata["_retrieval"] = {"mode": "dense", "backend": "chromadb"}

            docs.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        return docs
