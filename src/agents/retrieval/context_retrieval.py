from concurrent.futures import ThreadPoolExecutor, as_completed

from numpy import array
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

from src.vector_store.utils import bm25_similarity
from src.states import RetrievalState
from src.utils.utils import (
    mongo_client,
    MONGO_DATABASE_NAME,
    embedding_model,
    logger,
    RetrievedContext,
    log_execution_time
)
from src.agents.retrieval.utils import aggregate_by_url


class ContextRetrieval:
    def __init__(self, num_chunks: int = 2):
        self.num_chunks = num_chunks
        self.graph_edges_collection = mongo_client[MONGO_DATABASE_NAME]["edges"]
        self.chunks_collection = mongo_client[MONGO_DATABASE_NAME]["chunks"]

    @log_execution_time
    def __call__(self, state: RetrievalState):
        retrieved_chunks = state["retrieved_chunks"]
        contexts = []

        def process(url, chunks):
            related_chunks = self.process_single_url(url, chunks)
            return RetrievedContext(
                source_url=url,
                chunks=chunks,
                related_chunks=related_chunks,
            )

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process, url, chunks)
                for url, chunks in retrieved_chunks.items()
            ]
            for future in as_completed(futures):
                contexts.append(future.result())

        return {"retrieved_context": contexts}

    def process_single_url(self, url, retrieved_chunks) -> dict:
        retrieved_context = " ".join([d.page_content for d in retrieved_chunks])

        related_urls = self._find_related_urls(url)
        logger.info("Found {} related URLs for {}".format(len(related_urls), url))

        chunks = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._get_chunks_for_url, url): url for url in related_urls}
            for future in as_completed(futures):
                new_chunks = future.result()
                chunks.extend([c for c in new_chunks if c not in retrieved_chunks])

        if chunks:
            context_embedding = embedding_model.encode([retrieved_context]).tolist()
            related_chunks = self._combined_similarity(
                retrieved_context=retrieved_context,
                retrieved_context_embedding=context_embedding,
                chunks=chunks,
                bm25_similarity_func=bm25_similarity,
                bm25_weight=0,
                top_n=self.num_chunks,
            )
            return aggregate_by_url([c["chunk"] for c in related_chunks])
        else:
            return {}

    def _find_related_urls(self, node: str):
        """Get list of all related urls for a given url."""
        collection = self.graph_edges_collection

        results = collection.find({
            "$or": [
                {"source": node},
                {"target": node}
            ]
        })

        related_nodes = set(node)
        for doc in results:
            if doc["source"] == node:
                related_nodes.add(doc["target"])
            elif doc["target"] == node:
                related_nodes.add(doc["source"])

        return list(related_nodes)

    def _get_chunks_for_url(self, url: str):
        """Returns all chunks for a given url."""
        chunks = list(self.chunks_collection.find({"metadata.url": url}))
        return [
            Document(page_content=d["text"], metadata=d["metadata"] | {"embedding": d["embedding"]}) for d in chunks
        ]

    def _combined_similarity(
        self,
        retrieved_context: str,
        retrieved_context_embedding: list[float],
        chunks: list[Document],
        bm25_similarity_func,
        bm25_weight: float = 0.5,
        top_n: int = 5
    ) -> list[dict]:
        """
        Combines BM25 and embedding-based similarity to rank chunks.

        Args:
            retrieved_context: The query or summary text.
            retrieved_context_embedding: Precomputed embedding of the summary.
            chunks: List of chunks, each with 'text' and 'embedding' fields.
            bm25_similarity_func: A function that takes two strings and returns BM25 similarity.
            bm25_weight: Weight for BM25 similarity in the final score (0 ≤ bm25_weight ≤ 1).
            top_n: Number of top chunks to return.

        Returns:
            List of top_n chunks with their combined similarity score.
        """
        summary_embedding = array(retrieved_context_embedding).reshape(1, -1)
        chunk_embeddings = array([chunk.metadata["embedding"] for chunk in chunks])

        embedding_similarities = cosine_similarity(summary_embedding, chunk_embeddings)[0]

        combined_scores = []
        for idx, chunk in enumerate(chunks):
            bm25_score = bm25_similarity_func(retrieved_context, chunk.page_content)
            embedding_score = embedding_similarities[idx]
            combined_score = bm25_weight * bm25_score + (1 - bm25_weight) * embedding_score
            combined_scores.append((idx, combined_score))

        top_indices = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_n]

        return [
            {
                "chunk": chunks[i],
                "combined_score": score,
                "bm25_score": bm25_similarity_func(retrieved_context, chunks[i].page_content),
                "embedding_score": embedding_similarities[i],
            }
            for i, score in top_indices
        ]
