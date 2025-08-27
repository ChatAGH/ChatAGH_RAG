from src.states import RetrievalState
from src.vector_store.mongodb import MongoDBVectorStore
from src.utils.utils import logger, log_execution_time, mongo_client, MONGO_DATABASE_NAME
from src.agents.retrieval.utils import aggregate_by_url


class SimilaritySearch:
    def __init__(self, index_name, num_retrieved_chunks: int = 5, window_size: int = 1):
        self.num_retrieved_chunks = num_retrieved_chunks
        self.window_size = window_size
        self.vector_store = MongoDBVectorStore(index_name)
        self.chunks_collection = mongo_client[MONGO_DATABASE_NAME]["chunks"]

    def __call__(self, state: RetrievalState):
        retrieved_chunks = self.vector_store.search(state["query"], k=self.num_retrieved_chunks)
        aggregated_docs = aggregate_by_url(retrieved_chunks)

        logger.info(
            "Retrieved {} documents, source urls: {}".format(len(retrieved_chunks), aggregated_docs.keys())
        )
        chunks_windows = self.get_chunks_windows(aggregated_docs)

        return {"retrieved_chunks": chunks_windows}

    @log_execution_time
    def get_chunks_windows(self, urls):
        """Returns chunks for specific sequence_numbers per URL (batched and deduplicated)."""
        retrieved_docs = {}

        for url, docs in urls.items():
            seq_numbers = set()
            for doc in docs:
                seq = doc.metadata["sequence_number"]
                window_range = range(seq - self.window_size, seq + self.window_size + 1)
                seq_numbers.update(window_range)

            query = {
                "metadata.url": url,
                "metadata.sequence_number": {"$in": list(seq_numbers)}
            }
            results = self.chunks_collection.find(query)

            seen = set()
            unique_docs = []
            for d in results:
                key = (d["metadata"]["url"], d["metadata"]["sequence_number"])
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(d)

            retrieved_docs[url] = sorted(unique_docs, key=lambda d: d["metadata"]["sequence_number"])

        return retrieved_docs


