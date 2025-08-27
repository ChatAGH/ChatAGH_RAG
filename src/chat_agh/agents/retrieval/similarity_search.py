from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document

from chat_agh.states import RetrievalState
from chat_agh.vector_store.mongodb import MongoDBVectorStore
from chat_agh.utils.utils import logger, log_execution_time, mongo_client, MONGO_DATABASE_NAME
from chat_agh.agents.retrieval.utils import aggregate_by_url


class SimilaritySearch:
    def __init__(self, index_name, num_retrieved_chunks: int = 5, window_size: int = 1):
        self.num_retrieved_chunks = num_retrieved_chunks
        self.window_size = window_size
        self.vector_store = MongoDBVectorStore(index_name)
        self.chunks_collection = mongo_client[MONGO_DATABASE_NAME]["chunks"]

    @log_execution_time
    def __call__(self, state: RetrievalState):
        retrieved_chunks = self.vector_store.search(state["query"], k=self.num_retrieved_chunks)
        aggregated_docs = aggregate_by_url(retrieved_chunks)

        logger.info(
            "Retrieved {} documents, source urls: {}".format(len(retrieved_chunks), aggregated_docs.keys())
        )
        chunks_windows = self.get_chunks_windows(aggregated_docs)

        return {"retrieved_chunks": chunks_windows}

    def get_chunks_windows(self, urls):
        """Returns chunks for specific sequence_numbers per URL (batched and deduplicated)."""
        retrieved_docs = {}

        def process_url(url, docs):
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

            unique_docs = [
                Document(page_content=d["text"], metadata=d["metadata"])
                for d in unique_docs
            ]
            return url, sorted(unique_docs, key=lambda d: d.metadata["sequence_number"])

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_url, url, docs)
                for url, docs in urls.items()
            ]
            for future in as_completed(futures):
                url, docs = future.result()
                retrieved_docs[url] = docs

        return retrieved_docs


