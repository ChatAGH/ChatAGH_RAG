from src.states import RetrievalState
from src.utils.utils import mongo_client, MONGO_DATABASE_NAME, embedding_model


class ContextRetrieval:
    def __init__(self, num_chunks: int = 5):
        self.num_chunks = num_chunks
        self.graph_edges_collection = mongo_client[MONGO_DATABASE_NAME]["edges"]
        self.chunks_collection = mongo_client[MONGO_DATABASE_NAME]["chunks"]

    def __call__(self, state: RetrievalState):
        retrieved_chunks = state["retrieved_chunks"]
