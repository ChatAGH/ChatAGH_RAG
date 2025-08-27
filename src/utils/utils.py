from dataclasses import dataclass
import os
import time
import logging
import functools
from dotenv import load_dotenv

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer


load_dotenv("/Users/wnowogorski/PycharmProjects/ChatAGH_RAG/.env")

logger = logging.getLogger("chat_graph_logger")
logger.setLevel(logging.INFO)

def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        class_name = self.__class__.__name__
        fun_name = func.__name__
        start = time.time()
        result = func(self, *args, **kwargs)
        end = time.time()
        logger.info(f"[{class_name}.{fun_name}] Execution time: {end - start:.4f}s")
        return result
    return wrapper

mongo_client = MongoClient(os.environ.get("MONGODB_URI"), tlsAllowInvalidCertificates=True)
MONGO_DATABASE_NAME = "chat_agh"

embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")

@dataclass
class RetrievedContext:
    source_url: str
    chunks: list
    related_chunks: dict[str, list]

    @property
    def text(self):
        chunks_text = "\n".join(self.chunks)
        related_chunks_text = "\n".join(self.related_chunks)
        text = (
            f"Source URL: {self.source_url}\n"
            f"Retrieved chunks: {chunks_text}\n"
            f"Related context: {related_chunks_text}"
        )
        return text

