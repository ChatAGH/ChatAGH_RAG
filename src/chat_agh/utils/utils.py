import functools
import logging
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

load_dotenv("/Users/wnowogorski/PycharmProjects/ChatAGH_RAG/.env")
logger = logging.getLogger("chat_graph_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        class_name = self.__class__.__name__
        fun_name = func.__name__
        start = time.time()
        result = func(self, *args, **kwargs)
        end = time.time()
        logger.info(
            f"[{class_name}.{fun_name}] Execution time: {end - start:.4f}s"
        )
        return result

    return wrapper


def retry_on_exception(attempts=3, delay=1, backoff=10, exception=Exception):
    """
    A decorator to retry a function call if it raises a specified exception.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exception as e:
                    if attempt == attempts:
                        raise
                    else:
                        logger.info(
                            f"Attempt {attempt} failed: {e}. Retrying in {current_delay} seconds..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff

        return wrapper

    return decorator


mongo_client: MongoClient = MongoClient(
    os.environ.get("MONGODB_URI"), tlsAllowInvalidCertificates=True
)
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
