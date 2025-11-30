from __future__ import annotations

import secrets
from typing import Any, Optional

from opik import Prompt, track

from chat_agh.utils.utils import logger
from chat_agh.vector_store.mongodb import MongoDBVectorStore
from scripts.evaluation_tasks.base import CurrentSearchParameters


class VectorSearchEvaluationTask:
    """Run an evaluation round with custom MongoDB vector search parameters."""

    def __init__(
        self,
        *,
        collection_name: str,
        search_setting: CurrentSearchParameters,
        prompts: Optional[dict[str, str]] = None,
        similarity: str = "cosine",
        vector_index_name: str = "vector_index",
        search_index_name: str = "default",
    ) -> None:
        prompts_mapping = prompts or {}
        self.prompts = [
            Prompt(name=prompt_name, prompt=prompt_template)
            for prompt_name, prompt_template in prompts_mapping.items()
        ]
        self._collection_name = collection_name
        self._search_setting = search_setting
        self._distance_metric = "cosine"
        self._vector_store = MongoDBVectorStore(
            collection_name=collection_name,
            vector_index_name=vector_index_name,
            search_index_name=search_index_name,
            similarity=similarity,
            create_indexes=False,
        )

        suffix_parts = [
            f"mode={search_setting.mode}",
            f"k={search_setting.k}",
            f"num_candidates={search_setting.num_candidates}",
            f"fuzzy={'on' if search_setting.fuzzy else 'off'}",
            f"weights={search_setting.vector_weight}-{search_setting.text_weight}",
        ]
        if search_setting.inner_limits:
            limits_repr = "-".join(
                f"{key}:{value}" for key, value in search_setting.inner_limits.items()
            )
            suffix_parts.append(f"limits={limits_repr}")
        if search_setting.dense_limit is not None:
            suffix_parts.append(f"dense_limit={search_setting.dense_limit}")
        if search_setting.lexical_limit is not None:
            suffix_parts.append(f"lexical_limit={search_setting.lexical_limit}")

        suffix = ",".join(suffix_parts)
        unique_suffix = secrets.token_hex(3)
        self.task_name = (
            f"vector-search::{self._collection_name}::{suffix}::{unique_suffix}"
        )
        self.experiment_config = {
            "collection": self._collection_name,
            "similarity": similarity,
            **search_setting.to_dict(),
            "distance_metric": self._distance_metric,
        }
        self._similarity = similarity

    @track(name="vector_search_run()")
    def run(self, input_data: dict[str, str]) -> dict[str, Any]:
        question = input_data.get("question")
        if not question:
            raise ValueError("Dataset entry does not contain a 'question' field")
        logger.info(
            "Executing vector search on '%s' with params %s",
            self._collection_name,
            self._search_setting.to_dict(),
        )

        documents = self._search_setting.run(self._vector_store, question)

        serialized_context = [
            {
                "url": doc.metadata.get("url"),
                "score": doc.metadata.get("score"),
                "sequence_number": doc.metadata.get("sequence_number"),
            }
            for doc in documents
        ]

        response_text = "\n".join(doc.page_content for doc in documents)
        retrieved_contexts = [doc.page_content for doc in documents]

        return {
            "question": question,
            "response": response_text,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_context_metadata": serialized_context,
            "search_parameters": self._search_setting.to_dict(),
            "distance_metric": self._distance_metric,
            "similarity": self._similarity,
        }
