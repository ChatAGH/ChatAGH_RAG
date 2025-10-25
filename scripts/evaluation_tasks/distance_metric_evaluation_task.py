from __future__ import annotations

import secrets
from typing import Any

from langchain_core.messages import HumanMessage
from opik import Prompt, track

from chat_agh.graph import ChatGraph
from chat_agh.utils.chat_history import ChatHistory
from chat_agh.utils.utils import logger
from scripts.consts import EVALUATION_MODEL


class DistanceMetricEvaluationTask:
    def __init__(
        self,
        prompts: dict[str, str],
        distance_metric: str,
        *,
        temperature: float = 1.0,
    ) -> None:
        self.prompts = [
            Prompt(name=prompt_name, prompt=prompt_template)
            for prompt_name, prompt_template in prompts.items()
        ]

        self._distance_metric = distance_metric
        self.temperature = temperature

        self.experiment_config = {
            "model": EVALUATION_MODEL,
            "temperature": self.temperature,
            "distance_metric": self._distance_metric,
        }
        self.task_name = (
            f"distance-metric-{self._distance_metric}-{secrets.token_hex(3)}"
        )

    def _prepare_chat_history(self, question: str) -> ChatHistory:
        return ChatHistory(messages=[HumanMessage(question)])

    @track(name="run()")  # type: ignore[misc]
    def run(self, input_data: dict[str, str]) -> dict[str, Any]:
        question = input_data.get("question")

        chat_graph = ChatGraph()
        chat_history = self._prepare_chat_history(question)  # type: ignore

        logger.info("Starting Opik run for distance metric '%s'", self._distance_metric)

        result_state = chat_graph.invoke_with_details(chat_history)
        response = result_state.get("response")
        retrieved_contexts = result_state.get("context", [])

        result: dict[str, Any] = {
            "question": question,
            "response": response,
            "retrieved_contexts": [
                context.metadata.get("url") for context in retrieved_contexts
            ],
            "distance_metric": self._distance_metric,
        }

        return result
