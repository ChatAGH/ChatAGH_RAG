from typing import Mapping

from opik import track
from opik.evaluation.metrics.score_result import ScoreResult
from ragas.metrics import AnswerRelevancy

from scripts.consts import PROJECT_NAME
from scripts.metrics.base_metric import BaseMetricWrapper


class AnswerRelevancyWrapper(BaseMetricWrapper):
    def __init__(self, metric: AnswerRelevancy) -> None:
        super().__init__(
            metric,
            "response_relevancy_metric",
        )

    @track(project_name=PROJECT_NAME, name="answer_relevancy_metric")  # type: ignore[misc]
    def score(
        self,
        question: str,
        response: str,
        retrieved_contexts: list[str],
        **_: Mapping[str, str | list[str]],
    ) -> ScoreResult:
        row = {
            "user_input": question,
            "response": response,
            "context": retrieved_contexts,
        }
        return self.base_score(row)
