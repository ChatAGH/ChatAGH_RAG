from typing import Sequence

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from opik import Opik, evaluate
from opik.evaluation.evaluation_result import EvaluationResult
from opik.integrations.langchain import OpikTracer
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision

from chat_agh.utils.utils import GEMINI_API_KEY
from scripts.consts import PROJECT_NAME
from scripts.evaluation_tasks.distance_metric_evaluation_task import (
    DistanceMetricEvaluationTask,
)
from scripts.metrics.answer_relevancy_metric import AnswerRelevancyWrapper
from scripts.metrics.base_metric import BaseMetricWrapper
from scripts.metrics.context_precision_metric import ContextPrecisionWrapper

opik_tracer = OpikTracer(project_name=PROJECT_NAME)


class ExperimentRunner:
    def __init__(
        self,
        client: Opik,
        dataset_name: str,
        project_name: str,
        criteria_evaluator_model_name: str,
        ragas_evaluator_model_name: str,
        evaluator_embeddings_model_name: str,
    ) -> None:
        self._client = client
        self._dataset = self._client.get_dataset(name=dataset_name)
        self.project_name = project_name
        self.criteria_evaluator_model_name = criteria_evaluator_model_name
        self.ragas_evaluator_model_name = ragas_evaluator_model_name
        self.evaluator_embeddings_model_name = evaluator_embeddings_model_name

    def configure_scoring_metrics(self) -> Sequence[BaseMetricWrapper]:
        emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        )

        evaluator_llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model=self.ragas_evaluator_model_name,
                api_key=GEMINI_API_KEY,
                temperature=1.0,
                callbacks=[opik_tracer],
            )
        )
        answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=emb)
        context_precision = ContextPrecision(llm=evaluator_llm)

        return [
            AnswerRelevancyWrapper(answer_relevancy),
            ContextPrecisionWrapper(context_precision),
        ]

    def run_experiment(
        self, evaluation_task: DistanceMetricEvaluationTask
    ) -> EvaluationResult:
        return evaluate(
            dataset=self._dataset,
            task=evaluation_task.run,
            scoring_metrics=self.configure_scoring_metrics(),
            experiment_config=evaluation_task.experiment_config,
            project_name=self.project_name,
            experiment_name=evaluation_task.task_name,
            prompts=evaluation_task.prompts,
        )
