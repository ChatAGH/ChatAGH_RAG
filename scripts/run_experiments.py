from hashlib import shake_256
from pathlib import Path

import opik
import pandas as pd
import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv
from opik import Opik
from opik.rest_api.core import ApiError

from chat_agh.prompts import (
    GENERATION_PROMPT_TEMPLATE,
    SUMMARY_GENERATION_PROMPT_TEMPLATE,
    SUPERVISOR_AGENT_PROMPT_TEMPLATE,
)
from chat_agh.utils import logger
from scripts.consts import (
    DATASET_NAME,
    EMBEDDINGS_MODEL,
    EVALUATION_MODEL,
    PROJECT_NAME,
)
from scripts.evaluation_tasks.distance_metric_evaluation_task import (
    DistanceMetricEvaluationTask,
)
from scripts.experiment_runner import ExperimentRunner

opik.configure(url="http://localhost:5173", use_local=True)


def _content_hash(path: Path) -> str:
    length = 8
    with path.open("rb") as f:
        data = f.read()
    return shake_256(data).hexdigest(length // 2)


def _yaml_to_df(path: Path) -> pd.DataFrame:
    logger.info(f"Processing {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    return pd.DataFrame(data)


def _create_opik_dataset(client: Opik, dataset_name: str, dataset_path: Path) -> None:
    df = _yaml_to_df(dataset_path)
    logger.info(f"Inserting {dataset_name} to Opik")
    try:
        dataset = client.create_dataset(name=dataset_name)
        dataset.insert_from_pandas(dataframe=df)
    except ApiError:
        logger.info(f"{dataset_name} already uploaded to Opik")


def _prepare_opik_dataset(client: Opik, dataset_name: str) -> str:
    dataset_path = Path(__file__).parent / "datasets" / (dataset_name + ".yaml")

    dataset_id = _content_hash(dataset_path)

    opik_dataset_name = f"{dataset_name}_{dataset_id}"

    _create_opik_dataset(client, opik_dataset_name, dataset_path)

    return opik_dataset_name


if __name__ == "__main__":
    load_dotenv()

    client = Opik(project_name="chat_agh_rag")
    dataset_name = _prepare_opik_dataset(client, DATASET_NAME)
    runner = ExperimentRunner(
        client=client,
        dataset_name=dataset_name,
        project_name=PROJECT_NAME,
        ragas_evaluator_model_name=EVALUATION_MODEL,
        criteria_evaluator_model_name=EVALUATION_MODEL,
        evaluator_embeddings_model_name=EMBEDDINGS_MODEL,
    )

    distance_metrics = ["cosine", "dot", "l2"]
    tasks = [
        DistanceMetricEvaluationTask(
            prompts={
                "SUPERVISOR_AGENT_PROMPT": SUPERVISOR_AGENT_PROMPT_TEMPLATE,
                "SUMMARY_GENERATION": SUMMARY_GENERATION_PROMPT_TEMPLATE,
                "GENERATION_PROMPT": GENERATION_PROMPT_TEMPLATE,
            },
            distance_metric=metric,
        )
        for metric in distance_metrics
    ]

    for task in tasks:
        logger.info(f"Running experiment: {task.task_name}")
        runner.run_experiment(task)
