from hashlib import shake_256
from pathlib import Path

import opik
import pandas as pd
import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv
from opik import Opik
from opik.rest_api.core import ApiError

from chat_agh.utils.utils import logger
from scripts.consts import (
    DATASET_NAME,
    EMBEDDINGS_MODEL,
    EVALUATION_MODEL,
    PROJECT_NAME,
)
from scripts.evaluation_tasks.base import SearchParametersOverride
from scripts.evaluation_tasks.vector_search_evaluation_task import (
    VectorSearchEvaluationTask,
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
    dataset_file_path = Path(__file__).parent / "datasets" / f"{DATASET_NAME}.yaml"

    runner = ExperimentRunner(
        client=client,
        dataset_name=dataset_name,
        project_name=PROJECT_NAME,
        ragas_evaluator_model_name=EVALUATION_MODEL,
        criteria_evaluator_model_name=EVALUATION_MODEL,
        evaluator_embeddings_model_name=EMBEDDINGS_MODEL,
        dataset_path=dataset_file_path,
    )

    k_limit = [5, 6, 7, 8]
    fuzzy = True
    fuzzy_maxedits = [1, 2]
    fuzzy_prefix_length = [0, 1, 2]

    vector_search_settings = [
        SearchParametersOverride(),
        SearchParametersOverride(mode="dense"),
        SearchParametersOverride(mode="lexical"),
        SearchParametersOverride(k=8),
        SearchParametersOverride(num_candidates=80),
        SearchParametersOverride(fuzzy=False),
        SearchParametersOverride(vector_weight=1.2),
        SearchParametersOverride(text_weight=0.8),
        SearchParametersOverride(inner_limits={"dense": 20, "lexical": 20}),
        SearchParametersOverride(exact=True),
        SearchParametersOverride(dense_limit=12, lexical_limit=18),
        SearchParametersOverride(
            filter={"equals": {"path": "metadata.language", "value": "en"}}
        ),
    ]

    tasks = [
        VectorSearchEvaluationTask(
            collection_name="cluster_0",
            search_setting=setting,
        )
        for setting in vector_search_settings
    ]

    for task in tasks:
        logger.info(f"Running experiment: {task.task_name}")
        runner.run_experiment(task)
