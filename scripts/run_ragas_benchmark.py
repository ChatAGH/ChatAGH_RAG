from __future__ import annotations

import datetime as dt
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from chat_agh.graph import ChatGraph
from chat_agh.utils.chat_history import ChatHistory

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    faithfulness,
    context_precision,
)

# ======================
# BASE CONFIG
# ======================

BASE_CONFIG: Dict[str, Any] = {
    "yaml_path": "/Users/wnowogorski/PycharmProjects/ChatAGH_RAG/scripts/datasets/benchmark_dataset.yaml",
    "env_path": "/Users/wnowogorski/PycharmProjects/ChatAGH_RAG/.env",
    "out_dir": "/Users/wnowogorski/PycharmProjects/ChatAGH_RAG/benchmarking_results_4",
    "out_prefix": "results",  # final name will be: <out_prefix>_<experiment_name>.json
    "sample_size": None,
    # ChatGraph defaults (can be overridden per experiment)
    "enable_retrieval": False,
    "initial_retrieval_config": None,  # NEW: pass through to ChatGraph.__init__
    # Context handling
    "top_k_contexts": 100,
    "max_context_chars": 40000,
    "store_raw_state": False,
    # RAGAS
    "eval_llm_provider": "openai",
    "eval_model": os.getenv("RAGAS_EVAL_MODEL", "gpt-4o-mini"),
    "eval_embeddings_model": os.getenv(
        "RAGAS_EVAL_EMB_MODEL", "text-embedding-3-small"
    ),
    "eval_temperature": 0.0,
    "show_progress": True,
    # Parallelism
    "chat_workers": int(os.getenv("CHAT_WORKERS", "10")),
    "ragas_workers": int(os.getenv("RAGAS_WORKERS", "10")),
}

# ======================
# EXPERIMENTS
# ======================
EXPERIMENTS: List[Dict[str, Any]] = [
    # {
    #     "name": "Simple RAG",
    #     "enable_retrieval": False,
    #     "initial_retrieval_config": {
    #         "mode": "dense",
    #         "final_limit": 4,
    #         "dense_limit": 50,
    #         "num_dense_candidates": 100,
    #         "window": {"enabled": False},
    #         "augment": False,
    #     },
    # },
    # {
    #     "name": "Advanced retrieval",
    #     "enable_retrieval": False,
    #     "initial_retrieval_config": {
    #         "mode": "hybrid",
    #         "final_limit": 5,
    #
    #         "hybrid_candidate_limit": 400,
    #         "hybrid_num_candidates": 2000,
    #         "dense_weight": 1.0,
    #         "bm25_weight": 1.0,
    #         "rrf_k": 60,
    #
    #         "window": {
    #             "enabled": True,
    #             "window_before": 1,
    #             "window_after": 1,
    #             "group_fields": ["url"],
    #             "max_total_results": 50,
    #         },
    #         "augment": False,
    #     },
    # },
    #
    # {
    #     "name": "Advanced retrieval - 2",
    #     "enable_retrieval": False,
    #     "initial_retrieval_config": {
    #         "mode": "hybrid",
    #         "final_limit": 5,
    #
    #         "hybrid_candidate_limit": 400,
    #         "hybrid_num_candidates": 2000,
    #         "dense_weight": 0.8,
    #         "bm25_weight": 0.2,
    #         "rrf_k": 60,
    #
    #         "window": {
    #             "enabled": True,
    #             "window_before": 1,
    #             "window_after": 1,
    #             "group_fields": ["url"],
    #             "max_total_results": 50,
    #         },
    #         "augment": False,
    #     },
    # },
    #
    # {
    #     "name": "Advanced retrieval - 3",
    #     "enable_retrieval": False,
    #     "initial_retrieval_config": {
    #         "mode": "hybrid",
    #         "final_limit": 5,
    #
    #         "hybrid_candidate_limit": 400,
    #         "hybrid_num_candidates": 2000,
    #         "dense_weight": 0.3,
    #         "bm25_weight": 0.7,
    #         "rrf_k": 60,
    #
    #         "window": {
    #             "enabled": True,
    #             "window_before": 1,
    #             "window_after": 1,
    #             "group_fields": ["url"],
    #             "max_total_results": 50,
    #         },
    #         "augment": False,
    #     },
    # },
    #
    # {
    #     "name": "Advanced retrieval - 4",
    #     "enable_retrieval": False,
    #     "initial_retrieval_config": {
    #         "mode": "hybrid",
    #         "final_limit": 5,
    #
    #         "hybrid_candidate_limit": 400,
    #         "hybrid_num_candidates": 4000,
    #         "dense_weight": 1.0,
    #         "bm25_weight": 1.0,
    #         "rrf_k": 60,
    #
    #         "window": {
    #             "enabled": True,
    #             "window_before": 2,
    #             "window_after": 2,
    #             "group_fields": ["url"],
    #             "max_total_results": 50,
    #         },
    #         "augment": False,
    #     },
    # },
    #
    #
    # {
    #     "name": "Graph augmentation",
    #     "enable_retrieval": False,
    #     "initial_retrieval_config": {
    #         "mode": "hybrid",
    #         "final_limit": 5,
    #
    #         "hybrid_candidate_limit": 400,
    #         "hybrid_num_candidates": 2000,
    #         "dense_weight": 1.0,
    #         "bm25_weight": 1.0,
    #         "rrf_k": 60,
    #
    #         "window": {
    #             "enabled": True,
    #             "window_before": 2,
    #             "window_after": 2,
    #             "group_fields": ["url"],
    #             "max_total_results": 50,
    #         },
    #
    #         "augment": True,
    #         "augment_kwargs": {
    #             "per_doc_max_entities": 20,
    #             "per_doc_max_relationships": 20,
    #             "agg_max_entities": 40,
    #             "agg_max_relationships": 40,
    #
    #             "candidate_k": 800,
    #             "bm25_min_score": 0.0,
    #             "min_match_tokens": 1,
    #             "min_match_ratio": 0.9,
    #             "enable_subsequence": True,
    #         },
    #     },
    # },
    #
    # {
    #     "name": "Agentic RAG - 2",
    #     "enable_retrieval": True,
    #     "initial_retrieval_config": {
    #         "mode": "hybrid",
    #         "final_limit": 3,
    #
    #         "hybrid_candidate_limit": 400,
    #         "hybrid_num_candidates": 2000,
    #         "dense_weight": 1.0,
    #         "bm25_weight": 1.0,
    #         "rrf_k": 60,
    #
    #         "window": {
    #             "enabled": False,
    #             "window_before": 1,
    #             "window_after": 1,
    #             "group_fields": ["url"],
    #             "max_total_results": 50,
    #         },
    #
    #         "augment": False,
    #         "augment_kwargs": {
    #             "per_doc_max_entities": 30,
    #             "per_doc_max_relationships": 30,
    #             "agg_max_entities": 20,
    #             "agg_max_relationships": 20,
    #
    #             "candidate_k": 800,
    #             "bm25_min_score": 0.0,
    #             "min_match_tokens": 1,
    #             "min_match_ratio": 0.9,
    #             "enable_subsequence": True,
    #         },
    #     },
    # },
]

# ---------------------------
# Helpers
# ---------------------------


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if hasattr(obj, "page_content") and hasattr(obj, "metadata"):
        return {
            "page_content": _json_safe(getattr(obj, "page_content", "")),
            "metadata": _json_safe(getattr(obj, "metadata", {})),
        }
    return str(obj)


def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def _merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge: overrides keys on base copy."""
    out = dict(base)
    out.update(overrides)
    return out


# ---------------------------
# Data models
# ---------------------------


@dataclass
class BenchItem:
    question: str
    ground_truth: str
    evaluation_criteria: List[str]


@dataclass
class RunRow:
    question: str
    ground_truth: str
    evaluation_criteria: List[str]
    model_answer: str
    contexts: List[str]
    contexts_meta: List[Dict[str, Any]]

    retrieval_decision: Optional[bool] = None
    agents_queries: Optional[Any] = None
    num_search_tries: Optional[int] = None
    raw_state: Optional[Dict[str, Any]] = None

    ragas: Optional[Dict[str, Any]] = None


# ---------------------------
# Load YAML
# ---------------------------


def load_yaml_benchmark(path: str, sample_size: int | None = None) -> List[BenchItem]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if sample_size is not None:
        data = data[:sample_size]

    if not isinstance(data, list):
        raise ValueError("Benchmark YAML must be a list of items.")

    items: List[BenchItem] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Item #{i} is not a mapping/dict.")
        q = row.get("question")
        a = row.get("answer")
        crit = row.get("evaluation_criteria", []) or []

        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"Item #{i} missing/invalid 'question'.")
        if not isinstance(a, str) or not a.strip():
            raise ValueError(f"Item #{i} missing/invalid 'answer'.")
        if not isinstance(crit, list) or not all(isinstance(x, str) for x in crit):
            raise ValueError(
                f"Item #{i} invalid 'evaluation_criteria' (must be list[str])."
            )

        items.append(
            BenchItem(
                question=q.strip(), ground_truth=a.strip(), evaluation_criteria=crit
            )
        )
    return items


# ---------------------------
# Run system: parallel ChatGraph.invoke (per experiment)
# ---------------------------


def run_chatagh_parallel(
    items: List[BenchItem], cfg: Dict[str, Any]
) -> Tuple[List[RunRow], Dict[str, Any]]:
    """
    Runs ChatGraph in parallel.

    IMPORTANT: Graph is created per-thread and per-experiment run (thread-local inside this function),
    so different configs won't leak across experiments.
    """
    _thread_local = threading.local()

    def _get_thread_graph() -> ChatGraph:
        g = getattr(_thread_local, "graph", None)
        if g is None:
            g = ChatGraph(
                enable_retrieval=cfg["enable_retrieval"],
                initial_retrieval_config=cfg.get("initial_retrieval_config"),
            )
            _thread_local.graph = g
        return g

    # Usage accounting (kept like you had)
    from chat_agh.utils.model_inference import OpenAIModelInference

    start_usage = OpenAIModelInference().get_usage()

    top_k = int(cfg["top_k_contexts"])
    max_chars = int(cfg["max_context_chars"])

    def run_one(item: BenchItem) -> RunRow:
        chat_graph = _get_thread_graph()
        chat_history = ChatHistory(messages=[HumanMessage(item.question)])
        state = chat_graph.invoke(chat_history)

        model_answer = state.get("response", "")
        if not isinstance(model_answer, str):
            model_answer = str(model_answer)

        docs = state.get("retrieved_chunks") or []

        contexts: List[str] = []
        contexts_meta: List[Dict[str, Any]] = []

        limit = top_k if top_k and top_k > 0 else None
        for d in docs[:limit]:
            page = getattr(d, "page_content", str(d))
            meta = getattr(d, "metadata", {})
            contexts.append(_truncate(str(page), max_chars))
            contexts_meta.append(_json_safe(meta))

        return RunRow(
            question=item.question,
            ground_truth=item.ground_truth,
            evaluation_criteria=item.evaluation_criteria,
            model_answer=model_answer,
            contexts=contexts,
            contexts_meta=contexts_meta,
            retrieval_decision=state.get("retrieval_decision"),
            agents_queries=state.get("agents_queries"),
            num_search_tries=state.get("num_search_tries"),
            raw_state=_json_safe(state) if cfg["store_raw_state"] else None,
        )

    max_workers = int(cfg.get("chat_workers", 8))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        rows = list(ex.map(run_one, items))  # preserves order

    end_usage = OpenAIModelInference().get_usage()
    total_usage = {k: end_usage[k] - start_usage[k] for k in end_usage.keys()}

    return rows, total_usage


# ---------------------------
# RAGAS evaluation
# ---------------------------


def evaluate_with_ragas(rows: List[RunRow], cfg: Dict[str, Any]) -> Dict[str, Any]:
    ragas_rows = [
        {
            "question": r.question,
            "answer": r.model_answer,
            "contexts": r.contexts,
            "ground_truth": r.ground_truth,
        }
        for r in rows
    ]
    ds = Dataset.from_list(ragas_rows)

    if cfg["eval_llm_provider"].lower() == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        llm = ChatOpenAI(model=cfg["eval_model"], temperature=cfg["eval_temperature"])
        embs = OpenAIEmbeddings(model=cfg["eval_embeddings_model"])
    else:
        raise ValueError(f"Unsupported eval_llm_provider={cfg['eval_llm_provider']}")

    metrics = [answer_relevancy, answer_correctness, faithfulness, context_precision]

    eval_kwargs: Dict[str, Any] = dict(
        metrics=metrics,
        llm=llm,
        embeddings=embs,
        raise_exceptions=False,
        show_progress=cfg["show_progress"],
    )

    # Optional concurrency for ragas (API differs across versions)
    try:
        from ragas.run_config import RunConfig  # type: ignore

        eval_kwargs["run_config"] = RunConfig(
            max_workers=int(cfg.get("ragas_workers", 8))
        )
    except Exception:
        pass

    results = evaluate(ds, **eval_kwargs)

    df = results.to_pandas()
    score_rows = df.to_dict(orient="records")

    numeric_cols = [
        c
        for c in df.columns
        if c not in {"question", "answer", "contexts", "ground_truth"}
    ]
    aggregates: Dict[str, Any] = {}
    for c in numeric_cols:
        try:
            aggregates[c] = {
                "mean": float(df[c].mean()),
                "min": float(df[c].min()),
                "max": float(df[c].max()),
                "nan_count": int(df[c].isna().sum()),
            }
        except Exception:
            pass

    return {"score_rows": score_rows, "aggregates": aggregates}


# ---------------------------
# One experiment run
# ---------------------------


def run_experiment(items: List[BenchItem], cfg: Dict[str, Any]) -> Dict[str, Any]:
    rows, usage = run_chatagh_parallel(items, cfg)
    ragas_out = evaluate_with_ragas(rows, cfg)

    for r, scores in zip(rows, ragas_out["score_rows"]):
        r.ragas = {
            k: _json_safe(v)
            for k, v in scores.items()
            if k not in {"question", "answer", "contexts", "ground_truth"}
        }

    payload = {
        "config": {
            "created_at": dt.datetime.now().isoformat(),
            "yaml_path": os.path.abspath(cfg["yaml_path"]),
            "experiment_name": cfg["name"],
            "enable_retrieval": cfg["enable_retrieval"],
            "initial_retrieval_config": _json_safe(cfg.get("initial_retrieval_config")),
            "top_k_contexts": cfg["top_k_contexts"],
            "max_context_chars": cfg["max_context_chars"],
            "parallelism": {
                "chat_workers": cfg["chat_workers"],
                "ragas_workers": cfg["ragas_workers"],
            },
            "ragas": {
                "eval_llm_provider": cfg["eval_llm_provider"],
                "eval_model": cfg["eval_model"],
                "eval_embeddings_model": cfg["eval_embeddings_model"],
                "eval_temperature": cfg["eval_temperature"],
                "metrics": [
                    "answer_relevancy",
                    "answer_correctness",
                    "faithfulness",
                    "context_precision",
                ],
            },
        },
        "usage": usage,
        "rows": [_json_safe(asdict(r)) for r in rows],
        "aggregates": ragas_out["aggregates"],
    }
    return payload


def main() -> None:
    # Load env once
    if BASE_CONFIG.get("env_path"):
        load_dotenv(BASE_CONFIG["env_path"])

    # Load benchmark once
    items = load_yaml_benchmark(
        BASE_CONFIG["yaml_path"], sample_size=BASE_CONFIG["sample_size"]
    )
    print(f"Loaded benchmark items: {len(items)}")

    out_dir = Path(BASE_CONFIG["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for exp in EXPERIMENTS:
        while True:
            try:
                if "name" not in exp or not exp["name"]:
                    raise ValueError("Each experiment must include non-empty 'name'.")

                cfg = _merge_config(BASE_CONFIG, exp)

                payload = run_experiment(items, cfg)

                out_path = out_dir / f"{BASE_CONFIG['out_prefix']}_{cfg['name']}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                print(f"[{cfg['name']}] Saved: {out_path}")
                print(json.dumps(payload["aggregates"], ensure_ascii=False, indent=2))
                break
            except Exception as e:
                print(f"[{cfg['name']}] Failed: {e}")


if __name__ == "__main__":
    main()
