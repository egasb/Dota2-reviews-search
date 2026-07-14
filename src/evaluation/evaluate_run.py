import sys
from pathlib import Path

import ir_measures
from ir_measures import MRR, nDCG, P, R
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.core.config import settings
from src.utils.io import read_json

console = Console()


def evaluate(run_file_path: Path) -> None:
    """Evaluates ranking metrics using ir_measures and presents a rich table."""
    if not settings.validation_set_file.exists():
        logger.error(f"Ground truth missing: {settings.validation_set_file}")
        sys.exit(1)

    if not run_file_path.exists():
        logger.error(f"Run file missing: {run_file_path}")
        sys.exit(1)

    logger.info("Loading Qrels (Ground Truth)...")
    qrels = [
        {
            "query_id": item["query_id"],
            "doc_id": item["relevant_doc_id"],
            "relevance": 1,
        }
        for item in read_json(settings.validation_set_file)
    ]

    logger.info(f"Loading Run (Predictions) from {run_file_path.name}...")
    run_dict = read_json(run_file_path)

    run = [
        {"query_id": q_id, "doc_id": doc_id, "score": score}
        for q_id, docs in run_dict.items()
        for doc_id, score in docs.items()
    ]

    metrics = [nDCG @ 5, nDCG @ 10, P @ 1, P @ 5, R @ 5, R @ 20, MRR]

    logger.info("Calculating aggregate metrics...")
    try:
        results = ir_measures.calc_aggregate(metrics, qrels, run)
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        sys.exit(1)

    table = Table(
        title=f"Retrieval Metrics: {settings.model_name}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="white", width=15)
    table.add_column("Score", justify="right", style="bold green", width=10)

    for metric in metrics:
        table.add_row(str(metric), f"{results[metric]:.4f}")

    console.print()
    console.print(table)
    logger.success("Evaluation pipeline completed.")


if __name__ == "__main__":
    safe_model_name = settings.model_name.replace("/", "_")
    target_run = settings.vectors_file.parent / f"run_{safe_model_name}.json"

    evaluate(target_run)
