# FILE: src/evaluation/spot_check.py

import random
from loguru import logger
from src.core.config import settings
from src.utils.io import read_json


def run_spot_check() -> None:
    safe_model_name = settings.model_name.replace("/", "_")
    run_file = settings.vectors_file.parent / f"run_hybrid_{safe_model_name}.json"

    if not run_file.exists():
        logger.error(f"Run file not found: {run_file}")
        return

    validation_data = read_json(settings.validation_set_file)
    run_data = read_json(run_file)
    payload = read_json(settings.payload_file)

    doc_id_to_text = {str(item["id"]): item["text"] for item in payload}

    query_map = {
        item["query_id"]: (item["query"], str(item["relevant_doc_id"]))
        for item in validation_data
    }

    available_q_ids = [q_id for q_id in query_map if q_id in run_data]
    if not available_q_ids:
        logger.warning("No validation queries found in the run file.")
        return

    # Выбираем случайные 5 запросов
    sampled_q_ids = random.sample(available_q_ids, min(5, len(available_q_ids)))

    for q_id in sampled_q_ids:
        query, ground_truth_id = query_map[q_id]
        retrieved = run_data[q_id]  # Словарь {doc_id: rrf_score}
        sorted_retrieved = sorted(retrieved.items(), key=lambda x: x[1], reverse=True)[
            :3
        ]

        print("\n" + "=" * 100)
        print(f"ЗАПРОС: '{query}'")
        print(f"ЭТАЛОННЫЙ ID: {ground_truth_id}")
        print(
            f"ЭТАЛОННЫЙ ТЕКСТ: {doc_id_to_text.get(ground_truth_id, 'Не найден')[:150]}..."
        )
        print("-" * 100)
        print("ЧТО ВЫДАЛ ПОИСК (TOP-3):")

        for idx, (doc_id, score) in enumerate(sorted_retrieved, start=1):
            text = doc_id_to_text.get(doc_id, "Текст не найден")
            is_gt = " [СОВПАДЕНИЕ С ЭТАЛОНОМ]" if doc_id == ground_truth_id else ""
            print(f"  {idx}. [RRF Score: {score:.5f}] ID: {doc_id}{is_gt}")
            print(f"     Текст: {text[:220]}...\n")
        print("=" * 100)


if __name__ == "__main__":
    run_spot_check()
