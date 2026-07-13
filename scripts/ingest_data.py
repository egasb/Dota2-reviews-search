"""
Потоковая заливка датасета отзывов Dota 2 (500k+) в обе коллекции Qdrant.

Логика источника векторов:
  1. Если существует settings.VECTORS_PATH (data/embeddings.npy или .txt)
     и не передан флаг --no-precomputed — читаем предрассчитанные векторы
     через FileAdapter батчами, синхронно с батчами JSONL.
  2. Иначе — считаем векторы на лету через ModelAdapter (mock или реальная
     модель, в зависимости от settings.USE_MOCK_EMBEDDER).

Запуск:
    python scripts/ingest_data.py --recreate
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from adapters.file_adapter import FileAdapter  # noqa: E402
from adapters.model_adapter import ModelAdapter  # noqa: E402
from config import settings  # noqa: E402
from database.collections import ensure_collections, recreate_collections  # noqa: E402
from database.operations import batch_upsert  # noqa: E402


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_batches(path: str, batch_size: int) -> Iterator[List[dict]]:
    batch: List[dict] = []
    for record in iter_jsonl(path):
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_point_id(review_id: str) -> int:
    """
    ID точки в Qdrant должен быть int или UUID. Отзывы Steam приходят с
    числовым id в виде строки ("229761481") — конвертируем напрямую.
    Если вдруг встретится нечисловой id — используем детерминированный
    хэш как запасной вариант, не теряя объект.
    """
    try:
        return int(review_id)
    except ValueError:
        return abs(hash(review_id)) % (2**63)


def build_payload(record: dict) -> dict:
    return {
        "id": str(record["id"]),
        "text": record["text"],
        "voted_up": record["voted_up"],
        "votes_up": record["votes_up"],
        "votes_funny": record["votes_funny"],
        "comment_count": record["comment_count"],
        "weighted_vote_score": record["weighted_vote_score"],
        "timestamp_created": record["timestamp_created"],
        "timestamp_updated": record["timestamp_updated"],
        "playtime_hours": record["playtime_hours"],
        "playtime_at_review_hours": record["playtime_at_review_hours"],
        "num_games_owned": record["num_games_owned"],
        "num_reviews": record["num_reviews"],
    }


def run(recreate: bool, use_precomputed_vectors: bool) -> None:
    print("[ingest] Подготовка коллекций Qdrant...")
    if recreate:
        recreate_collections()
    else:
        ensure_collections()

    embedder = ModelAdapter(
        vector_size=settings.VECTOR_SIZE,
        use_mock=settings.USE_MOCK_EMBEDDER,
        model_name=settings.EMBEDDER_MODEL_NAME,
    )
    file_adapter = FileAdapter(vector_size=settings.VECTOR_SIZE)

    vectors_path = Path(settings.VECTORS_PATH)
    use_file_vectors = use_precomputed_vectors and vectors_path.exists()

    vector_batch_iter = None
    if use_file_vectors:
        print(f"[ingest] Используем предрассчитанные векторы из {vectors_path}")
        vector_batch_iter = file_adapter.load_batches(str(vectors_path), settings.BATCH_SIZE)
    else:
        source = "mock" if settings.USE_MOCK_EMBEDDER else "live-модель"
        print(f"[ingest] Предрассчитанные векторы не найдены — считаем на лету ({source})")

    total_inserted = 0
    for batch in read_batches(settings.DATA_PATH, settings.BATCH_SIZE):
        ids = [build_point_id(record["id"]) for record in batch]
        payloads = [build_payload(record) for record in batch]

        if use_file_vectors:
            try:
                vectors = next(vector_batch_iter)  # type: ignore[arg-type]
            except StopIteration as exc:
                raise RuntimeError(
                    "Векторов в файле меньше, чем отзывов в JSONL — "
                    "проверьте, что файлы синхронизированы."
                ) from exc
            if vectors.shape[0] != len(batch):
                raise RuntimeError(
                    f"Несовпадение размера батча: {vectors.shape[0]} векторов "
                    f"против {len(batch)} отзывов. Файлы рассинхронизированы."
                )
        else:
            texts = [record["text"] for record in batch]
            vectors = np.array(embedder.encode_batch(texts), dtype=np.float32)

        batch_upsert(settings.COLLECTION_FLAT, ids, vectors, payloads, settings.BATCH_SIZE)
        batch_upsert(settings.COLLECTION_QUANTIZED, ids, vectors, payloads, settings.BATCH_SIZE)

        total_inserted += len(batch)
        print(f"[ingest] Загружено {total_inserted} отзывов...")

    print(f"[ingest] Готово. Всего загружено {total_inserted} отзывов в обе коллекции.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Заливка датасета отзывов Dota 2 в Qdrant")
    parser.add_argument(
        "--recreate", action="store_true", help="Пересоздать коллекции перед заливкой (удалит данные)"
    )
    parser.add_argument(
        "--no-precomputed",
        action="store_true",
        help="Игнорировать файл с готовыми векторами и всегда использовать ModelAdapter",
    )
    args = parser.parse_args()

    run(recreate=args.recreate, use_precomputed_vectors=not args.no_precomputed)
