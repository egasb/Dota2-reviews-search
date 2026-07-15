"""
Операции с данными: пакетная вставка и поиск с фильтрацией.
"""

from typing import List, Optional, Sequence

import numpy as np
from qdrant_client.models import (
    FieldCondition,
    Filter,
    PointStruct,
    Range,
    ScoredPoint,
    SearchParams,
)

from src.core.config import settings
from src.database.client import QdrantClientSingleton


def batch_upsert(
    collection_name: str,
    ids: Sequence[int],
    vectors: np.ndarray,
    payloads: Sequence[dict],
    batch_size: int = settings.batch_size,
) -> int:
    """
    Пакетная вставка точек в коллекцию батчами по batch_size (по умолчанию
    5000), чтобы не перегружать один HTTP/gRPC запрос на датасете 500k+.
    Возвращает суммарное количество вставленных точек.
    """
    if len(ids) != len(vectors) or len(ids) != len(payloads):
        raise ValueError("ids, vectors и payloads должны быть одинаковой длины")

    client = QdrantClientSingleton.get_client()
    total = len(ids)
    inserted = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        points = [
            PointStruct(
                id=ids[i],
                vector=np.asarray(vectors[i], dtype=np.float32).tolist(),
                payload=payloads[i],
            )
            for i in range(start, end)
        ]
        client.upsert(collection_name=collection_name, points=points, wait=True)
        inserted += len(points)

    return inserted


def build_playtime_filter(
    min_hours: Optional[float], max_hours: Optional[float]
) -> Optional[Filter]:
    """Построить фильтр Qdrant по полю playtime_hours (диапазон)."""
    if min_hours is None and max_hours is None:
        return None

    return Filter(
        must=[
            FieldCondition(
                key="playtime_hours",
                range=Range(gte=min_hours, lte=max_hours),
            )
        ]
    )


def search(
    collection_name: str,
    query_vector: List[float],
    top_k: int = 10,
    min_playtime: Optional[float] = None,
    max_playtime: Optional[float] = None,
    exact: bool = False,
) -> List[ScoredPoint]:
    """
    Similarity search с опциональной фильтрацией по playtime_hours.

    exact=True  -> точный (brute-force) поиск — используется для коллекции
                   dota2_flat (Итерация 1, Baseline).
    exact=False -> приближённый поиск через HNSW (+ квантование, если оно
                   включено у коллекции) — используется для dota2_quantized
                   (Итерация 2, Optimized).
    """
    client = QdrantClientSingleton.get_client()
    query_filter = build_playtime_filter(min_playtime, max_playtime)

    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        search_params=SearchParams(exact=exact),
        with_payload=True,
    )
    return result.points

