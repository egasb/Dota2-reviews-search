"""
Создание двух коллекций Qdrant для сравнительного анализа:

  - dota2_flat       (Итерация 1, Baseline)  — точный (exact) поиск.
    Индекс HNSW используется как обычно, но при поиске (см.
    database/operations.py) выставляется SearchParams(exact=True), что
    заставляет Qdrant делать полный перебор (brute-force) вместо обхода
    графа — это и есть эталонный "точный" поиск для сравнения.

  - dota2_quantized  (Итерация 2, Optimized) — быстрый поиск с встроенным
    Scalar Quantization в INT8. Векторы дополнительно хранятся в
    квантованном виде (always_ram=True — квантованные векторы держатся в
    оперативной памяти для максимальной скорости), поиск идёт по HNSW
    с приближёнными (approximate) результатами.

Обе коллекции используют одинаковую размерность вектора и метрику
косинусного расстояния, чтобы сравнение было корректным.
"""

import time

from qdrant_client.models import (
    Distance,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

from src.core.config import settings
from src.database.client import QdrantClientSingleton


def _distance_from_config() -> Distance:
    mapping = {
        "Cosine": Distance.COSINE,
        "Euclid": Distance.EUCLID,
        "Dot": Distance.DOT,
    }
    return mapping.get(settings.distance_metric, Distance.COSINE)


def create_flat_collection() -> None:
    """Итерация 1 (Baseline): коллекция для точного поиска."""
    client = QdrantClientSingleton.get_client()
    if client.collection_exists(settings.collection_flat):
        return

    client.create_collection(
        collection_name=settings.collection_flat,
        vectors_config=VectorParams(
            size=settings.vector_size,
            distance=_distance_from_config(),
        ),
    )


def create_quantized_collection() -> None:
    """Итерация 2 (Optimized): коллекция со Scalar Quantization INT8."""
    client = QdrantClientSingleton.get_client()
    if client.collection_exists(settings.collection_quantized):
        return

    client.create_collection(
        collection_name=settings.collection_quantized,
        vectors_config=VectorParams(
            size=settings.vector_size,
            distance=_distance_from_config(),
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        ),
    )


def ensure_collections() -> None:
    """Идемпотентно создать обе коллекции, если их ещё нет."""
    create_flat_collection()
    create_quantized_collection()


def recreate_collections() -> None:
    """Полностью пересоздать обе коллекции (удаляет все данные!)."""
    client = QdrantClientSingleton.get_client()
    for name in (settings.collection_flat, settings.collection_quantized):
        if client.collection_exists(name):
            client.delete_collection(name)
    time.sleep(1.0)
    create_flat_collection()
    create_quantized_collection()

