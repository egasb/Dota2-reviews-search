"""
Централизованная конфигурация проекта.

Все параметры можно переопределить через переменные окружения (например,
при запуске в Docker/CI), не трогая код. Значения по умолчанию рассчитаны
на локальный запуск через docker-compose.yml из корня репозитория.
"""

import os
from dataclasses import dataclass


def _get_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # --- Подключение к Qdrant ---
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_GRPC_PORT: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    PREFER_GRPC: bool = _get_bool("QDRANT_PREFER_GRPC", False)
    QDRANT_TIMEOUT: float = float(os.getenv("QDRANT_TIMEOUT", "30.0"))

    # --- Названия коллекций (Итерация 1 / Итерация 2) ---
    COLLECTION_FLAT: str = os.getenv("COLLECTION_FLAT", "dota2_flat")
    COLLECTION_QUANTIZED: str = os.getenv("COLLECTION_QUANTIZED", "dota2_quantized")

    # --- Параметры векторов ---
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "384"))
    DISTANCE_METRIC: str = os.getenv("DISTANCE_METRIC", "Cosine")

    # --- Пути к данным ---
    DATA_PATH: str = os.getenv("DATA_PATH", "data/raw_reviews.jsonl")
    VECTORS_PATH: str = os.getenv("VECTORS_PATH", "data/embeddings.npy")

    # --- Загрузка данных ---
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "5000"))

    # --- Адаптер модели (Участник 1 / ML-инженер ещё не отдал модель) ---
    # True -> ModelAdapter возвращает детерминированные mock-векторы.
    # False -> ModelAdapter пытается загрузить реальную sentence-transformers модель.
    USE_MOCK_EMBEDDER: bool = _get_bool("USE_MOCK_EMBEDDER", True)
    EMBEDDER_MODEL_NAME: str = os.getenv("EMBEDDER_MODEL_NAME", "cointegrated/rubert-tiny2")

    # --- API ---
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))


settings = Settings()
