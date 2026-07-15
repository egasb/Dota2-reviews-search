"""
Singleton-подключение к Qdrant.

Гарантирует, что на весь процесс (FastAPI-воркер, скрипт заливки,
бенчмарк) создаётся только один QdrantClient с одним пулом соединений,
а не по клиенту на каждый запрос/модуль.
"""

from typing import Optional

from qdrant_client import QdrantClient

from src.core.config import settings


class QdrantClientSingleton:
    _instance: Optional[QdrantClient] = None

    @classmethod
    def get_client(cls) -> QdrantClient:
        if cls._instance is None:
            cls._instance = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                grpc_port=settings.qdrant_grpc_port,
                prefer_grpc=settings.qdrant_prefer_grpc,
                timeout=settings.qdrant_timeout,
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Полезно в тестах: сбросить singleton и пересоздать соединение."""
        if cls._instance is not None:
            cls._instance.close()
        cls._instance = None

