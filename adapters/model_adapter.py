"""
Адаптер модели: превращает текст поискового запроса в вектор.

Пока Участник 1 (ML-инженер) не согласовал финальную модель и размерность
эмбеддингов, ModelAdapter с use_mock=True возвращает детерминированный
псевдослучайный вектор заданной размерности (по умолчанию 384). Это
позволяет бесшовно тестировать весь бэкенд (API, Qdrant, бенчмарк) без
реальной модели: один и тот же текст всегда даёт один и тот же вектор.

Когда модель будет готова — достаточно передать use_mock=False (и, при
необходимости, свой model_name), не меняя ни строчки в api/ или database/.
"""

import hashlib
from typing import List, Optional

import numpy as np

from .base import BaseEmbedderAdapter


class ModelAdapter(BaseEmbedderAdapter):
    def __init__(
        self,
        vector_size: int = 384,
        use_mock: bool = True,
        model_name: Optional[str] = None,
    ) -> None:
        self.vector_size = vector_size
        self.use_mock = use_mock
        self.model_name = model_name
        self._model = None

        if not self.use_mock:
            self._load_real_model()

    def _load_real_model(self) -> None:
        """Ленивая загрузка реальной модели (sentence-transformers)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "use_mock=False, но пакет 'sentence-transformers' не установлен. "
                "Добавьте его в requirements.txt или используйте use_mock=True "
                "для работы с mock-векторами."
            ) from exc

        self._model = SentenceTransformer(self.model_name or "cointegrated/rubert-tiny2")
        actual_dim = self._model.get_sentence_embedding_dimension()
        if actual_dim != self.vector_size:
            raise ValueError(
                f"Размерность модели ({actual_dim}) не совпадает с ожидаемой "
                f"VECTOR_SIZE ({self.vector_size}). Обновите config.py."
            )

    def encode(self, text: str) -> List[float]:
        if self.use_mock:
            return self._mock_encode(text)
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if self.use_mock:
            return [self._mock_encode(text) for text in texts]
        vectors = self._model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        )
        return vectors.tolist()

    def _mock_encode(self, text: str) -> List[float]:
        """
        Детерминированный псевдослучайный вектор на основе хэша текста.
        Одинаковый текст -> одинаковый вектор (важно для повторяемых тестов
        и бенчмарков), но при этом векторы для разных текстов распределены
        по единичной сфере, как и настоящие эмбеддинги.
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)

        vector = rng.normal(loc=0.0, scale=1.0, size=self.vector_size)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.astype(np.float32).tolist()
