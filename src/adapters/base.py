"""
Абстрактные интерфейсы для стыковочного модуля (Adapter Pattern).

Бэкенд общается только с этими интерфейсами и не знает, приходят ли
векторы из реальной модели Участника 1 или из mock-генератора, и читаются
ли предрассчитанные эмбеддинги из .npy или из текстового файла.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List

import numpy as np


class BaseEmbedderAdapter(ABC):
    """Переводит текст поискового запроса в вектор эмбеддинга."""

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """Вернуть эмбеддинг одного текста."""
        raise NotImplementedError

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Вернуть эмбеддинги для списка текстов (батчем, эффективнее)."""
        raise NotImplementedError


class BaseVectorLoaderAdapter(ABC):
    """Читает предрассчитанные векторы датасета из файла на диске."""

    @abstractmethod
    def load(self, path: str) -> np.ndarray:
        """Загрузить весь файл целиком в память как np.ndarray (N, dim)."""
        raise NotImplementedError

    @abstractmethod
    def load_batches(self, path: str, batch_size: int) -> Iterator[np.ndarray]:
        """
        Потоково читать файл батчами по batch_size строк/векторов.
        Критично для датасета 500k+ объектов, чтобы не грузить всё в RAM.
        """
        raise NotImplementedError
