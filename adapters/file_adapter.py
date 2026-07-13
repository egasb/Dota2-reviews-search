"""
Адаптер файлов: читает предрассчитанные векторы датасета с диска.

Поддерживает два формата:
  - .npy   — бинарный numpy-массив формы (N, vector_size). Читается через
             mmap, поэтому 500k+ векторов не грузятся в RAM целиком.
  - текст  — по одному вектору на строку, значения через запятую
             (например, выгрузка из pandas/CSV без заголовка).

Формат выбирается автоматически по расширению файла.
"""

from pathlib import Path
from typing import Iterator, List

import numpy as np

from .base import BaseVectorLoaderAdapter


class FileAdapter(BaseVectorLoaderAdapter):
    def __init__(self, vector_size: int = 384) -> None:
        self.vector_size = vector_size

    def load(self, path: str) -> np.ndarray:
        path_obj = self._resolve_path(path)
        if path_obj.suffix == ".npy":
            return np.load(path_obj)
        return self._load_text_full(path_obj)

    def load_batches(self, path: str, batch_size: int = 5000) -> Iterator[np.ndarray]:
        path_obj = self._resolve_path(path)
        if path_obj.suffix == ".npy":
            yield from self._load_npy_batches(path_obj, batch_size)
        else:
            yield from self._load_text_batches(path_obj, batch_size)

    # --- внутренние методы ---

    @staticmethod
    def _resolve_path(path: str) -> Path:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Файл с векторами не найден: {path}")
        return path_obj

    def _load_npy_batches(self, path: Path, batch_size: int) -> Iterator[np.ndarray]:
        # mmap_mode="r" — файл не грузится в RAM целиком, страницы читаются лениво.
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] != self.vector_size:
            raise ValueError(
                f"Ожидалась форма (N, {self.vector_size}), получено {array.shape}"
            )
        total = array.shape[0]
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            yield np.array(array[start:end], dtype=np.float32)

    def _load_text_full(self, path: Path) -> np.ndarray:
        vectors = list(self._iter_text_vectors(path))
        return np.array(vectors, dtype=np.float32)

    def _load_text_batches(self, path: Path, batch_size: int) -> Iterator[np.ndarray]:
        batch = []
        for vector in self._iter_text_vectors(path):
            batch.append(vector)
            if len(batch) >= batch_size:
                yield np.array(batch, dtype=np.float32)
                batch = []
        if batch:
            yield np.array(batch, dtype=np.float32)

    def _iter_text_vectors(self, path: Path) -> Iterator[List[float]]:
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                values = [float(x) for x in line.split(",")]
                if len(values) != self.vector_size:
                    raise ValueError(
                        f"{path}:{line_number} — ожидалось {self.vector_size} "
                        f"значений, получено {len(values)}"
                    )
                yield values

