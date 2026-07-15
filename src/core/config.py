"""
Централизованная конфигурация проекта.

Все параметры можно переопределить через переменные окружения (например,
в Docker/CI или через файл .env), не трогая код. Pydantic автоматически
смаппит заглавные переменные окружения (QDRANT_HOST) в атрибуты (qdrant_host).
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.utils.reproducibility import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    # Настройки Pydantic: чтение из .env файла и игнорирование лишних переменных
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Подключение к Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = False
    qdrant_timeout: float = 30.0

    # --- Названия коллекций ---
    collection_flat: str = "dota2_flat"
    collection_quantized: str = "dota2_quantized"
    
    # --- Параметры векторов ---
    vector_size: int = 312
    distance_metric: str = "Cosine"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Парсер (Steam) ---
    app_id: str = "570"
    language: str = "russian"
    num_per_page: int = 100

    # --- Фильтры ---
    min_text_len: int = 20
    cyrillic_ratio_threshold: float = 0.3
    min_vote_score: float = 0.48

    # --- Модели и Эмбеддинги ---
    # True -> ModelAdapter возвращает детерминированные mock-векторы.
    # False -> Загружает реальную sentence-transformers модель.
    use_mock_embedder: bool = True
    model_name: str = "intfloat/multilingual-e5-small"
    batch_size: int = 256

    # --- Структура путей к данным ---
    raw_file: Path = DATA_DIR / "raw" / "reviews.jsonl"
    cursors_dir: Path = DATA_DIR / "raw" / ".cursors"
    interim_file: Path = DATA_DIR / "interim" / "filtered.jsonl"
    vectors_file: Path = DATA_DIR / "processed" / "vectors.npy"
    payload_file: Path = DATA_DIR / "processed" / "payload.json"

    # --- Валидационный датасет и LLM ---
    validation_size: int = 200
    validation_set_file: Path = DATA_DIR / "interim" / "validation_set.json"
    gemma_model: str = "google/gemma-4-E4B-it"
    load_in_4bit: bool = True

    # --- Инфраструктура ---
    hf_token: str | None = None

    # --- Воспроизводимость ---
    seed: int = 20260505


# Инициализация настроек
settings = Settings()

# Создание необходимых директорий при импорте настроек
settings.raw_file.parent.mkdir(parents=True, exist_ok=True)
settings.cursors_dir.mkdir(parents=True, exist_ok=True)
settings.interim_file.parent.mkdir(parents=True, exist_ok=True)
settings.vectors_file.parent.mkdir(parents=True, exist_ok=True)

# Фиксация seed'а для воспроизводимости
set_seed(settings.seed)