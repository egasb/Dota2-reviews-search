from pathlib import Path
from pydantic_settings import BaseSettings
from src.utils.reproducibility import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    # Parser
    app_id: str = "570"
    language: str = "russian"
    num_per_page: int = 100

    # Filters
    min_text_len: int = 20
    cyrillic_ratio_threshold: float = 0.3
    min_vote_score: float = 0.48

    # Embeddings
    model_name: str = "intfloat/multilingual-e5-small"
    batch_size: int = 256

    # Paths
    raw_file: Path = DATA_DIR / "raw" / "reviews.jsonl"
    cursors_dir: Path = DATA_DIR / "raw" / ".cursors"
    interim_file: Path = DATA_DIR / "interim" / "filtered.jsonl"
    vectors_file: Path = DATA_DIR / "processed" / "vectors.npy"
    payload_file: Path = DATA_DIR / "processed" / "payload.json"

    # Validation set
    validation_size: int = 200
    gemma_model: str = "google/gemma-4-E4B-it"
    validation_set_file: Path = DATA_DIR / "interim"
    validation_set_file: Path = DATA_DIR / "interim" / "validation_set.json"
    load_in_4bit: bool = True

    # Infrastructure
    hf_token: str | None = None

    # Reproducibility
    seed: int = 20260505


settings = Settings()

# Create directories
settings.raw_file.parent.mkdir(parents=True, exist_ok=True)
settings.cursors_dir.mkdir(parents=True, exist_ok=True)
settings.interim_file.parent.mkdir(parents=True, exist_ok=True)
settings.vectors_file.parent.mkdir(parents=True, exist_ok=True)
settings.validation_set_file.parent.mkdir(parents=True, exist_ok=True)
set_seed(settings.seed)
