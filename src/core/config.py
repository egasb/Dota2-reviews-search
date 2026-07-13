from pathlib import Path
from pydantic_settings import BaseSettings

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
    batch_size: int = 1024

    # Paths
    raw_file: Path = DATA_DIR / "raw" / "reviews.jsonl"
    cursors_dir: Path = DATA_DIR / "raw" / ".cursors"
    interim_file: Path = DATA_DIR / "interim" / "filtered.jsonl"
    vectors_file: Path = DATA_DIR / "processed" / "vectors.npy"
    payload_file: Path = DATA_DIR / "processed" / "payload.json"


settings = Settings()

# Create directories
settings.raw_file.parent.mkdir(parents=True, exist_ok=True)
settings.cursors_dir.mkdir(parents=True, exist_ok=True)
settings.interim_file.parent.mkdir(parents=True, exist_ok=True)
settings.vectors_file.parent.mkdir(parents=True, exist_ok=True)
