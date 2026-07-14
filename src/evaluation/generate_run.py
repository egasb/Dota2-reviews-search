import sys
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.core.config import settings
from src.utils.io import read_json, write_json


class VectorRetriever:
    """Computes Top-K retrieval results using tensor operations."""

    def __init__(self, top_k: int = 50) -> None:
        self.top_k = top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not settings.vectors_file.exists():
            logger.error(f"Vectors file not found: {settings.vectors_file}")
            sys.exit(1)

        logger.info("Loading embeddings and payload into RAM...")
        raw_embeddings = np.load(settings.vectors_file)
        self.doc_ids = [item["id"] for item in read_json(settings.payload_file)]

        logger.info(f"Initializing {settings.model_name} on {self.device}")
        self.model = SentenceTransformer(settings.model_name, device=self.device)

        logger.info("Transferring and normalizing document embeddings on GPU...")
        self.doc_tensors = torch.from_numpy(raw_embeddings).to(
            self.device, dtype=torch.float32
        )
        self.doc_tensors = F.normalize(self.doc_tensors, p=2, dim=1)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        del self.model
        del self.doc_tensors
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _compute_topk(
        self, query_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Matrix multiplication on device with a CPU fallback for OOM scenarios."""
        try:
            scores = torch.matmul(query_embeddings, self.doc_tensors.T)
            return torch.topk(scores, k=self.top_k, dim=1)
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM. Switching to CPU for matrix multiplication.")
            torch.cuda.empty_cache()
            scores = torch.matmul(query_embeddings.cpu(), self.doc_tensors.cpu().T)
            return torch.topk(scores, k=self.top_k, dim=1)

    def generate(self, run_output_path: Path) -> None:
        """Runs the generation pipeline and writes results to disk."""
        if not settings.validation_set_file.exists():
            logger.error("Validation set not found. Generate it first.")
            sys.exit(1)

        val_data = read_json(settings.validation_set_file)

        prefix = "query: " if "e5" in settings.model_name.lower() else ""
        formatted_queries = [f"{prefix}{item['query']}" for item in val_data]
        query_ids = [item["query_id"] for item in val_data]

        logger.info(f"Encoding {len(formatted_queries)} queries...")
        query_embeddings = self.model.encode(
            formatted_queries,
            batch_size=settings.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        top_scores, top_indices = self._compute_topk(query_embeddings)

        scores_np = top_scores.cpu().numpy()
        indices_np = top_indices.cpu().numpy()

        run_dict: dict[str, dict[str, float]] = {
            q_id: {
                self.doc_ids[idx]: float(score)
                for idx, score in zip(indices_np[i], scores_np[i])
            }
            for i, q_id in enumerate(query_ids)
        }

        write_json(run_output_path, run_dict)
        logger.success(f"Run file saved: {run_output_path}")


if __name__ == "__main__":
    safe_model_name = settings.model_name.replace("/", "_")
    output_path = settings.vectors_file.parent / f"run_{safe_model_name}.json"

    with VectorRetriever(top_k=50) as retriever:
        retriever.generate(output_path)
