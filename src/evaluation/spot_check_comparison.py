# FILE: src/evaluation/spot_check_comparison.py

import sys
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.core.config import settings
from src.utils.io import read_json

try:
    import bm25s
except ImportError:
    logger.error("bm25s is not installed. Run 'uv add bm25s PyStemmer' first.")
    sys.exit(1)

type ScoredDocs = dict[str, float]


class RetrievalComparator:
    """Computes and compares Dense, Sparse, RRF, and Weighted Fusion retrieval paths."""

    def __init__(self, top_k: int = 3, candidate_pool_size: int = 100) -> None:
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not settings.vectors_file.exists():
            logger.error(f"Vectors file not found: {settings.vectors_file}")
            sys.exit(1)

        # Load resources
        self.raw_embeddings = np.load(settings.vectors_file)
        payload_data = read_json(settings.payload_file)
        self.doc_ids = [str(item["id"]) for item in payload_data]
        self.corpus = [item["text"] for item in payload_data]
        self.doc_id_to_text = dict(zip(self.doc_ids, self.corpus))

        # Dense model initialization
        self.model = SentenceTransformer(settings.model_name, device=self.device)
        self.doc_tensors = torch.from_numpy(self.raw_embeddings).to(
            self.device, dtype=torch.float32
        )
        self.doc_tensors = F.normalize(self.doc_tensors, p=2, dim=1)

        # Sparse model initialization
        self.stemmer = None
        try:
            import Stemmer

            self.stemmer = Stemmer.Stemmer("russian")
        except ImportError:
            logger.warning("PyStemmer not found. Stemming is disabled.")

        logger.info("Indexing BM25S...")
        corpus_tokens = bm25s.tokenize(
            self.corpus, stopwords="ru", stemmer=self.stemmer
        )
        self.sparse_retriever = bm25s.BM25()
        self.sparse_retriever.index(corpus_tokens)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        del self.model
        del self.doc_tensors
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _normalize_scores(self, scored_docs: ScoredDocs) -> ScoredDocs:
        """Applies Min-Max normalization to map scores strictly to the [0, 1] range."""
        if not scored_docs:
            return {}
        scores = list(scored_docs.values())
        min_score = min(scores)
        max_score = max(scores)
        denom = max_score - min_score
        if denom < 1e-9:
            return {doc_id: 1.0 for doc_id in scored_docs}
        return {
            doc_id: (score - min_score) / denom for doc_id, score in scored_docs.items()
        }

    def _apply_rrf(
        self, dense_ids: list[str], sparse_ids: list[str], k: int = 60
    ) -> ScoredDocs:
        """Computes Reciprocal Rank Fusion scores."""
        rrf_scores: ScoredDocs = {}
        for rank, doc_id in enumerate(dense_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        for rank, doc_id in enumerate(sparse_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return dict(
            sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        )

    def _apply_weighted_fusion(
        self, dense_scores: ScoredDocs, sparse_scores: ScoredDocs, alpha: float = 0.7
    ) -> ScoredDocs:
        """Combines normalized dense and sparse scores with a priority factor alpha."""
        norm_dense = self._normalize_scores(dense_scores)
        norm_sparse = self._normalize_scores(sparse_scores)

        fused_scores: ScoredDocs = {}
        all_doc_ids = set(norm_dense.keys()) | set(norm_sparse.keys())

        for doc_id in all_doc_ids:
            d_score = norm_dense.get(doc_id, 0.0)
            s_score = norm_sparse.get(doc_id, 0.0)
            fused_scores[doc_id] = alpha * d_score + (1.0 - alpha) * s_score

        return dict(
            sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        )

    def compare(self, query: str, ground_truth_id: str, alpha: float = 0.7) -> None:
        """Executes and prints comparison of all retrieval strategies for a single query."""
        # 1. Dense retrieval
        prefix = "query: " if "e5" in settings.model_name.lower() else ""
        query_emb = self.model.encode(
            f"{prefix}{query}", convert_to_tensor=True, normalize_embeddings=True
        )
        scores = torch.matmul(query_emb, self.doc_tensors.T)
        top_dense_scores, top_dense_indices = torch.topk(
            scores, k=self.candidate_pool_size
        )

        dense_scores_np = top_dense_scores.cpu().numpy()
        dense_indices_np = top_dense_indices.cpu().numpy()

        dense_map: ScoredDocs = {
            self.doc_ids[idx]: float(score)
            for idx, score in zip(dense_indices_np, dense_scores_np)
        }
        dense_top_list = list(dense_map.keys())

        # 2. Sparse retrieval
        query_tokens = bm25s.tokenize([query], stemmer=self.stemmer)
        results = self.sparse_retriever.retrieve(
            query_tokens,
            corpus=self.doc_ids,
            k=self.candidate_pool_size,
            return_as="tuple",
        )
        sparse_map: ScoredDocs = {
            doc_id: float(score)
            for doc_id, score in zip(results.documents[0], results.scores[0])
        }
        sparse_top_list = list(sparse_map.keys())

        # 3. Fusions
        rrf_results = self._apply_rrf(dense_top_list, sparse_top_list)
        weighted_results = self._apply_weighted_fusion(
            dense_map, sparse_map, alpha=alpha
        )

        # Print layout
        gt_text = self.doc_id_to_text.get(ground_truth_id, "Not found")
        print("\n" + "=" * 120)
        print(f"ЗАПРОС: '{query}'")
        print(f"ЭТАЛОН: ID {ground_truth_id} | Текст: {gt_text[:120]}...")
        print("=" * 120)

        # Print Dense
        print("\n[1] ЧИСТЫЙ СЕМАНТИЧЕСКИЙ ПОИСК (Dense E5-Small):")
        for i, doc_id in enumerate(dense_top_list[: self.top_k], start=1):
            text = self.doc_id_to_text.get(doc_id, "Not found")
            is_match = " [GT MATCH]" if doc_id == ground_truth_id else ""
            print(f"  {i}. [Score: {dense_map[doc_id]:.4f}] ID: {doc_id}{is_match}")
            print(f"     Текст: {text[:150]}...")

        # Print Sparse
        print("\n[2] ЧИСТЫЙ ЛЕКСИЧЕСКИЙ ПОИСК (Sparse BM25S):")
        for i, doc_id in enumerate(sparse_top_list[: self.top_k], start=1):
            text = self.doc_id_to_text.get(doc_id, "Not found")
            is_match = " [GT MATCH]" if doc_id == ground_truth_id else ""
            print(f"  {i}. [Score: {sparse_map[doc_id]:.4f}] ID: {doc_id}{is_match}")
            print(f"     Текст: {text[:150]}...")

        # Print RRF
        print("\n[3] СТАНДАРТНЫЙ РАНГОВЫЙ ГИБРИД (RRF):")
        for i, (doc_id, score) in enumerate(rrf_results.items(), start=1):
            text = self.doc_id_to_text.get(doc_id, "Not found")
            is_match = " [GT MATCH]" if doc_id == ground_truth_id else ""
            print(f"  {i}. [RRF Score: {score:.5f}] ID: {doc_id}{is_match}")
            print(f"     Текст: {text[:150]}...")

        # Print Weighted
        print(f"\n[4] ВЗВЕШЕННЫЙ СКОРИНГОВЫЙ ГИБРИД (Weighted Fusion, Alpha={alpha}):")
        for i, (doc_id, score) in enumerate(weighted_results.items(), start=1):
            text = self.doc_id_to_text.get(doc_id, "Not found")
            is_match = " [GT MATCH]" if doc_id == ground_truth_id else ""
            print(f"  {i}. [Fused Score: {score:.5f}] ID: {doc_id}{is_match}")
            print(f"     Текст: {text[:150]}...")
        print("-" * 120)


def main() -> None:
    if not settings.validation_set_file.exists():
        logger.error("Validation file missing.")
        sys.exit(1)

    val_data = read_json(settings.validation_set_file)

    # Извлекаем ваши примеры из валидационного датасета для точной диагностики
    targets = {
        "комьюнити токсики дота 2": None,
        "всегда помогут в игре": None,
        "какая это игра блин": None,
        "потерял время в игре": None,
        "такое мнение о доте": None,
    }

    for item in val_data:
        q = item["query"].strip().lower()
        for target_q in list(targets.keys()):
            if target_q in q:
                targets[target_q] = (item["query"], str(item["relevant_doc_id"]))

    logger.info("Initializing comparator pipeline...")
    with RetrievalComparator(top_k=3, candidate_pool_size=100) as comparator:
        for target_key, data in targets.items():
            if data is None:
                logger.warning(
                    f"Target query fragment '{target_key}' not found in validation."
                )
                continue
            query_str, gt_id = data
            # Запускаем сравнение с весом альфа = 0.75 в пользу семантики
            comparator.compare(query_str, gt_id, alpha=0.75)


if __name__ == "__main__":
    main()
