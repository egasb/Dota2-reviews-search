import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any, Self

import ir_measures
import numpy as np
import torch
import torch.nn.functional as F
from ir_measures import MRR, nDCG, P, R
from loguru import logger
from sentence_transformers import SentenceTransformer

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.config import settings
from src.utils.io import read_json, write_json

try:
    import bm25s
except ImportError:
    bm25s = None

type RunDict = dict[str, dict[str, float]]
type AnnotationSchema = dict[str, dict[str, int]]


def get_single_char() -> str:
    """Reads a single keypress from standard input without waiting for Enter."""
    if sys.platform == "win32":
        import msvcrt

        try:
            char = msvcrt.getch()
            if char in (b"\x03", b"\x11"):
                raise KeyboardInterrupt
            return char.decode("utf-8").lower()
        except (UnicodeDecodeError, AttributeError):
            return ""
    else:
        import termios
        import tty

        fd = sys.stdin.fileno()
        if not sys.stdin.isatty():
            return sys.stdin.read(1).lower()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            char = sys.stdin.read(1)
            if char in ("\x03", "\x11"):
                raise KeyboardInterrupt
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return char.lower()


class HybridRetriever:
    """Computes dense (E5) and sparse (BM25S) retrieval paths with Weighted Score Fusion."""

    def __init__(
        self,
        top_k: int = 50,
        candidate_pool_size: int = 200,
        alpha: float = 0.55,
        vote_weight: float = 0.15,
    ) -> None:
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size
        self.alpha = alpha
        self.vote_weight = vote_weight
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not settings.vectors_file.exists():
            logger.error(f"Vectors file not found: {settings.vectors_file}")
            sys.exit(1)

        self.raw_embeddings = np.load(settings.vectors_file)
        payload_data = read_json(settings.payload_file)
        self.doc_ids = [str(item["id"]) for item in payload_data]
        self.corpus = [item["text"] for item in payload_data]

        # Load static quality priors (weighted_vote_score)
        self.doc_vote_scores = {
            str(item["id"]): float(item.get("score", 0.0)) for item in payload_data
        }

        self.model = SentenceTransformer(settings.model_name, device=self.device)
        self.doc_tensors = torch.from_numpy(self.raw_embeddings).to(
            self.device, dtype=torch.float32
        )
        self.doc_tensors = F.normalize(self.doc_tensors, p=2, dim=1)

        self.stemmer: Any = None
        try:
            import Stemmer

            self.stemmer = Stemmer.Stemmer("russian")
            logger.debug("PyStemmer successfully loaded.")
        except ImportError:
            logger.warning("PyStemmer not installed. Stemming disabled.")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        del self.model
        del self.doc_tensors
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _compute_dense_topk(
        self, query_embeddings: torch.Tensor
    ) -> list[dict[str, float]]:
        k = min(self.candidate_pool_size, len(self.doc_ids))
        try:
            scores = torch.matmul(query_embeddings, self.doc_tensors.T)
            top_scores, top_indices = torch.topk(scores, k=k, dim=1)
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM detected. Falling back to CPU.")
            torch.cuda.empty_cache()
            scores = torch.matmul(query_embeddings.cpu(), self.doc_tensors.cpu().T)
            top_scores, top_indices = torch.topk(scores, k=k, dim=1)

        scores_np = top_scores.cpu().numpy()
        indices_np = top_indices.cpu().numpy()

        return [
            {
                self.doc_ids[idx]: float(score)
                for idx, score in zip(query_indices, query_scores)
            }
            for query_indices, query_scores in zip(indices_np, scores_np)
        ]

    def _compute_sparse_topk(self, raw_queries: list[str]) -> list[dict[str, float]]:
        if bm25s is None:
            logger.error("bm25s library is missing. Install it using 'uv add bm25s'")
            sys.exit(1)

        corpus_tokens = bm25s.tokenize(
            self.corpus, stopwords="ru", stemmer=self.stemmer
        )
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        query_tokens = bm25s.tokenize(raw_queries, stemmer=self.stemmer)
        k = min(self.candidate_pool_size, len(self.corpus))
        results = retriever.retrieve(
            query_tokens, corpus=self.doc_ids, k=k, return_as="tuple"
        )

        # Unpack tuple to avoid "Cannot access attribute" type checking errors
        retrieved_docs, retrieved_scores = results

        query_scores: list[dict[str, float]] = []
        for i in range(len(raw_queries)):
            query_scores.append(
                {
                    doc_id: float(score)
                    for doc_id, score in zip(retrieved_docs[i], retrieved_scores[i])
                }
            )
        return query_scores

    def _normalize_scores(self, scored_docs: dict[str, float]) -> dict[str, float]:
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

    def _apply_weighted_fusion(
        self, dense_scores: dict[str, float], sparse_scores: dict[str, float]
    ) -> dict[str, float]:
        norm_dense = self._normalize_scores(dense_scores)
        norm_sparse = self._normalize_scores(sparse_scores)

        fused_scores: dict[str, float] = {}
        all_doc_ids = set(norm_dense.keys()) | set(norm_sparse.keys())

        for doc_id in all_doc_ids:
            d_score = norm_dense.get(doc_id, 0.0)
            s_score = norm_sparse.get(doc_id, 0.0)
            base_score = self.alpha * d_score + (1.0 - self.alpha) * s_score

            # Static quality helpfulness boost factor
            vote_prior = self.doc_vote_scores.get(doc_id, 0.0)
            fused_scores[doc_id] = base_score * (1.0 + self.vote_weight * vote_prior)

        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[
            : self.top_k
        ]
        return dict(sorted_docs)

    def generate(self, run_output_path: Path) -> None:
        """Runs the hybrid generation pipeline and writes results to disk."""
        if not settings.validation_set_file.exists():
            logger.error("Validation set not found. Generate it first.")
            sys.exit(1)

        val_data = read_json(settings.validation_set_file)
        query_ids = [item["query_id"] for item in val_data]
        raw_queries = [item["query"] for item in val_data]

        prefix = "query: " if "e5" in settings.model_name.lower() else ""
        formatted_queries = [f"{prefix}{query}" for query in raw_queries]

        logger.info(
            f"Encoding {len(formatted_queries)} queries via {settings.model_name}..."
        )
        query_embeddings = self.model.encode(
            formatted_queries,
            batch_size=settings.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        logger.info("Computing dense candidate paths...")
        dense_candidates = self._compute_dense_topk(query_embeddings)

        logger.info("Computing sparse candidate paths...")
        sparse_candidates = self._compute_sparse_topk(raw_queries)

        logger.info("Applying Weighted Linear Fusion...")
        run_dict: RunDict = {}
        for i, q_id in enumerate(query_ids):
            run_dict[q_id] = self._apply_weighted_fusion(
                dense_candidates[i], sparse_candidates[i]
            )

        write_json(run_output_path, run_dict)
        logger.success(f"Hybrid run file saved: {run_output_path}")


class PipelineEvaluator:
    """Implements pipeline tasks: closed-loop testing, standard evaluation, pruning, and interactive annotation."""

    @staticmethod
    def _calculate_ndcg3(ratings: list[int]) -> float:
        """Calculates normalized Discounted Cumulative Gain at rank 3 using exponential gain formula."""
        # Standard formulation: (2^rel - 1) / log2(i + 1)
        dcg = (
            (2.0 ** ratings[0] - 1.0)
            + (2.0 ** ratings[1] - 1.0) / math.log2(3)
            + (2.0 ** ratings[2] - 1.0) / math.log2(4)
        )
        ideal_ratings = sorted(ratings, reverse=True)
        idcg = (
            (2.0 ** ideal_ratings[0] - 1.0)
            + (2.0 ** ideal_ratings[1] - 1.0) / math.log2(3)
            + (2.0 ** ideal_ratings[2] - 1.0) / math.log2(4)
        )
        if idcg < 1e-9:
            return 0.0
        return dcg / idcg

    @staticmethod
    def _calculate_mrr3(ratings: list[int]) -> float:
        """Calculates Mean Reciprocal Rank at rank 3 considering grades >= 2 as relevant."""
        for i, r in enumerate(ratings):
            if r >= 2:  # Relevant threshold (2 or 3)
                return 1.0 / (i + 1)
        return 0.0

    @classmethod
    def run_tune(cls) -> None:
        """Executes a Grid Search on the closed-loop dataset to find optimal hyperparameters."""
        logger.info("Initializing Auto-Tuner on closed dataset...")
        val_data = read_json(settings.validation_set_file)
        val_doc_ids = {str(item["relevant_doc_id"]) for item in val_data}

        payload = read_json(settings.payload_file)
        filtered_docs = [item for item in payload if str(item["id"]) in val_doc_ids]
        doc_ids = [str(item["id"]) for item in filtered_docs]
        corpus = [item["text"] for item in filtered_docs]
        priors = {
            str(item["id"]): float(item.get("score", 0.0)) for item in filtered_docs
        }

        if bm25s is None:
            logger.error("No bm25s module present")
            return

        # Sparse
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(corpus, stopwords="ru"))
        raw_queries = [item["query"] for item in val_data]
        sparse_res = retriever.retrieve(
            bm25s.tokenize(raw_queries),
            corpus=doc_ids,
            k=len(doc_ids),
            return_as="tuple",
        )

        # Unpack tuple to avoid "Cannot access attribute" type checking errors
        sparse_docs, sparse_scores = sparse_res

        # Dense
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(settings.model_name, device=device)
        prefix = "query: " if "e5" in settings.model_name.lower() else ""
        q_embs = model.encode(
            [f"{prefix}{q}" for q in raw_queries],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        d_embs = model.encode(
            [f"passage: {t}" for t in corpus],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        dense_matrix = torch.matmul(q_embs, d_embs.T).cpu().numpy()

        qrels = {
            str(item["query_id"]): {str(item["relevant_doc_id"]): 1}
            for item in val_data
        }

        normalized_queries = []
        for i, item in enumerate(val_data):
            d_map = {doc_ids[j]: float(s) for j, s in enumerate(dense_matrix[i])}
            s_map = {d: float(s) for d, s in zip(sparse_docs[i], sparse_scores[i])}

            d_min, d_max = min(d_map.values()), max(d_map.values())
            s_min, s_max = min(s_map.values()), max(s_map.values())

            norm_d = {
                d: (d_map[d] - d_min) / (d_max - d_min) if d_max > d_min else 1.0
                for d in doc_ids
            }
            norm_s = {
                d: (s_map.get(d, 0.0) - s_min) / (s_max - s_min)
                if s_max > s_min
                else 1.0
                for d in doc_ids
            }

            normalized_queries.append((str(item["query_id"]), norm_d, norm_s))

        best_ndcg, best_params = 0.0, (0.0, 0.0)
        logger.info("Grid scanning Alpha (0.0 - 1.0) and Vote Weight (0.0 - 0.3)...")

        for alpha in np.linspace(0.0, 1.0, 11):
            for vw in np.linspace(0.0, 0.3, 4):
                run = {}
                for q_id, norm_d, norm_s in normalized_queries:
                    fused = {
                        d: (alpha * norm_d[d] + (1.0 - alpha) * norm_s[d])
                        * (1.0 + vw * priors.get(d, 0.0))
                        for d in doc_ids
                    }
                    run[q_id] = dict(
                        sorted(fused.items(), key=lambda x: x[1], reverse=True)[:10]
                    )

                ndcg_score = ir_measures.calc_aggregate([nDCG @ 10], qrels, run)[
                    nDCG @ 10
                ]
                if ndcg_score > best_ndcg:
                    best_ndcg = ndcg_score
                    best_params = (alpha, vw)

        logger.success(
            f"Tuning Complete! Optimal Params -> Alpha: {best_params[0]:.2f}, "
            f"Vote Weight: {best_params[1]:.2f} (Closed nDCG@10: {best_ndcg:.4f})"
        )

    @classmethod
    def run_closed_loop(cls) -> None:
        """Evaluates metrics on the noiseless (closed base) subset of documents."""
        if not settings.validation_set_file.exists():
            logger.error("Validation set file missing.")
            sys.exit(1)

        val_data = read_json(settings.validation_set_file)
        val_doc_ids = {str(item["relevant_doc_id"]) for item in val_data}

        raw_embeddings = np.load(settings.vectors_file)
        payload = read_json(settings.payload_file)

        filtered_indices = []
        filtered_doc_ids = []
        for idx, item in enumerate(payload):
            doc_id = str(item["id"])
            if doc_id in val_doc_ids:
                filtered_indices.append(idx)
                filtered_doc_ids.append(doc_id)

        filtered_embeddings = raw_embeddings[filtered_indices]
        logger.info(f"Closed base test size: {len(filtered_doc_ids)} docs (noiseless).")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        doc_tensors = torch.from_numpy(filtered_embeddings).to(
            device, dtype=torch.float32
        )
        doc_tensors = F.normalize(doc_tensors, p=2, dim=1)

        model = SentenceTransformer(settings.model_name, device=device)
        prefix = "query: " if "e5" in settings.model_name.lower() else ""
        formatted_queries = [f"{prefix}{item['query']}" for item in val_data]
        query_ids = [item["query_id"] for item in val_data]

        query_embeddings = model.encode(
            formatted_queries, convert_to_tensor=True, normalize_embeddings=True
        )
        scores = torch.matmul(query_embeddings, doc_tensors.T)
        top_scores, top_indices = torch.topk(
            scores, k=min(10, len(filtered_doc_ids)), dim=1
        )

        scores_np = top_scores.cpu().numpy()
        indices_np = top_indices.cpu().numpy()

        run = {}
        for i, q_id in enumerate(query_ids):
            run[str(q_id)] = {
                str(filtered_doc_ids[idx]): float(score)
                for idx, score in zip(indices_np[i], scores_np[i])
            }

        qrels = {
            str(item["query_id"]): {str(item["relevant_doc_id"]): 1}
            for item in val_data
        }
        metrics = [nDCG @ 5, nDCG @ 10, P @ 1, R @ 1, MRR]
        results = ir_measures.calc_aggregate(metrics, qrels, run)

        logger.success("--- METRICS ON CLOSED LOOP TEST (NOISELESS) ---")
        for m in metrics:
            logger.info(f"{str(m):<10}: {results[m]:.4f}")

    @classmethod
    def run_evaluate_run(cls) -> None:
        """Evaluates metrics of the generated hybrid run against both synthetic and manual sets."""
        safe_model_name = settings.model_name.replace("/", "_")
        run_file_path = (
            settings.vectors_file.parent / f"run_hybrid_{safe_model_name}.json"
        )

        if not run_file_path.exists():
            logger.error(f"Run file missing: {run_file_path}. Run generate first.")
            sys.exit(1)

        val_data = read_json(settings.validation_set_file)
        qrels = {
            str(item["query_id"]): {str(item["relevant_doc_id"]): 1}
            for item in val_data
        }

        raw_run_dict = read_json(run_file_path)
        run = {
            str(q_id): {str(doc_id): float(score) for doc_id, score in docs.items()}
            for q_id, docs in raw_run_dict.items()
        }

        # 1. Standard evaluation using synthetic data
        metrics = [nDCG @ 5, nDCG @ 10, P @ 1, P @ 5, R @ 5, R @ 20, MRR]
        results = ir_measures.calc_aggregate(metrics, qrels, run)

        logger.success(
            f"--- SYNTHETIC METRICS (GEMMA DATASET WITH NOISE): {settings.model_name} ---"
        )
        for m in metrics:
            logger.info(f"{str(m):<10}: {results[m]:.4f}")

        # 2. Evaluation using manual user annotations
        annotations_file = (
            settings.validation_set_file.parent / "manual_annotations.json"
        )
        if annotations_file.exists():
            try:
                annotations = read_json(annotations_file)
                total_ann_queries = 0
                total_ann_docs = 0
                total_ann_rel_docs = 0
                total_ann_p1_rel = 0
                total_ndcg3 = 0.0
                total_mrr3 = 0.0
                hard_queries = []
                q_map = {item["query_id"]: item["query"] for item in val_data}

                for q_id, doc_ratings in annotations.items():
                    if q_id not in run:
                        continue
                    top_3 = sorted(run[q_id].items(), key=lambda x: x[1], reverse=True)[
                        :3
                    ]

                    if all(doc_id in doc_ratings for doc_id, _ in top_3):
                        total_ann_queries += 1
                        q_ratings = []
                        for rank, (doc_id, _) in enumerate(top_3, start=1):
                            rating = doc_ratings[doc_id]
                            q_ratings.append(rating)
                            total_ann_docs += 1

                            # Graded threshold >= 2 is considered relevant
                            if rating >= 2:
                                total_ann_rel_docs += 1
                                if rank == 1:
                                    total_ann_p1_rel += 1

                        total_ndcg3 += cls._calculate_ndcg3(q_ratings)
                        total_mrr3 += cls._calculate_mrr3(q_ratings)

                        # Save absolute failures (all candidates rated 0 or 1)
                        if sum(1 for r in q_ratings if r >= 2) == 0:
                            hard_queries.append(
                                {"query_id": q_id, "query": q_map.get(q_id, "")}
                            )

                if total_ann_queries > 0:
                    true_p3 = (
                        (total_ann_rel_docs / total_ann_docs)
                        if total_ann_docs > 0
                        else 0.0
                    )
                    true_p1 = total_ann_p1_rel / total_ann_queries
                    true_ndcg3 = total_ndcg3 / total_ann_queries
                    true_mrr3 = total_mrr3 / total_ann_queries

                    logger.success(
                        "--- TRUE METRICS (HUMAN-IN-THE-LOOP ANNOTATIONS) ---"
                    )
                    logger.info(f"Evaluated queries  : {total_ann_queries}")
                    logger.info(f"True Precision@1   : {true_p1:.2%}")
                    logger.info(f"True Precision@3   : {true_p3:.2%}")
                    logger.info(f"True nDCG@3        : {true_ndcg3:.2%}")
                    logger.info(f"True MRR@3         : {true_mrr3:.2%}")

                    if hard_queries:
                        hq_path = (
                            settings.validation_set_file.parent
                            / "hard_queries_report.json"
                        )
                        write_json(hq_path, hard_queries)
                        logger.warning(
                            f"Found {len(hard_queries)} failed queries. Exported to {hq_path.name} for analysis."
                        )
                else:
                    logger.warning(
                        "No fully annotated queries found in manual_annotations.json yet."
                    )
            except Exception as e:
                logger.warning(f"Failed to calculate manual metrics: {e}")

    @classmethod
    def run_pruning(cls, threshold: float = 0.78) -> None:
        """Filters validation dataset by removing low-quality generated queries."""
        if not settings.validation_set_file.exists():
            logger.error("Validation set file missing.")
            sys.exit(1)

        val_data = read_json(settings.validation_set_file)
        payload = read_json(settings.payload_file)
        doc_map = {str(item["id"]): item["text"] for item in payload}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {settings.model_name} on {device} for pruning...")
        model = SentenceTransformer(settings.model_name, device=device)

        queries = [item["query"] for item in val_data]
        prefix = "query: " if "e5" in settings.model_name.lower() else ""
        formatted_queries = [f"{prefix}{q}" for q in queries]
        documents = [
            f"passage: {doc_map.get(str(item['relevant_doc_id']), '')}"
            for item in val_data
        ]

        q_embs = model.encode(
            formatted_queries, convert_to_tensor=True, normalize_embeddings=True
        )
        d_embs = model.encode(
            documents, convert_to_tensor=True, normalize_embeddings=True
        )

        similarities = torch.sum(q_embs * d_embs, dim=1).cpu().numpy()

        pruned_data = []
        dropped_count = 0
        for idx, item in enumerate(val_data):
            score = float(similarities[idx])
            if score >= threshold:
                pruned_data.append(item)
            else:
                dropped_count += 1
                logger.debug(f"Dropped: '{item['query']}' | Similarity: {score:.4f}")

        write_json(settings.validation_set_file, pruned_data)
        logger.success(
            f"Pruning finished. Kept: {len(pruned_data)}. Dropped: {dropped_count}."
        )

    @classmethod
    def run_interactive_assessment(cls, num_samples: int = 15) -> None:
        """Runs interactive CLI tool to calculate real-world precision with state saving."""
        from rich.live import Live

        console = Console()
        (
            doc_id_to_text,
            query_map,
            run_data,
            annotations,
            annotations_file,
        ) = cls._load_assessment_data()

        unannotated_q_ids = cls._get_unannotated_queries(
            query_map, run_data, annotations
        )
        if not unannotated_q_ids:
            logger.success("All available queries have already been fully annotated!")
            unannotated_q_ids = [q for q in query_map if q in run_data]

        sampled_q_ids = random.sample(
            unannotated_q_ids, min(num_samples, len(unannotated_q_ids))
        )

        curr_idx = 0
        try:
            # Live с screen=True создает "оконное" приложение прямо в консоли, не засоряя историю
            with Live(console=console, screen=True, auto_refresh=False) as live:
                while curr_idx < len(sampled_q_ids):
                    q_id = sampled_q_ids[curr_idx]
                    query = query_map[q_id]
                    top_3 = sorted(
                        run_data[q_id].items(), key=lambda x: x[1], reverse=True
                    )[:3]

                    if q_id not in annotations:
                        annotations[q_id] = {}

                    # Ищем первого неоцененного кандидата
                    unrated_idx = next(
                        (
                            i
                            for i, (d_id, _) in enumerate(top_3)
                            if d_id not in annotations[q_id]
                        ),
                        -1,
                    )

                    if unrated_idx == -1:
                        curr_idx += 1
                        continue

                    doc_id = top_3[unrated_idx][0]
                    raw_text = doc_id_to_text.get(doc_id, "Текст отсутствует").replace(
                        "\n", " "
                    )

                    # 1. Отрисовка UI
                    layout = cls._build_assessment_card(
                        query=query,
                        doc_id=doc_id,
                        raw_text=raw_text,
                        curr_idx=curr_idx,
                        total_queries=len(sampled_q_ids),
                        rank=unrated_idx + 1,
                    )
                    live.update(layout, refresh=True)

                    # 2. Ожидание экшена
                    action = None
                    while True:
                        user_input = get_single_char()
                        if user_input in ("0", "1", "2", "3"):
                            annotations[q_id][doc_id] = int(user_input)
                            write_json(annotations_file, annotations)  # type: ignore
                            action = "next"
                            break
                        elif user_input == "b":
                            action = "back"
                            break
                        elif user_input == "s":
                            action = "skip"
                            break
                        elif user_input in ("q", "\x03"):
                            raise KeyboardInterrupt

                    # 3. Обработка навигации
                    if action == "back":
                        curr_idx = cls._handle_back_action(
                            curr_idx,
                            unrated_idx,
                            q_id,
                            sampled_q_ids,
                            run_data,
                            annotations,
                        )
                    elif action == "skip":
                        if q_id in annotations:
                            del annotations[q_id]
                        curr_idx += 1

        except KeyboardInterrupt:
            console.print(
                "[bold yellow]Сессия приостановлена. Прогресс сохранен.[/bold yellow]"
            )

        cls._calculate_and_show_metrics(annotations, run_data, console)

    # =========================================================================
    # Вспомогательные методы (Helpers) для обеспечения чистоты архитектуры
    # =========================================================================

    @classmethod
    def _load_assessment_data(cls) -> tuple:
        import sys

        safe_model_name = settings.model_name.replace("/", "_")
        run_file = settings.vectors_file.parent / f"run_hybrid_{safe_model_name}.json"

        if not run_file.exists():
            logger.error(f"Run file missing: {run_file}.")
            sys.exit(1)

        val_data = read_json(settings.validation_set_file)
        run_data = read_json(run_file)
        payload = read_json(settings.payload_file)

        doc_id_to_text = {str(item["id"]): item["text"] for item in payload}
        query_map = {item["query_id"]: item["query"] for item in val_data}

        annotations_file = (
            settings.validation_set_file.parent / "manual_annotations.json"
        )
        annotations: AnnotationSchema = {}
        if annotations_file.exists():
            try:
                annotations = read_json(annotations_file)  # type: ignore
            except Exception:
                pass

        return doc_id_to_text, query_map, run_data, annotations, annotations_file

    @classmethod
    def _get_unannotated_queries(
        cls, query_map: dict, run_data: dict, annotations: dict
    ) -> list:
        available_q_ids = [q_id for q_id in query_map if q_id in run_data]
        return [
            q
            for q in available_q_ids
            if not all(
                d in annotations.get(q, {})
                for d, _ in sorted(
                    run_data[q].items(), key=lambda x: x[1], reverse=True
                )[:3]
            )
        ]

    @staticmethod
    def _highlight_text(text: str, query: str):
        import re
        from rich.text import Text

        terms = [re.escape(w) for w in query.split() if len(w) > 2]
        rich_text = Text(text[:800] + ("..." if len(text) > 800 else ""))
        if terms:
            pattern = re.compile(f"({'|'.join(terms)})", re.IGNORECASE)
            for match in pattern.finditer(rich_text.plain):
                rich_text.stylize("bold black on yellow", match.start(), match.end())
        return rich_text

    @classmethod
    def _build_assessment_card(
        cls,
        query: str,
        doc_id: str,
        raw_text: str,
        curr_idx: int,
        total_queries: int,
        rank: int,
    ):
        from rich.layout import Layout
        from rich.align import Align

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="footer", size=5),
        )

        layout["header"].update(
            Panel(
                f"[bold white]{query}[/bold white]",
                title=f"[bold cyan]ЗАПРОС {curr_idx + 1}/{total_queries}[/bold cyan] • [dim]Кандидат {rank}/3[/dim]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        layout["body"].update(
            Panel(
                cls._highlight_text(raw_text, query),
                title=f"[bold blue]Текст отзыва (ID: {doc_id})[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
        )

        footer_text = (
            "[bold]Оценка:[/bold] [[green]3[/green]=Топ | [yellow]2[/yellow]=Рел | [magenta]1[/magenta]=Около | [red]0[/red]=Мусор]\n"
            "[dim]────────────────────────────────────────────────────────────────────────[/dim]\n"
            "[bold]Экшен:[/bold] [[blue]b[/blue]=Назад | [yellow]s[/yellow]=Скип | [dim]q[/dim]=Выход]      👉 Ваш выбор: _"
        )

        layout["footer"].update(
            Panel(
                Align.center(footer_text),
                border_style="white",
            )
        )
        return layout

    @classmethod
    def _handle_back_action(
        cls,
        curr_idx: int,
        unrated_idx: int,
        q_id: str,
        sampled_q_ids: list,
        run_data: dict,
        annotations: dict,
    ) -> int:
        """Откатывает оценку на один шаг назад, возвращая обновленный индекс."""
        if unrated_idx > 0:
            top_3 = sorted(run_data[q_id].items(), key=lambda x: x[1], reverse=True)[:3]
            prev_doc = top_3[unrated_idx - 1][0]
            del annotations[q_id][prev_doc]
            return curr_idx

        if curr_idx > 0:
            new_idx = curr_idx - 1
            prev_q_id = sampled_q_ids[new_idx]
            prev_top_3 = sorted(
                run_data[prev_q_id].items(), key=lambda x: x[1], reverse=True
            )[:3]
            for r in reversed(range(3)):
                p_doc = prev_top_3[r][0]
                if p_doc in annotations.get(prev_q_id, {}):
                    del annotations[prev_q_id][p_doc]
                    break
            return new_idx

        return curr_idx

    @classmethod
    def _calculate_and_show_metrics(cls, annotations: dict, run_data: dict, console):
        total_ann_queries = 0
        total_ann_docs = 0
        total_ann_rel_docs = 0
        total_ann_p1_rel = 0
        total_ndcg3 = 0.0
        total_mrr3 = 0.0

        for q_id, doc_ratings in annotations.items():
            if q_id not in run_data:
                continue
            top_3 = sorted(run_data[q_id].items(), key=lambda x: x[1], reverse=True)[:3]
            if all(doc_id in doc_ratings for doc_id, _ in top_3):
                total_ann_queries += 1
                q_ratings = []
                for rank, (doc_id, _) in enumerate(top_3, start=1):
                    is_rel = doc_ratings[doc_id]
                    q_ratings.append(is_rel)
                    total_ann_docs += 1
                    if is_rel >= 2:
                        total_ann_rel_docs += 1
                        if rank == 1:
                            total_ann_p1_rel += 1

                total_ndcg3 += cls._calculate_ndcg3(q_ratings)
                total_mrr3 += cls._calculate_mrr3(q_ratings)

        if total_ann_queries > 0:
            table = Table(
                title="ОБЩИЕ НАКОПЛЕННЫЕ ИСТИННЫЕ МЕТРИКИ (HUMAN-IN-THE-LOOP)",
                show_header=True,
                header_style="bold magenta",
                title_style="bold blue",
                expand=True,
            )
            table.add_column("Метрика", justify="left", style="white")
            table.add_column("Значение", justify="right", style="bold yellow")

            table.add_row("Всего размечено запросов", str(total_ann_queries))
            table.add_row(
                "Истинный Precision@1", f"{total_ann_p1_rel / total_ann_queries:.2%}"
            )
            table.add_row(
                "Истинный Precision@3",
                f"{(total_ann_rel_docs / total_ann_docs) if total_ann_docs > 0 else 0.0:.2%}",
            )
            table.add_row("Истинный nDCG@3", f"{total_ndcg3 / total_ann_queries:.2%}")
            table.add_row("Истинный MRR@3", f"{total_mrr3 / total_ann_queries:.2%}")

            console.print(table)
        else:
            console.print(
                "\n[bold red]Недостаточно данных для расчета метрик.[/bold red]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Evaluator Suite for Hybrid Search Optimization."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["tune", "generate", "evaluate", "prune", "annotate"],
        help="Command mode: 'tune' (hyperparameter tuner), 'generate' (hybrid run), "
        "'evaluate' (standard test), 'prune' (cleanup dataset), 'annotate' (TUI).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Weight for dense retrieval (default: 0.55).",
    )
    parser.add_argument(
        "--vote-weight",
        type=float,
        default=0.15,
        help="Boost factor for highly helpful reviews (default: 0.15).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.78,
        help="Cosine similarity threshold for pruning (default: 0.78).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=15,
        help="Number of samples to grade in interactive annotation (default: 15).",
    )
    args = parser.parse_args()

    if args.mode == "tune":
        PipelineEvaluator.run_tune()
    elif args.mode == "generate":
        safe_model_name = settings.model_name.replace("/", "_")
        output_path = (
            settings.vectors_file.parent / f"run_hybrid_{safe_model_name}.json"
        )
        with HybridRetriever(
            alpha=args.alpha, vote_weight=args.vote_weight
        ) as retriever:
            retriever.generate(output_path)
    elif args.mode == "evaluate":
        PipelineEvaluator.run_evaluate_run()
    elif args.mode == "prune":
        PipelineEvaluator.run_pruning(threshold=args.threshold)
    elif args.mode == "annotate":
        PipelineEvaluator.run_interactive_assessment(num_samples=args.samples)


if __name__ == "__main__":
    main()
