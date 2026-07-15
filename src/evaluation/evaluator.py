# FILE: src/evaluation/evaluator.py

import argparse
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
            if char in (b"\x03", b"\x11"):  # Ctrl+C or Ctrl+Q
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
        self, top_k: int = 50, candidate_pool_size: int = 200, alpha: float = 0.55
    ) -> None:
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size
        self.alpha = alpha
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

        self.stemmer = None
        try:
            import Stemmer

            self.stemmer = Stemmer.Stemmer("russian")
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
            logger.warning("CUDA OOM. Switching to CPU.")
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

        query_scores: list[dict[str, float]] = []
        for i in range(len(raw_queries)):
            query_scores.append(
                {
                    doc_id: float(score)
                    for doc_id, score in zip(results.documents[i], results.scores[i])
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
            fused_scores[doc_id] = base_score * (1.0 + 0.15 * vote_prior)

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
    def run_closed_loop() -> None:
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

    @staticmethod
    def run_evaluate_run() -> None:
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

                for q_id, doc_ratings in annotations.items():
                    if q_id not in run:
                        continue
                    top_3 = sorted(run[q_id].items(), key=lambda x: x[1], reverse=True)[
                        :3
                    ]

                    if all(doc_id in doc_ratings for doc_id, _ in top_3):
                        total_ann_queries += 1
                        for rank, (doc_id, _) in enumerate(top_3, start=1):
                            is_rel = doc_ratings[doc_id]
                            total_ann_docs += 1
                            if is_rel:
                                total_ann_rel_docs += 1
                                if rank == 1:
                                    total_ann_p1_rel += 1

                if total_ann_queries > 0:
                    true_p3 = (
                        (total_ann_rel_docs / total_ann_docs)
                        if total_ann_docs > 0
                        else 0.0
                    )
                    true_p1 = total_ann_p1_rel / total_ann_queries
                    logger.success(
                        "--- TRUE METRICS (HUMAN-IN-THE-LOOP ANNOTATIONS) ---"
                    )
                    logger.info(f"Evaluated queries  : {total_ann_queries}")
                    logger.info(f"True Precision@1   : {true_p1:.4%}")
                    logger.info(f"True Precision@3   : {true_p3:.4%}")
                else:
                    logger.warning(
                        "No fully annotated queries found in manual_annotations.json yet."
                    )
            except Exception as e:
                logger.warning(f"Failed to calculate manual metrics: {e}")

    @staticmethod
    def run_pruning(threshold: float = 0.78) -> None:
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

    @staticmethod
    def run_interactive_assessment(num_samples: int = 15) -> None:
        """Runs interactive CLI tool to calculate real-world precision with state saving."""
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
                logger.info(f"Loaded {len(annotations)} previously annotated queries.")
            except Exception as e:
                logger.warning(
                    f"Failed to load existing progress: {e}. Starting fresh."
                )

        available_q_ids = [q_id for q_id in query_map if q_id in run_data]
        if not available_q_ids:
            logger.error("No valid queries found.")
            return

        unannotated_q_ids = []
        for q_id in available_q_ids:
            top_3 = sorted(run_data[q_id].items(), key=lambda x: x[1], reverse=True)[:3]
            already_annotated = all(
                doc_id in annotations.get(q_id, {}) for doc_id, _ in top_3
            )
            if not already_annotated:
                unannotated_q_ids.append(q_id)

        if not unannotated_q_ids:
            logger.success("All available queries have already been fully annotated!")
            unannotated_q_ids = available_q_ids

        sampled_q_ids = random.sample(
            unannotated_q_ids, min(num_samples, len(unannotated_q_ids))
        )

        print("\n" + "=" * 100)
        print("ИНТЕРАКТИВНЫЙ АССЕССОРСКИЙ ТЕСТ (Сохраняемый прогресс)")
        print(f"Оцените Top-3 кандидатов для {len(sampled_q_ids)} запросов.")
        print(
            "Нажмите: '1' — если релевантен, '0' — если нет, 'q' — сохранить и выйти."
        )
        print("=" * 100)

        queries_completed = 0
        total_evaluated_docs = 0
        relevant_docs_count = 0
        total_p1_relevant = 0

        try:
            for q_idx, q_id in enumerate(sampled_q_ids, start=1):
                query = query_map[q_id]
                top_3 = sorted(
                    run_data[q_id].items(), key=lambda x: x[1], reverse=True
                )[:3]

                print(f"\n[{q_idx}/{len(sampled_q_ids)}] ЗАПРОС: '{query}'")
                print("-" * 100)

                if q_id not in annotations:
                    annotations[q_id] = {}

                for rank, (doc_id, _) in enumerate(top_3, start=1):
                    text = doc_id_to_text.get(doc_id, "Текст отсутствует")

                    if doc_id in annotations[q_id]:
                        is_relevant = annotations[q_id][doc_id]
                        print(
                            f"  Кандидат #{rank} (ID: {doc_id}) -> Авто-загружено: {is_relevant}"
                        )
                    else:
                        print(f"  Кандидат #{rank} (ID: {doc_id})")
                        print(f"  Текст: {text[:280]}...\n")
                        print(
                            "  Релевантен? (1 = Да, 0 = Нет, q = Выйти): ",
                            end="",
                            flush=True,
                        )

                        while True:
                            user_input = get_single_char()
                            if user_input in ("1", "0"):
                                is_relevant = int(user_input)
                                print(is_relevant)
                                annotations[q_id][doc_id] = is_relevant
                                write_json(annotations_file, annotations)  # type: ignore
                                break
                            elif user_input in ("q", "\x03"):
                                print("q\n")
                                raise KeyboardInterrupt

                    total_evaluated_docs += 1
                    if is_relevant:
                        relevant_docs_count += 1
                        if rank == 1:
                            total_p1_relevant += 1

                queries_completed += 1
                print("-" * 100)

        except KeyboardInterrupt:
            print("\nСессия приостановлена. Прогресс успешно сохранен.")

        total_ann_queries = 0
        total_ann_docs = 0
        total_ann_rel_docs = 0
        total_ann_p1_rel = 0

        for q_id, doc_ratings in annotations.items():
            if q_id not in run_data:
                continue
            top_3 = sorted(run_data[q_id].items(), key=lambda x: x[1], reverse=True)[:3]
            if all(doc_id in doc_ratings for doc_id, _ in top_3):
                total_ann_queries += 1
                for rank, (doc_id, _) in enumerate(top_3, start=1):
                    is_rel = doc_ratings[doc_id]
                    total_ann_docs += 1
                    if is_rel:
                        total_ann_rel_docs += 1
                        if rank == 1:
                            total_ann_p1_rel += 1

        if total_ann_queries > 0:
            true_p3 = (
                (total_ann_rel_docs / total_ann_docs) if total_ann_docs > 0 else 0.0
            )
            true_p1 = total_ann_p1_rel / total_ann_queries
            print("\n" + "=" * 100)
            print("ОБЩИЕ НАКОПЛЕННЫЕ ИСТИННЫЕ МЕТРИКИ (HUMAN-IN-THE-LOOP):")
            print(f"  Всего размечено запросов : {total_ann_queries}")
            print(f"  Истинный Precision@1     : {true_p1:.2%}")
            print(f"  Истинный Precision@3     : {true_p3:.2%}")
            print("=" * 100)
        else:
            print("\nНедостаточно данных для расчета накопленных метрик.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Evaluator Suite for Hybrid Search Optimization."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["closed", "generate", "evaluate", "prune", "annotate"],
        help="Command mode: 'closed' (noiseless test), 'generate' (hybrid run), "
        "'evaluate' (standard test), 'prune' (cleanup dataset), 'annotate' (TUI).",
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

    if args.mode == "closed":
        PipelineEvaluator.run_closed_loop()
    elif args.mode == "generate":
        safe_model_name = settings.model_name.replace("/", "_")
        output_path = (
            settings.vectors_file.parent / f"run_hybrid_{safe_model_name}.json"
        )
        with HybridRetriever(
            top_k=50, candidate_pool_size=200, alpha=0.55
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
