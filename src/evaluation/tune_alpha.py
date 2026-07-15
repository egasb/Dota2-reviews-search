# FILE: src/evaluation/tune_alpha.py

import sys
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sentence_transformers import SentenceTransformer
import ir_measures
from ir_measures import nDCG, MRR

from src.core.config import settings
from src.utils.io import read_json

try:
    import bm25s
except ImportError:
    logger.error("bm25s is not installed.")
    sys.exit(1)


def tune_fusion_parameters() -> None:
    if not settings.validation_set_file.exists():
        logger.error("Validation file missing.")
        sys.exit(1)

    val_data = read_json(settings.validation_set_file)
    val_doc_ids = [str(item["relevant_doc_id"]) for item in val_data]
    query_ids = [str(item["query_id"]) for item in val_data]
    raw_queries = [item["query"] for item in val_data]

    payload = read_json(settings.payload_file)
    raw_embeddings = np.load(settings.vectors_file)

    # 1. Фильтруем базу под закрытый тест (только валидационные документы)
    val_doc_set = set(val_doc_ids)
    filtered_indices = []
    filtered_doc_ids = []
    filtered_corpus = []

    for idx, item in enumerate(payload):
        doc_id = str(item["id"])
        if doc_id in val_doc_set:
            filtered_indices.append(idx)
            filtered_doc_ids.append(doc_id)
            filtered_corpus.append(item["text"])

    logger.info(f"Closed base size: {len(filtered_doc_ids)} docs.")

    # 2. Вычисляем Dense-матрицы сходства
    device = "cuda" if torch.cuda.is_available() else "cpu"
    doc_tensors = torch.from_numpy(raw_embeddings[filtered_indices]).to(
        device, dtype=torch.float32
    )
    doc_tensors = F.normalize(doc_tensors, p=2, dim=1)

    model = SentenceTransformer(settings.model_name, device=device)
    prefix = "query: " if "e5" in settings.model_name.lower() else ""
    formatted_queries = [f"{prefix}{q}" for q in raw_queries]

    query_embeddings = model.encode(
        formatted_queries, convert_to_tensor=True, normalize_embeddings=True
    )
    dense_scores_matrix = torch.matmul(query_embeddings, doc_tensors.T).cpu().numpy()

    # 3. Индексируем BM25S на закрытой базе
    stemmer = None
    try:
        import Stemmer

        stemmer = Stemmer.Stemmer("russian")
    except ImportError:
        pass

    corpus_tokens = bm25s.tokenize(filtered_corpus, stopwords="ru", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    query_tokens = bm25s.tokenize(raw_queries, stemmer=stemmer)
    # Запрашиваем все документы закрытой базы для построения полной матрицы
    bm25_results = retriever.retrieve(
        query_tokens,
        corpus=filtered_doc_ids,
        k=len(filtered_doc_ids),
        return_as="tuple",
    )

    # Пересобираем разреженные скоры в удобный вид
    sparse_scores_dict = {}
    for i, q_id in enumerate(query_ids):
        sparse_scores_dict[q_id] = {
            doc_id: float(score)
            for doc_id, score in zip(bm25_results.documents[i], bm25_results.scores[i])
        }

    qrels = {
        str(item["query_id"]): {str(item["relevant_doc_id"]): 1} for item in val_data
    }
    metrics = [nDCG @ 10, MRR]

    logger.info("Starting Grid Search for Alpha parameter...")
    best_mrr = 0.0
    best_alpha = 0.0

    print(f"{'Alpha':<10} | {'nDCG@10':<10} | {'MRR':<10}")
    print("-" * 38)

    # Перебираем веса от 0.0 (чистый BM25) до 1.0 (чистый E5)
    for alpha_val in np.linspace(0.0, 1.0, 21):
        run = {}
        for i, q_id in enumerate(query_ids):
            # Извлекаем dense скоры
            dense_map = {
                filtered_doc_ids[idx]: float(score)
                for idx, score in enumerate(dense_scores_matrix[i])
            }
            sparse_map = sparse_scores_dict[q_id]

            # Нормализация
            norm_dense = {}
            if dense_map:
                vals = list(dense_map.values())
                min_v, max_v = min(vals), max(vals)
                denom = max_v - min_v
                norm_dense = {
                    d: (s - min_v) / denom if denom > 1e-9 else 1.0
                    for d, s in dense_map.items()
                }

            norm_sparse = {}
            if sparse_map:
                vals = list(sparse_map.values())
                min_v, max_v = min(vals), max(vals)
                denom = max_v - min_v
                norm_sparse = {
                    d: (s - min_v) / denom if denom > 1e-9 else 1.0
                    for d, s in sparse_map.items()
                }

            # Слияние
            fused = {}
            for doc_id in filtered_doc_ids:
                d_s = norm_dense.get(doc_id, 0.0)
                s_s = norm_sparse.get(doc_id, 0.0)
                fused[doc_id] = alpha_val * d_s + (1.0 - alpha_val) * s_s

            run[q_id] = dict(
                sorted(fused.items(), key=lambda x: x[1], reverse=True)[:10]
            )

        results = ir_measures.calc_aggregate(metrics, qrels, run)
        print(
            f"{alpha_val:<10.2f} | {results[nDCG @ 10]:<10.4f} | {results[MRR]:<10.4f}"
        )

        if results[MRR] > best_mrr:
            best_mrr = results[MRR]
            best_alpha = alpha_val

    logger.success(
        f"Tuning finished. Best Alpha: {best_alpha:.2f} (MRR: {best_mrr:.4f})"
    )


if __name__ == "__main__":
    tune_fusion_parameters()
