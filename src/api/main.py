"""
FastAPI-сервер поисковой системы по отзывам Dota 2.

Роуты:
  GET  /health  — статус сервиса и подключения к Qdrant.
  POST /search  — семантический поиск по отзывам с опциональной
                   фильтрацией по playtime_hours.
"""

import time

from fastapi import FastAPI, HTTPException

from src.adapters.model_adapter import ModelAdapter
from src.api.schemas import (
    HealthResponse,
    ReviewPayload,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from src.core.config import settings
from src.database.client import QdrantClientSingleton
from src.database.collections import ensure_collections
from src.database.operations import search as vector_search

app = FastAPI(
    title="Dota 2 Reviews Search Engine",
    description="Поисковая система по отзывам на Dota 2 из Steam (Qdrant, Flat vs Quantized)",
    version="1.0.0",
)

# Единственный экземпляр адаптера модели на весь процесс API.
embedder = ModelAdapter(
    vector_size=settings.VECTOR_SIZE,
    use_mock=settings.USE_MOCK_EMBEDDER,
    model_name=settings.EMBEDDER_MODEL_NAME,
)

COLLECTION_MAP = {
    "flat": settings.COLLECTION_FLAT,
    "quantized": settings.COLLECTION_QUANTIZED,
}


@app.on_event("startup")
def on_startup() -> None:
    """Гарантируем, что обе коллекции существуют к моменту первого запроса."""
    ensure_collections()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    client = QdrantClientSingleton.get_client()
    try:
        collections = client.get_collections().collections
        names = [c.name for c in collections]
        connected = True
    except Exception:
        names = []
        connected = False

    return HealthResponse(
        status="ok" if connected else "degraded",
        qdrant_connected=connected,
        collections=names,
    )


@app.post("/search", response_model=SearchResponse)
def search_reviews(request: SearchRequest) -> SearchResponse:
    collection_name = COLLECTION_MAP.get(request.collection)
    if collection_name is None:
        raise HTTPException(status_code=400, detail="Неизвестная коллекция")

    query_vector = embedder.encode(request.query)
    # Для baseline-коллекции всегда делаем точный (brute-force) поиск,
    # для quantized — приближённый через HNSW + квантование.
    exact = request.collection == "flat"

    start = time.perf_counter()
    try:
        points = vector_search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=request.top_k,
            min_playtime=request.min_playtime_hours,
            max_playtime=request.max_playtime_hours,
            exact=exact,
        )
    except Exception as exc:  # noqa: BLE001 — оборачиваем любую ошибку Qdrant в 500
        raise HTTPException(
            status_code=500, detail=f"Ошибка поиска в Qdrant: {exc}"
        ) from exc

    took_ms = (time.perf_counter() - start) * 1000

    results = [
        SearchResultItem(score=point.score, payload=ReviewPayload(**point.payload))
        for point in points
        if point.payload is not None
    ]

    return SearchResponse(
        query=request.query,
        collection_used=collection_name,
        exact_search=exact,
        took_ms=round(took_ms, 2),
        total_results=len(results),
        results=results,
    )
