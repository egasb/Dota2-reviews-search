"""
Pydantic-схемы валидации запросов и ответов API.

ReviewPayload зеркалит структуру исходного data/raw_reviews.jsonl —
именно такой payload хранится в точках Qdrant и возвращается в выдаче.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ReviewPayload(BaseModel):
    id: str
    text: str
    voted_up: bool
    votes_up: int
    votes_funny: int
    comment_count: int
    weighted_vote_score: float
    timestamp_created: int
    timestamp_updated: int
    playtime_hours: float
    playtime_at_review_hours: float
    num_games_owned: int
    num_reviews: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Текст поискового запроса")
    top_k: int = Field(10, ge=1, le=100, description="Сколько результатов вернуть")
    collection: str = Field(
        "flat",
        pattern="^(flat|quantized)$",
        description="Какую коллекцию использовать: 'flat' (Итерация 1) или 'quantized' (Итерация 2)",
    )
    min_playtime_hours: Optional[float] = Field(
        None, ge=0, description="Нижняя граница фильтра по playtime_hours"
    )
    max_playtime_hours: Optional[float] = Field(
        None, ge=0, description="Верхняя граница фильтра по playtime_hours"
    )


class SearchResultItem(BaseModel):
    score: float
    payload: ReviewPayload


class SearchResponse(BaseModel):
    query: str
    collection_used: str
    exact_search: bool
    took_ms: float
    total_results: int
    results: List[SearchResultItem]


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    collections: List[str]
