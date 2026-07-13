# Dota 2 Reviews Search Engine

Поисковая система по отзывам на Dota 2 из Steam (500k+ объектов) на базе Qdrant.
Курс "Deep Learning for Search" (Лето 2026).

Сравниваются две итерации поиска:

| | Итерация 1 (Baseline) | Итерация 2 (Optimized) |
|---|---|---|
| Коллекция | `dota2_flat` | `dota2_quantized` |
| Поиск | Точный (exact / brute-force) | HNSW + Scalar Quantization INT8 |
| Цель | Эталон качества/скорости | Максимальная скорость при минимальной просадке качества |

## Архитектура

Проект изолирован через **Adapter Pattern** (`adapters/`), поэтому бэкенд
полностью рабочий уже сейчас, даже если у Участника 1 (ML-инженера) ещё нет
финальной модели эмбеддингов:

- `adapters/model_adapter.py` — если реальной модели нет, возвращает
  детерминированные mock-векторы (одинаковый текст → одинаковый вектор)
  размерности `VECTOR_SIZE` (по умолчанию 384). Когда модель будет готова —
  меняется один флаг (`USE_MOCK_EMBEDDER=false`), остальной код не трогаем.
- `adapters/file_adapter.py` — потоково читает предрассчитанные эмбеддинги
  из `.npy` (через mmap) или текстового файла, не загружая всё в RAM.

## Пошаговый запуск

### 1. Поднять Qdrant

```bash
docker compose up -d
```

Проверить, что поднялся:

```bash
curl http://localhost:6333/healthz
```

### 2. Установить зависимости Python

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Разместить данные

Положите датасет отзывов в `data/raw_reviews.jsonl` (формат — построчный JSON,
как в примере из ТЗ).

Опционально — если у Участника 1 уже есть готовые эмбеддинги, положите их в
`data/embeddings.npy` (форма `(N, VECTOR_SIZE)`, порядок строк **должен
совпадать** с порядком строк в `raw_reviews.jsonl`). Если файла нет, векторы
посчитаются на лету через `ModelAdapter` (mock по умолчанию).

### 4. Залить данные в обе коллекции Qdrant

```bash
python scripts/ingest_data.py --recreate
```

Флаги:
- `--recreate` — пересоздать коллекции с нуля (удаляет старые данные).
- `--no-precomputed` — игнорировать `data/embeddings.npy`, даже если он есть,
  и всегда считать векторы через `ModelAdapter`.

Скрипт читает JSONL потоково и заливает батчами по 5000 объектов
(`BATCH_SIZE` в `config.py`) в обе коллекции одновременно.

### 5. Запустить API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
        "query": "лучшая игра для игры с друзьями",
        "top_k": 5,
        "collection": "quantized",
        "min_playtime_hours": 100
      }'
```

Swagger UI доступен на `http://localhost:8000/docs`.

### 6. Запустить бенчмарк

```bash
python scripts/run_benchmark.py --requests 100
```

Скрипт делает 100 запросов к каждой из двух коллекций и выводит сравнительный
отчёт: RPS, средняя/P50/P95/P99 латентность и пиковое потребление RAM
процессом Qdrant (через `psutil`; если Qdrant запущен в Docker Desktop на
macOS/Windows, процесс контейнера не виден с хоста — в этом случае скрипт
честно пишет `N/A` и советует использовать `docker stats dota2_qdrant`).

## Переменные окружения

Все параметры (`config.py`) можно переопределить через env, например:

```bash
export VECTOR_SIZE=768
export USE_MOCK_EMBEDDER=false
export EMBEDDER_MODEL_NAME=cointegrated/rubert-tiny2
export QDRANT_HOST=localhost
export BATCH_SIZE=5000
```

## Структура проекта

```
dota2-search-engine/
├── docker-compose.yml
├── requirements.txt
├── config.py
├── adapters/
│   ├── base.py              # абстрактные интерфейсы
│   ├── model_adapter.py     # текст -> вектор (mock/реальная модель)
│   └── file_adapter.py      # чтение .npy/.txt с готовыми эмбеддингами
├── database/
│   ├── client.py            # singleton QdrantClient
│   ├── collections.py       # создание dota2_flat и dota2_quantized
│   └── operations.py        # batch upsert, similarity search + фильтры
├── api/
│   ├── main.py               # FastAPI: /health, /search
│   └── schemas.py            # Pydantic-схемы
└── scripts/
    ├── ingest_data.py        # заливка 500k+ отзывов в Qdrant
    └── run_benchmark.py      # сравнительный бенчмарк flat vs quantized
```
