# Используем официальный образ Python
FROM python:3.13-slim

# Копируем утилиту uv из официального образа
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Копируем файлы конфигурации менеджера пакетов
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости в изолированное окружение
RUN uv sync --frozen --no-dev

# Копируем весь остальной код проекта
COPY . .

ENV PYTHONPATH="/app"
ENV PATH="/app/.venv/bin:$PATH"


# Запускаем FastAPI сервер через uv (с учетом новой структуры, см. пункт 2)
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]