FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models/bge_m3

WORKDIR /app

# Install the locked runtime dependency set (including TensorFlow and NeMo).
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --no-install-project --no-cache

COPY app ./app
COPY config ./config
COPY models/bge_m3 ./models/bge_m3

EXPOSE 8000
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
