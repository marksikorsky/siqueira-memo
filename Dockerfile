# syntax=docker/dockerfile:1.6

# Siqueira Memo runtime image. Used by docker-compose's ``api`` and ``worker``
# services. Pinning Python 3.12-slim keeps the image ~150 MB without dropping
# stdlib features the codebase relies on.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini
COPY plugins ./plugins
COPY scripts ./scripts

# Install runtime deps + postgres/queue extras so the compose-style deploy
# works without a second layer. Dev/test extras are intentionally excluded.
RUN pip install --upgrade pip \
    && pip install ".[postgres,queue,otel]"

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8787/healthz || exit 1

CMD ["uvicorn", "siqueira_memo.main:app", "--host", "0.0.0.0", "--port", "8787"]
