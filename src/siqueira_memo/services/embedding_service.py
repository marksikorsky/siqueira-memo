"""Embedding provider abstraction. Plan §4.2 / §31.6.

Three providers are supported:

* ``mock`` — deterministic 16-dim hash-derived embedding, used by unit tests.
* ``openai`` — ``text-embedding-3-large`` (3072 dim), network call.
* ``local`` — BAAI/bge-m3 placeholder; production deployments must install the
  matching sentence-transformers stack.

The service is selected from settings at runtime. Tests never make network
calls because the default test settings pin ``mock``.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
from dataclasses import dataclass
from typing import Any, Protocol

from siqueira_memo.config import Settings, get_settings


@dataclass(frozen=True)
class EmbeddingSpec:
    provider: str
    model_name: str
    model_version: str
    dimensions: int
    table_name: str


class EmbeddingProvider(Protocol):
    spec: EmbeddingSpec

    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class MockEmbeddingProvider:
    """Deterministic 16-dimensional embedding suitable for unit tests.

    The vector is derived from a SHA-256 hash so identical inputs embed
    identically and different inputs end up in different directions. The final
    vector is L2-normalised so cosine similarity is just a dot product.
    """

    spec = EmbeddingSpec(
        provider="mock",
        model_name="mock",
        model_version="1",
        dimensions=16,
        table_name="chunk_embeddings_mock",
    )

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        vec = [b / 255.0 - 0.5 for b in digest[: self.spec.dimensions]]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class OpenAIEmbeddingProvider:
    """OpenAI text-embedding-3-large (3072 dims). Plan §18.1."""

    def __init__(self, api_key: str, *, model: str = "text-embedding-3-large") -> None:
        self._api_key = api_key
        self.spec = EmbeddingSpec(
            provider="openai",
            model_name=model,
            model_version="2024-01-25",
            dimensions=3072,
            table_name="chunk_embeddings_openai_text_embedding_3_large",
        )

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import httpx  # noqa: WPS433

        # Batching up to the documented limit; callers should slice if needed.
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body: dict[str, Any] = {"model": self.spec.model_name, "input": texts}
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/embeddings", headers=headers, json=body
            )
            resp.raise_for_status()
            data = resp.json()
        embeddings: list[list[float]] = [item["embedding"] for item in data["data"]]
        return embeddings


class LocalBGEM3Provider:
    """Placeholder for local bge-m3 embedder.

    Raises at construction if sentence-transformers is unavailable so tests
    never accidentally instantiate this provider.
    """

    def __init__(self) -> None:
        # ``sentence_transformers`` is an optional dependency — it has no type
        # stubs and is never installed in the default/test environment. We use
        # importlib to keep mypy from choking on the missing module.
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers required for the local bge-m3 provider"
            ) from exc
        self._model: Any = module.SentenceTransformer("BAAI/bge-m3")
        self.spec = EmbeddingSpec(
            provider="local",
            model_name="bge-m3",
            model_version="1",
            dimensions=1024,
            table_name="chunk_embeddings_bge_m3",
        )

    def embed(self, text: str) -> list[float]:  # pragma: no cover
        vector: list[float] = self._model.encode(text, normalize_embeddings=True).tolist()
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        return [v.tolist() for v in self._model.encode(texts, normalize_embeddings=True)]


def build_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    name = settings.embedding_provider
    if name == "mock":
        return MockEmbeddingProvider()
    if name == "openai":
        api_key = settings.openai_api_key.get_secret_value() or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI embedding provider requires an API key")
        return OpenAIEmbeddingProvider(api_key, model=settings.embedding_model)
    if name == "local":
        return LocalBGEM3Provider()
    raise ValueError(f"unsupported embedding provider: {name}")


def cosine(a: list[float], b: list[float]) -> float:
    """L2-cosine similarity helper used by the SQLite retrieval path."""
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(length):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)
