"""``chunks`` + per-model embedding tables. Plan §3.5 / §31.6."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from siqueira_memo.models.base import Base, UUIDPrimaryKey
from siqueira_memo.models.constants import SENSITIVITY_NORMAL
from siqueira_memo.models.types import GUID, JSONB, StringArray, Vector


class Chunk(UUIDPrimaryKey, Base):
    """A unit of text + metadata ready for retrieval.

    Embeddings do **not** live on this row in production (§31.6): the canonical
    vector storage is a per-model/dimension physical table. For testability and
    non-Postgres backends we keep a nullable inline vector column — SQL views
    are not portable — but production queries must hit ``chunk_embeddings_*``.
    """

    __tablename__ = "chunks"

    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    chunk_text: Mapped[str] = mapped_column(Text(), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    token_count: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    tokenizer_name: Mapped[str] = mapped_column(
        String(64), nullable=False, default="unknown"
    )
    tokenizer_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    project: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    topic: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    entities: Mapped[list[str]] = mapped_column(StringArray(), nullable=False, default=list)
    sensitivity: Mapped[str] = mapped_column(
        String(16), nullable=False, default=SENSITIVITY_NORMAL
    )
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    __table_args__ = (
        UniqueConstraint(
            "profile_id", "source_type", "source_id", "chunk_index", name="uq_chunks_source_index"
        ),
    )


class EmbeddingIndex(UUIDPrimaryKey, Base):
    """Registry of physical embedding tables. Plan §31.6."""

    __tablename__ = "embedding_indexes"

    table_name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer(), nullable=False)
    distance_metric: Mapped[str] = mapped_column(
        String(16), nullable=False, default="cosine"
    )
    active: Mapped[bool] = mapped_column(nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class _EmbeddingRowMixin(UUIDPrimaryKey):
    """Shared columns for per-model chunk embedding tables."""

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer(), nullable=False)


class ChunkEmbeddingOpenAITEL3(_EmbeddingRowMixin, Base):
    """Physical embedding table for OpenAI text-embedding-3-large (3072)."""

    __tablename__ = "chunk_embeddings_openai_text_embedding_3_large"

    embedding: Mapped[list[float] | None] = mapped_column(Vector(3072), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("chunk_id", "model_version", name="uq_cemb_openai_te3l_chunk_version"),
        Index("ix_cemb_openai_te3l_model", "model_name", "model_version"),
    )


class ChunkEmbeddingBGEM3(_EmbeddingRowMixin, Base):
    """Physical embedding table for BAAI/bge-m3 (1024)."""

    __tablename__ = "chunk_embeddings_bge_m3"

    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("chunk_id", "model_version", name="uq_cemb_bge_m3_chunk_version"),
    )


class ChunkEmbeddingMock(_EmbeddingRowMixin, Base):
    """Tiny deterministic table used in tests (dim=16)."""

    __tablename__ = "chunk_embeddings_mock"

    embedding: Mapped[list[float] | None] = mapped_column(Vector(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("chunk_id", "model_version", name="uq_cemb_mock_chunk_version"),
    )


EMBEDDING_TABLE_BY_MODEL = {
    ("mock", 16): ChunkEmbeddingMock,
    ("text-embedding-3-large", 3072): ChunkEmbeddingOpenAITEL3,
    ("bge-m3", 1024): ChunkEmbeddingBGEM3,
}
