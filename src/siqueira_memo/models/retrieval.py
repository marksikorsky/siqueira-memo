"""``retrieval_logs`` + ``memory_conflicts`` + ``prompt_versions``. Plan §10.3 / §21.6 / §22.2."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from siqueira_memo.models.base import Base, UUIDPrimaryKey
from siqueira_memo.models.constants import CONFLICT_STATUS_OPEN
from siqueira_memo.models.types import GUID, JSONB, UUIDArray


class RetrievalLog(UUIDPrimaryKey, Base):
    __tablename__ = "retrieval_logs"

    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    query: Mapped[str] = mapped_column(Text(), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="balanced")
    types: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    selected_source_ids: Mapped[list[uuid.UUID]] = mapped_column(
        UUIDArray(), nullable=False, default=list
    )
    rejected_count: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    candidates_count: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    conflicts_count: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    embedding_table: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reranker_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )


class MemoryConflict(UUIDPrimaryKey, Base):
    __tablename__ = "memory_conflicts"

    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    conflict_type: Mapped[str] = mapped_column(String(32), nullable=False)
    left_type: Mapped[str] = mapped_column(String(32), nullable=False)
    left_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False)
    right_type: Mapped[str] = mapped_column(String(32), nullable=False)
    right_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="medium")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=CONFLICT_STATUS_OPEN, index=True
    )
    resolution: Mapped[str | None] = mapped_column(Text(), nullable=True)
    resolution_hint: Mapped[str | None] = mapped_column(Text(), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class PromptVersion(UUIDPrimaryKey, Base):
    __tablename__ = "prompt_versions"

    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    body: Mapped[str | None] = mapped_column(Text(), nullable=True)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (UniqueConstraint("name", "version", name="uq_prompt_versions_name_version"),)
