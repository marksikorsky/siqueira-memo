"""``facts`` + ``fact_sources``. Plan §3.7 / §28.2 / §31.8."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from siqueira_memo.models.base import Base, UUIDPrimaryKey
from siqueira_memo.models.constants import STATUS_CANDIDATE
from siqueira_memo.models.types import GUID, JSONB, UUIDArray


class Fact(UUIDPrimaryKey, Base):
    __tablename__ = "facts"

    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    subject: Mapped[str] = mapped_column(String(512), nullable=False)
    predicate: Mapped[str] = mapped_column(String(256), nullable=False)
    object: Mapped[str] = mapped_column(Text(), nullable=False)
    statement: Mapped[str] = mapped_column(Text(), nullable=False)
    canonical_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    project: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    topic: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    confidence: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default=STATUS_CANDIDATE, index=True
    )
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source_event_ids: Mapped[list[uuid.UUID]] = mapped_column(
        UUIDArray(), nullable=False, default=list
    )
    source_message_ids: Mapped[list[uuid.UUID]] = mapped_column(
        UUIDArray(), nullable=False, default=list
    )
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("facts.id"), nullable=True
    )
    extractor_name: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    extractor_version: Mapped[str] = mapped_column(String(64), nullable=False, default="0")
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False, default="0")
    model_provider: Mapped[str] = mapped_column(String(64), nullable=False, default="mock")
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, default="mock")
    source_scope: Mapped[str] = mapped_column(String(16), nullable=False, default="message")
    schema_version: Mapped[str] = mapped_column(String(16), nullable=False, default="v1")
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index(
            "uq_facts_active_canonical",
            "profile_id",
            "canonical_key",
            unique=True,
            sqlite_where=text("status = 'active'"),
            postgresql_where=text("status = 'active'"),
        ),
        Index("ix_facts_extractor_version", "extractor_version"),
        Index("ix_facts_prompt_version", "prompt_version"),
    )


class FactSource(Base):
    __tablename__ = "fact_sources"

    fact_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("facts.id", ondelete="CASCADE"), primary_key=True
    )
    event_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("memory_events.id", ondelete="RESTRICT"), primary_key=True
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
