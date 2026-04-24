"""``decisions`` + ``decision_sources``. Plan §3.8."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column

from siqueira_memo.models.base import Base, UUIDPrimaryKey
from siqueira_memo.models.constants import STATUS_CANDIDATE
from siqueira_memo.models.types import GUID, JSONB, UUIDArray


class Decision(UUIDPrimaryKey, Base):
    __tablename__ = "decisions"

    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    project: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    topic: Mapped[str] = mapped_column(String(256), nullable=False)
    decision: Mapped[str] = mapped_column(Text(), nullable=False)
    context: Mapped[str] = mapped_column(Text(), nullable=False)
    options_considered: Mapped[list[dict[str, Any]]] = mapped_column(JSONB(), nullable=False, default=list)
    rationale: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    tradeoffs: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    canonical_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default=STATUS_CANDIDATE, index=True
    )
    reversible: Mapped[bool] = mapped_column(nullable=False, default=True)
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("decisions.id"), nullable=True
    )
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_event_ids: Mapped[list[uuid.UUID]] = mapped_column(
        UUIDArray(), nullable=False, default=list
    )
    source_message_ids: Mapped[list[uuid.UUID]] = mapped_column(
        UUIDArray(), nullable=False, default=list
    )
    extractor_name: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    extractor_version: Mapped[str] = mapped_column(String(64), nullable=False, default="0")
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False, default="0")
    model_provider: Mapped[str] = mapped_column(String(64), nullable=False, default="mock")
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, default="mock")
    source_scope: Mapped[str] = mapped_column(String(16), nullable=False, default="window")
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
            "uq_decisions_active_canonical",
            "profile_id",
            "canonical_key",
            unique=True,
            sqlite_where=text("status = 'active'"),
            postgresql_where=text("status = 'active'"),
        ),
        Index("ix_decisions_extractor_version", "extractor_version"),
        Index("ix_decisions_prompt_version", "prompt_version"),
    )


class DecisionSource(Base):
    __tablename__ = "decision_sources"

    decision_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("decisions.id", ondelete="CASCADE"), primary_key=True
    )
    event_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("memory_events.id", ondelete="RESTRICT"), primary_key=True
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
