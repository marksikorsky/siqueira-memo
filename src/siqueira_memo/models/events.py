"""Append-only ``memory_events`` table. Plan §3.1."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from siqueira_memo.models.base import Base, UUIDPrimaryKey
from siqueira_memo.models.types import JSONB


class MemoryEvent(UUIDPrimaryKey, Base):
    """Every ingest/extraction/deletion operation creates exactly one row here.

    The row is the canonical source of truth for provenance. Derived tables
    (``messages``, ``tool_events``, ``facts``, ``decisions`` …) reference it.
    """

    __tablename__ = "memory_events"

    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    agent_context: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    __table_args__ = (
        Index("ix_memory_events_profile_session", "profile_id", "session_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<MemoryEvent {self.event_type} {self.id}>"


if TYPE_CHECKING:  # pragma: no cover
    _ensure_event_id: uuid.UUID
