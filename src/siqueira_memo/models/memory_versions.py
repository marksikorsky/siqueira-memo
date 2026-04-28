"""Version history for durable memory rows."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from siqueira_memo.models.base import Base, UUIDPrimaryKey
from siqueira_memo.models.types import GUID, JSONB


class MemoryVersion(UUIDPrimaryKey, Base):
    """Immutable audit/version entry for a fact or decision snapshot.

    ``target_id`` is intentionally polymorphic rather than a hard foreign key:
    hard-delete rollback/audit must keep history even after the target row is gone.
    """

    __tablename__ = "memory_versions"

    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    target_type: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    target_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer(), nullable=False)
    operation: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(128), nullable=False, default="system")
    reason: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    event_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("memory_events.id", ondelete="SET NULL"), nullable=True, index=True
    )
    rollback_to_version: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    before_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB(), nullable=True)
    after_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "target_type",
            "target_id",
            "version",
            name="uq_memory_versions_target_version",
        ),
        Index("ix_memory_versions_target_created", "target_type", "target_id", "created_at"),
    )
