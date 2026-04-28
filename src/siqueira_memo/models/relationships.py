"""First-class cross-memory relationship graph. Roadmap Phase 4."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Index, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from siqueira_memo.models.base import Base, UUIDPrimaryKey
from siqueira_memo.models.constants import STATUS_ACTIVE
from siqueira_memo.models.types import GUID, JSONB, UUIDArray


class MemoryRelationship(UUIDPrimaryKey, Base):
    """Directed edge between memories, entities, summaries, and messages.

    Deliberately not constrained with polymorphic foreign keys: ``source_type`` /
    ``target_type`` can point at several tables. Validation lives in
    ``RelationshipService`` so writes remain profile-scoped and auditable.
    """

    __tablename__ = "memory_relationships"

    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    relationship_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    target_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    target_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float(), nullable=False, default=1.0)
    rationale: Mapped[str | None] = mapped_column(Text(), nullable=True)
    source_event_ids: Mapped[list[uuid.UUID]] = mapped_column(
        UUIDArray(), nullable=False, default=list
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False, default="system")
    status: Mapped[str] = mapped_column(String(24), nullable=False, default=STATUS_ACTIVE, index=True)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "source_type",
            "source_id",
            "relationship_type",
            "target_type",
            "target_id",
            "status",
            name="uq_memory_relationships_directed_status",
        ),
        Index(
            "ix_memory_relationships_source",
            "profile_id",
            "source_type",
            "source_id",
            "status",
        ),
        Index(
            "ix_memory_relationships_target",
            "profile_id",
            "target_type",
            "target_id",
            "status",
        ),
    )
