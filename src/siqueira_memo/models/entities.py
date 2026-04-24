"""``entities`` + aliases + relationships. Plan §3.6 / §19."""

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
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from siqueira_memo.models.base import Base, UUIDPrimaryKey
from siqueira_memo.models.constants import STATUS_ACTIVE, STATUS_CANDIDATE
from siqueira_memo.models.types import GUID, JSONB, StringArray, UUIDArray


class Entity(UUIDPrimaryKey, Base):
    __tablename__ = "entities"

    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    name_normalized: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    aliases: Mapped[list[str]] = mapped_column(StringArray(), nullable=False, default=list)
    description: Mapped[str | None] = mapped_column(Text(), nullable=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default=STATUS_CANDIDATE)
    merged_into: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True)
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
            "name_normalized",
            "type",
            name="uq_entities_profile_name_type",
        ),
    )


class EntityAlias(UUIDPrimaryKey, Base):
    __tablename__ = "entity_aliases"

    entity_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    alias: Mapped[str] = mapped_column(String(256), nullable=False)
    alias_normalized: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default=STATUS_ACTIVE)
    source_event_ids: Mapped[list[uuid.UUID]] = mapped_column(
        UUIDArray(), nullable=False, default=list
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index(
            "uq_entity_aliases_norm_type_profile_active",
            "profile_id",
            "alias_normalized",
            "entity_type",
            unique=True,
        ),
    )


class EntityRelationship(UUIDPrimaryKey, Base):
    __tablename__ = "entity_relationships"

    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source_entity_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    relation: Mapped[str] = mapped_column(String(64), nullable=False)
    target_entity_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float(), nullable=False, default=1.0)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default=STATUS_ACTIVE)
    source_event_ids: Mapped[list[uuid.UUID]] = mapped_column(
        UUIDArray(), nullable=False, default=list
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
