"""Summary + project state tables. Plan §3.9."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from siqueira_memo.models.base import Base, UUIDPrimaryKey
from siqueira_memo.models.types import JSONB, StringArray, UUIDArray


class SessionSummary(UUIDPrimaryKey, Base):
    __tablename__ = "session_summaries"

    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    summary_short: Mapped[str] = mapped_column(Text(), nullable=False)
    summary_long: Mapped[str] = mapped_column(Text(), nullable=False)
    decisions: Mapped[list[uuid.UUID]] = mapped_column(
        UUIDArray(), nullable=False, default=list
    )
    facts: Mapped[list[uuid.UUID]] = mapped_column(UUIDArray(), nullable=False, default=list)
    open_questions: Mapped[list[str]] = mapped_column(
        StringArray(), nullable=False, default=list
    )
    source_event_ids: Mapped[list[uuid.UUID]] = mapped_column(
        UUIDArray(), nullable=False, default=list
    )
    model: Mapped[str] = mapped_column(String(128), nullable=False, default="mock")
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, default="0")
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False, default="0")
    version: Mapped[int] = mapped_column(Integer(), nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="active")
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class TopicSummary(UUIDPrimaryKey, Base):
    __tablename__ = "topic_summaries"

    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    topic: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    project: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    summary_short: Mapped[str] = mapped_column(Text(), nullable=False)
    summary_long: Mapped[str] = mapped_column(Text(), nullable=False)
    decisions: Mapped[list[uuid.UUID]] = mapped_column(
        UUIDArray(), nullable=False, default=list
    )
    facts: Mapped[list[uuid.UUID]] = mapped_column(UUIDArray(), nullable=False, default=list)
    source_event_ids: Mapped[list[uuid.UUID]] = mapped_column(
        UUIDArray(), nullable=False, default=list
    )
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="active")
    version: Mapped[int] = mapped_column(Integer(), nullable=False, default=1)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ProjectState(UUIDPrimaryKey, Base):
    __tablename__ = "project_states"

    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    project: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    overview: Mapped[str] = mapped_column(Text(), nullable=False)
    key_decisions: Mapped[list[uuid.UUID]] = mapped_column(
        UUIDArray(), nullable=False, default=list
    )
    key_facts: Mapped[list[uuid.UUID]] = mapped_column(UUIDArray(), nullable=False, default=list)
    entities: Mapped[list[uuid.UUID]] = mapped_column(UUIDArray(), nullable=False, default=list)
    source_event_ids: Mapped[list[uuid.UUID]] = mapped_column(
        UUIDArray(), nullable=False, default=list
    )
    version: Mapped[int] = mapped_column(Integer(), nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="active")
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_project_states_profile_project", "profile_id", "project"),
    )
