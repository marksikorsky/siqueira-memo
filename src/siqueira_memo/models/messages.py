"""``messages`` — raw + redacted conversational turns. Plan §3.2."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from siqueira_memo.models.base import Base, UUIDPrimaryKey
from siqueira_memo.models.constants import MESSAGE_SOURCE_LIVE_TURN
from siqueira_memo.models.types import GUID, JSONB, StringArray


class Message(UUIDPrimaryKey, Base):
    __tablename__ = "messages"

    event_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("memory_events.id", ondelete="RESTRICT"), nullable=False
    )
    profile_id: Mapped[str] = mapped_column(String(128), nullable=False)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False)
    platform: Mapped[str] = mapped_column(String(32), nullable=False)
    chat_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    thread_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content_raw: Mapped[str] = mapped_column(Text(), nullable=False)
    content_redacted: Mapped[str] = mapped_column(Text(), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    source: Mapped[str] = mapped_column(
        String(48), nullable=False, default=MESSAGE_SOURCE_LIVE_TURN
    )
    platform_message_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    project: Mapped[str | None] = mapped_column(String(128), nullable=True)
    topic: Mapped[str | None] = mapped_column(String(128), nullable=True)
    entities: Mapped[list[str]] = mapped_column(StringArray(), nullable=False, default=list)
    sensitivity: Mapped[str] = mapped_column(String(16), nullable=False, default="normal")
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_messages_profile_session", "profile_id", "session_id"),
        Index("ix_messages_profile_created", "profile_id", "created_at"),
        Index("ix_messages_project", "project"),
        Index("ix_messages_topic", "topic"),
        Index(
            "ix_messages_ingest_dedupe",
            "profile_id",
            "session_id",
            "role",
            "content_hash",
            unique=True,
            sqlite_where=None,
        ),
    )
