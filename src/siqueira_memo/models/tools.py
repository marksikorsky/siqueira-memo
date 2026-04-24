"""``tool_events`` — raw/redacted tool invocations and outputs. Plan §3.3."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from siqueira_memo.models.base import Base, UUIDPrimaryKey
from siqueira_memo.models.constants import EXIT_STATUS_UNKNOWN, SENSITIVITY_NORMAL
from siqueira_memo.models.types import GUID, JSONB, StringArray


class ToolEvent(UUIDPrimaryKey, Base):
    __tablename__ = "tool_events"

    event_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("memory_events.id", ondelete="RESTRICT"), nullable=False
    )
    profile_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    input_raw: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    input_redacted: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    output_raw: Mapped[str | None] = mapped_column(Text(), nullable=True)
    output_redacted: Mapped[str | None] = mapped_column(Text(), nullable=True)
    output_summary: Mapped[str | None] = mapped_column(Text(), nullable=True)
    output_pointer: Mapped[str | None] = mapped_column(String(512), nullable=True)
    output_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    output_size_bytes: Mapped[int | None] = mapped_column(nullable=True)
    exit_status: Mapped[str] = mapped_column(String(24), nullable=False, default=EXIT_STATUS_UNKNOWN)
    artifact_refs: Mapped[list[str]] = mapped_column(
        StringArray(), nullable=False, default=list
    )
    sensitivity: Mapped[str] = mapped_column(
        String(16), nullable=False, default=SENSITIVITY_NORMAL
    )
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
