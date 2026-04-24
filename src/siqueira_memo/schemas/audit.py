"""Audit + conflict admin schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import Field

from siqueira_memo.schemas.common import MemoBase


class AuditRequest(MemoBase):
    profile_id: str | None = None
    since: datetime | None = None
    limit: int = 100


class AuditEntrySchema(MemoBase):
    id: uuid.UUID
    event_type: str
    target_type: str | None = None
    target_id: str | None = None
    actor: str
    created_at: datetime
    reason: str | None = None
    mode: str | None = None


class AuditResponse(MemoBase):
    entries: list[AuditEntrySchema] = Field(default_factory=list)
