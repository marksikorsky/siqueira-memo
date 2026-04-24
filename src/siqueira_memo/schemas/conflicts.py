"""Conflict-service schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from siqueira_memo.schemas.common import MemoBase


class ConflictItem(MemoBase):
    id: uuid.UUID
    conflict_type: str
    left_type: str
    left_id: uuid.UUID
    right_type: str
    right_id: uuid.UUID
    severity: str
    status: str
    resolution_hint: str | None = None
    confidence: float = 0.0
    created_at: datetime
    resolved_at: datetime | None = None


class ConflictScanResponse(MemoBase):
    detected: int
    conflicts: list[ConflictItem]


class ConflictResolveRequest(MemoBase):
    conflict_id: uuid.UUID
    kept_id: uuid.UUID
    dropped_id: uuid.UUID
    actor: str = "auto"


class ConflictResolveResponse(MemoBase):
    id: uuid.UUID
    status: str
    resolution: str | None = None
    resolved_at: datetime | None = None
