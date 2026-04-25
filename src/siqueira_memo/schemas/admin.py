"""Admin/search schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from siqueira_memo.schemas.common import MemoBase


class AdminSearchRequest(MemoBase):
    profile_id: str | None = None
    query: str | None = None
    target_type: Literal["message", "fact", "decision", "summary", "chunk"] = "message"
    project_scope: Literal["all", "global", "project"] = "all"
    project: str | None = None
    topic: str | None = None
    status: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 50
    offset: int = 0


class AdminSearchHit(MemoBase):
    id: uuid.UUID
    target_type: str
    preview: str | None = None
    project: str | None = None
    topic: str | None = None
    status: str | None = None
    created_at: datetime


class AdminSearchResponse(MemoBase):
    hits: list[AdminSearchHit]
    total: int = 0


class HealthStatus(MemoBase):
    ok: bool
    env: str
    database: dict[str, Any] = Field(default_factory=dict)
    queue: dict[str, Any] = Field(default_factory=dict)
    pgvector: dict[str, Any] = Field(default_factory=dict)
    migration_version: str | None = None
    providers: list[str] = Field(default_factory=list)
    partitions: dict[str, Any] = Field(default_factory=dict)
    prompt_parity: dict[str, Any] = Field(default_factory=dict)
