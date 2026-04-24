"""Shared Pydantic primitives."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MemoBase(BaseModel):
    """Shared config for every schema: strict parsing, no extras."""

    model_config = ConfigDict(extra="forbid", frozen=False, str_strip_whitespace=False)


class SourceRef(MemoBase):
    event_id: str
    message_id: str | None = None
    tool_event_id: str | None = None
    snippet: str | None = None
    created_at: datetime | None = None


class SensitivityField(MemoBase):
    sensitivity: str = Field(default="normal")


class ExtractorMetadata(MemoBase):
    extractor_name: str = "manual"
    extractor_version: str = "0"
    prompt_version: str = "0"
    model_provider: str = "manual"
    model_name: str = "manual"
    source_scope: str = "message"
    schema_version: str = "v1"
    extras: dict[str, Any] = Field(default_factory=dict)
