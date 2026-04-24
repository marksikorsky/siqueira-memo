"""Ingest request schemas. Plan §4.2."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator

from siqueira_memo.models.constants import (
    ALL_ROLES,
    EXIT_STATUS_OK,
    MESSAGE_SOURCE_LIVE_TURN,
    SENSITIVITY_NORMAL,
)
from siqueira_memo.schemas.common import MemoBase


class MessageIngestIn(MemoBase):
    profile_id: str | None = None
    session_id: str
    platform: str = "generic"
    chat_id: str | None = None
    thread_id: str | None = None
    role: str
    content: str
    created_at: datetime | None = None
    source: str = MESSAGE_SOURCE_LIVE_TURN
    platform_message_id: str | None = None
    language: str | None = None
    project: str | None = None
    topic: str | None = None
    entities: list[str] = Field(default_factory=list)
    sensitivity: str = SENSITIVITY_NORMAL
    agent_context: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("role")
    @classmethod
    def _validate_role(cls, value: str) -> str:
        if value not in ALL_ROLES:
            raise ValueError(f"unsupported role: {value}")
        return value


class MessageIngestOut(MemoBase):
    event_id: uuid.UUID
    message_id: uuid.UUID
    duplicate: bool = False
    redactions: int = 0


class ToolEventIngestIn(MemoBase):
    profile_id: str | None = None
    session_id: str
    tool_name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: str | None = None
    output_pointer: str | None = None
    output_hash: str | None = None
    output_size_bytes: int | None = None
    exit_status: str = EXIT_STATUS_OK
    artifact_refs: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    agent_context: str | None = None
    sensitivity: str = SENSITIVITY_NORMAL
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolEventIngestOut(MemoBase):
    event_id: uuid.UUID
    tool_event_id: uuid.UUID
    redactions: int = 0


class ArtifactIngestIn(MemoBase):
    profile_id: str | None = None
    type: str
    path: str | None = None
    uri: str | None = None
    content_hash: str | None = None
    summary: str | None = None
    project: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class ArtifactIngestOut(MemoBase):
    event_id: uuid.UUID
    artifact_id: uuid.UUID


class GenericEventIn(MemoBase):
    profile_id: str | None = None
    session_id: str | None = None
    event_type: str
    source: str = "external"
    actor: str = "system"
    trace_id: str | None = None
    agent_context: str | None = None
    payload: dict[str, Any]


class GenericEventOut(MemoBase):
    event_id: uuid.UUID


class DelegationObservationIn(MemoBase):
    profile_id: str | None = None
    parent_session_id: str
    child_session_id: str | None = None
    task: str
    result: str
    toolsets: list[str] = Field(default_factory=list)
    model: str | None = None


class HermesAuxCompactionIn(MemoBase):
    profile_id: str | None = None
    session_id: str
    summary_text: str
    prefix: str = "[CONTEXT COMPACTION]"
    source_message_count: int = 0
    sensitivity: str = SENSITIVITY_NORMAL


class BuiltinMemoryMirrorIn(MemoBase):
    profile_id: str | None = None
    session_id: str
    action: Literal["add", "replace", "remove"]
    target: Literal["memory", "user"]
    content: str | None = None
    selector: str | None = None
