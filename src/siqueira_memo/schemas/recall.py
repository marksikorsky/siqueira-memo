"""Recall request/response + context pack schemas. Plan §4.3 / §5.8."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator

from siqueira_memo.models.constants import ALL_RECALL_MODES, RECALL_MODE_BALANCED
from siqueira_memo.schemas.common import MemoBase, SourceRef

RecallResultType = Literal[
    "decisions",
    "facts",
    "chunks",
    "summaries",
    "entities",
    "topics",
    "projects",
]

_DEFAULT_RECALL_TYPES: tuple[RecallResultType, ...] = (
    "decisions",
    "facts",
    "chunks",
    "summaries",
)


class RecallRequest(MemoBase):
    profile_id: str | None = None
    query: str
    project: str | None = None
    topic: str | None = None
    entities: list[str] = Field(default_factory=list)
    mode: str = RECALL_MODE_BALANCED
    types: list[RecallResultType] = Field(
        default_factory=lambda: list(_DEFAULT_RECALL_TYPES)
    )
    include_sources: bool = True
    include_conflicts: bool = True
    allow_secret_recall: bool = False
    limit: int = 20
    since: datetime | None = None
    until: datetime | None = None
    session_id: str | None = None

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        if value not in ALL_RECALL_MODES:
            raise ValueError(f"unsupported mode: {value}")
        return value


class RecallDecision(MemoBase):
    id: uuid.UUID
    project: str | None
    topic: str
    decision: str
    rationale: str
    status: str
    reversible: bool
    decided_at: datetime
    confidence: float = 0.0
    sensitivity: str = "internal"
    masked_preview: str | None = None
    secret_ref: str | None = None
    secret_masked: bool = False
    retrieval_lane: str = "structured"
    retrieval_explanation: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)


class RecallFact(MemoBase):
    id: uuid.UUID
    subject: str
    predicate: str
    object: str
    statement: str
    status: str
    confidence: float
    project: str | None = None
    topic: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    sensitivity: str = "internal"
    masked_preview: str | None = None
    secret_ref: str | None = None
    secret_masked: bool = False
    retrieval_lane: str = "structured"
    retrieval_explanation: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)


class RecallChunk(MemoBase):
    id: uuid.UUID
    source_type: str
    source_id: uuid.UUID
    chunk_text: str
    score: float
    project: str | None = None
    topic: str | None = None
    sensitivity: str = "normal"
    created_at: datetime | None = None


class RecallSummary(MemoBase):
    id: uuid.UUID
    scope: Literal["session", "topic", "project"]
    summary_short: str
    summary_long: str | None = None
    created_at: datetime | None = None


class ConflictEntry(MemoBase):
    older: dict[str, Any]
    newer: dict[str, Any]
    resolution: str
    severity: str = "medium"


class ContextPack(MemoBase):
    """The tight prompt-safe response shape returned to Hermes."""

    answer_context: str = ""
    decisions: list[RecallDecision] = Field(default_factory=list)
    facts: list[RecallFact] = Field(default_factory=list)
    chunks: list[RecallChunk] = Field(default_factory=list)
    summaries: list[RecallSummary] = Field(default_factory=list)
    source_snippets: list[SourceRef] = Field(default_factory=list)
    conflicts: list[ConflictEntry] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    warnings: list[str] = Field(default_factory=list)
    mode: str = RECALL_MODE_BALANCED
    latency_ms: int = 0
    embedding_table: str | None = None
    token_estimate: int = 0


class RecallResponse(MemoBase):
    context_pack: ContextPack
    query: str
    mode: str
    latency_ms: int
