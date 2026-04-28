"""Memory capture v2 schemas.

The capture classifier is allowed to emit multiple durable candidates for a
single turn. These schemas intentionally model the classifier contract rather
than the storage schema: workers decide which candidate actions can be promoted
immediately and which become audit/review events.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import Field, field_validator

from siqueira_memo.schemas.common import MemoBase

MemoryCaptureAction = Literal[
    "auto_save",
    "skip_noise",
    "merge",
    "supersede",
    "flag_conflict",
    "needs_review",
]
MemoryCandidateKind = Literal[
    "fact",
    "decision",
    "preference",
    "secret",
    "entity",
    "relationship",
    "summary",
]
MemorySensitivity = Literal["public", "internal", "private", "secret"]
MemoryRisk = Literal["low", "medium", "high", "critical"]


class MemoryRelationCandidate(MemoBase):
    """Classifier-proposed relationship to existing memory or source."""

    target_type: str | None = None
    target_id: uuid.UUID | None = None
    relationship_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None


class MemoryCandidate(MemoBase):
    """One candidate memory/action extracted from a single completed turn."""

    action: MemoryCaptureAction
    kind: MemoryCandidateKind
    statement: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    project: str | None = None
    topic: str | None = None
    entity_names: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.0, le=1.0)
    sensitivity: MemorySensitivity = "internal"
    risk: MemoryRisk = "low"
    rationale: str
    source_message_ids: list[uuid.UUID] = Field(default_factory=list)
    source_event_ids: list[uuid.UUID] = Field(default_factory=list)
    relation_to_existing: list[MemoryRelationCandidate] = Field(default_factory=list)
    review_reason: str | None = None

    @field_validator("statement", "rationale")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("field must not be empty")
        return text

    @field_validator("subject", "predicate", "object", "project", "topic", "review_reason")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class MemoryCaptureResult(MemoBase):
    """Full classifier result for one completed turn."""

    candidates: list[MemoryCandidate] = Field(default_factory=list)
    skipped_reason: str | None = None
    classifier_model: str | None = None
    prompt_version: str = "capture-v2"
