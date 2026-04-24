"""Discriminated-union payload schemas for ``memory_events.payload``. Plan §31.9."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter

from siqueira_memo.schemas.common import MemoBase


class _PayloadBase(MemoBase):
    event_type: str


class MessageReceivedPayload(_PayloadBase):
    event_type: Literal["message_received"] = "message_received"
    message_id: str
    role: str
    platform: str
    content_hash: str
    language: str | None = None


class AssistantMessageSentPayload(_PayloadBase):
    event_type: Literal["assistant_message_sent"] = "assistant_message_sent"
    message_id: str
    content_hash: str


class ToolCalledPayload(_PayloadBase):
    event_type: Literal["tool_called"] = "tool_called"
    tool_event_id: str
    tool_name: str


class ToolResultReceivedPayload(_PayloadBase):
    event_type: Literal["tool_result_received"] = "tool_result_received"
    tool_event_id: str
    tool_name: str
    exit_status: str
    output_hash: str | None = None
    output_size_bytes: int | None = None


class ArtifactCreatedPayload(_PayloadBase):
    event_type: Literal["artifact_created"] = "artifact_created"
    artifact_id: str
    path: str | None = None
    uri: str | None = None
    content_hash: str | None = None


class ArtifactModifiedPayload(_PayloadBase):
    event_type: Literal["artifact_modified"] = "artifact_modified"
    artifact_id: str
    content_hash: str | None = None


class SummaryCreatedPayload(_PayloadBase):
    event_type: Literal["summary_created"] = "summary_created"
    summary_id: str
    scope: Literal["session", "topic", "project"]


class FactExtractedPayload(_PayloadBase):
    event_type: Literal["fact_extracted"] = "fact_extracted"
    fact_id: str
    canonical_key: str
    status: str
    confidence: float


class DecisionRecordedPayload(_PayloadBase):
    event_type: Literal["decision_recorded"] = "decision_recorded"
    decision_id: str
    canonical_key: str
    status: str


class FactInvalidatedPayload(_PayloadBase):
    event_type: Literal["fact_invalidated"] = "fact_invalidated"
    fact_id: str
    reason: str
    superseded_by: str | None = None


class DecisionSupersededPayload(_PayloadBase):
    event_type: Literal["decision_superseded"] = "decision_superseded"
    decision_id: str
    superseded_by: str | None = None
    reason: str


class MemoryDeletedPayload(_PayloadBase):
    event_type: Literal["memory_deleted"] = "memory_deleted"
    target_type: str
    target_id: str
    mode: Literal["soft", "hard"]
    reason: str | None = None
    removed_chunks: int = 0
    removed_embeddings: int = 0


class UserCorrectionReceivedPayload(_PayloadBase):
    event_type: Literal["user_correction_received"] = "user_correction_received"
    correction_text: str
    target_type: str | None = None
    target_id: str | None = None


class HindsightImportedPayload(_PayloadBase):
    event_type: Literal["hindsight_imported"] = "hindsight_imported"
    source_id: str
    trust_level: str = "secondary"
    requires_verification: bool = True


class DelegationObservedPayload(_PayloadBase):
    event_type: Literal["delegation_observed"] = "delegation_observed"
    parent_session_id: str
    child_session_id: str | None = None
    task: str
    result: str
    toolsets: list[str] = []
    model: str | None = None
    created_at: datetime | None = None


class BuiltinMemoryMirrorPayload(_PayloadBase):
    event_type: Literal["builtin_memory_mirror"] = "builtin_memory_mirror"
    action: Literal["add", "replace", "remove"]
    target: Literal["memory", "user"]
    content_hash: str
    content_redacted: str | None = None
    selector: str | None = None


class HermesAuxCompactionPayload(_PayloadBase):
    event_type: Literal[
        "hermes_auxiliary_compaction_observed"
    ] = "hermes_auxiliary_compaction_observed"
    summary_text: str
    prefix: str = "[CONTEXT COMPACTION]"
    source_message_count: int = 0


class AnswerCardCreatedPayload(_PayloadBase):
    event_type: Literal["answer_card_created"] = "answer_card_created"
    question: str
    summary: str
    source_ids: list[str] = []


EventPayload = Annotated[
    MessageReceivedPayload
    | AssistantMessageSentPayload
    | ToolCalledPayload
    | ToolResultReceivedPayload
    | ArtifactCreatedPayload
    | ArtifactModifiedPayload
    | SummaryCreatedPayload
    | FactExtractedPayload
    | DecisionRecordedPayload
    | FactInvalidatedPayload
    | DecisionSupersededPayload
    | MemoryDeletedPayload
    | UserCorrectionReceivedPayload
    | HindsightImportedPayload
    | DelegationObservedPayload
    | BuiltinMemoryMirrorPayload
    | HermesAuxCompactionPayload
    | AnswerCardCreatedPayload,
    Field(discriminator="event_type"),
]


event_payload_adapter: TypeAdapter[EventPayload] = TypeAdapter(EventPayload)


def validate_event_payload(data: dict[str, Any]) -> _PayloadBase:
    """Validate a raw payload dict into the correct subtype."""
    return event_payload_adapter.validate_python(data)
