"""Pydantic schemas for API boundaries and event payloads."""

from __future__ import annotations

from siqueira_memo.schemas.admin import (
    AdminSearchHit,
    AdminSearchRequest,
    AdminSearchResponse,
    HealthStatus,
)
from siqueira_memo.schemas.common import ExtractorMetadata, MemoBase, SensitivityField, SourceRef
from siqueira_memo.schemas.event_payloads import (
    EventPayload,
    validate_event_payload,
)
from siqueira_memo.schemas.ingest import (
    ArtifactIngestIn,
    ArtifactIngestOut,
    BuiltinMemoryMirrorIn,
    DelegationObservationIn,
    GenericEventIn,
    GenericEventOut,
    HermesAuxCompactionIn,
    MessageIngestIn,
    MessageIngestOut,
    ToolEventIngestIn,
    ToolEventIngestOut,
)
from siqueira_memo.schemas.memory import (
    CorrectRequest,
    CorrectResponse,
    ForgetRequest,
    ForgetResponse,
    RememberRequest,
    RememberResponse,
    SourcesRequest,
    SourcesResponse,
    TimelineEntry,
    TimelineRequest,
    TimelineResponse,
)
from siqueira_memo.schemas.recall import (
    ConflictEntry,
    ContextPack,
    RecallChunk,
    RecallDecision,
    RecallFact,
    RecallRequest,
    RecallResponse,
    RecallSummary,
)

__all__ = [
    "AdminSearchHit",
    "AdminSearchRequest",
    "AdminSearchResponse",
    "ArtifactIngestIn",
    "ArtifactIngestOut",
    "BuiltinMemoryMirrorIn",
    "ConflictEntry",
    "ContextPack",
    "CorrectRequest",
    "CorrectResponse",
    "DelegationObservationIn",
    "EventPayload",
    "ExtractorMetadata",
    "ForgetRequest",
    "ForgetResponse",
    "GenericEventIn",
    "GenericEventOut",
    "HealthStatus",
    "HermesAuxCompactionIn",
    "MemoBase",
    "MessageIngestIn",
    "MessageIngestOut",
    "RecallChunk",
    "RecallDecision",
    "RecallFact",
    "RecallRequest",
    "RecallResponse",
    "RecallSummary",
    "RememberRequest",
    "RememberResponse",
    "SensitivityField",
    "SourceRef",
    "SourcesRequest",
    "SourcesResponse",
    "TimelineEntry",
    "TimelineRequest",
    "TimelineResponse",
    "ToolEventIngestIn",
    "ToolEventIngestOut",
    "validate_event_payload",
]
