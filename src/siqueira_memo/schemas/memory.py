"""Memory-management schemas (remember/correct/forget/timeline/sources). Plan §4.4."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from siqueira_memo.schemas.common import MemoBase, SourceRef

MemoryTarget = Literal["fact", "decision", "message", "chunk", "summary", "entity", "session"]


class RememberRequest(MemoBase):
    profile_id: str | None = None
    session_id: str | None = None
    kind: Literal["fact", "decision"]
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    statement: str | None = None
    project: str | None = None
    topic: str | None = None
    rationale: str | None = None
    options_considered: list[dict[str, Any]] = Field(default_factory=list)
    tradeoffs: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.9
    source_event_ids: list[uuid.UUID] = Field(default_factory=list)
    source_message_ids: list[uuid.UUID] = Field(default_factory=list)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    reversible: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class RememberResponse(MemoBase):
    id: uuid.UUID
    kind: str
    status: str
    canonical_key: str
    event_id: uuid.UUID
    superseded: list[uuid.UUID] = Field(default_factory=list)


class CorrectRequest(MemoBase):
    profile_id: str | None = None
    session_id: str | None = None
    target_type: MemoryTarget
    target_id: uuid.UUID | None = None
    correction_text: str
    replacement: RememberRequest | None = None


class CorrectResponse(MemoBase):
    event_id: uuid.UUID
    invalidated: list[uuid.UUID] = Field(default_factory=list)
    superseded: list[uuid.UUID] = Field(default_factory=list)
    replacement_id: uuid.UUID | None = None


class ForgetRequest(MemoBase):
    profile_id: str | None = None
    target_type: MemoryTarget
    target_id: uuid.UUID
    mode: Literal["soft", "hard"] = "soft"
    reason: str | None = None
    scrub_raw: bool = False


class ForgetResponse(MemoBase):
    event_id: uuid.UUID
    target_id: uuid.UUID
    mode: str
    invalidated_facts: int = 0
    invalidated_decisions: int = 0
    removed_chunks: int = 0
    removed_embeddings: int = 0
    regenerated_summaries: int = 0


class TimelineRequest(MemoBase):
    profile_id: str | None = None
    entity: str | None = None
    project: str | None = None
    topic: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 50


class TimelineEntry(MemoBase):
    id: uuid.UUID
    kind: Literal["message", "decision", "fact", "summary", "event"]
    title: str
    preview: str | None = None
    status: str | None = None
    created_at: datetime


class TimelineResponse(MemoBase):
    entries: list[TimelineEntry]


class SourcesRequest(MemoBase):
    profile_id: str | None = None
    target_type: Literal["fact", "decision", "summary"]
    target_id: uuid.UUID


class SourcesResponse(MemoBase):
    sources: list[SourceRef]


RelationshipNodeType = Literal["fact", "decision", "entity", "summary", "message"]
RelationshipDirection = Literal["incoming", "outgoing", "both"]


class MemoryRelationshipCreateRequest(MemoBase):
    profile_id: str | None = None
    source_type: RelationshipNodeType
    source_id: uuid.UUID
    relationship_type: str
    target_type: RelationshipNodeType
    target_id: uuid.UUID
    confidence: float = 1.0
    rationale: str | None = None
    source_event_ids: list[uuid.UUID] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRelationshipListRequest(MemoBase):
    profile_id: str | None = None
    target_type: RelationshipNodeType
    target_id: uuid.UUID
    direction: RelationshipDirection = "both"
    include_inactive: bool = False
    limit: int = 100


class MemoryRelationshipItem(MemoBase):
    id: uuid.UUID
    profile_id: str
    source_type: str
    source_id: uuid.UUID
    relationship_type: str
    target_type: str
    target_id: uuid.UUID
    confidence: float
    rationale: str | None = None
    source_event_ids: list[uuid.UUID] = Field(default_factory=list)
    created_by: str
    status: str
    created_at: datetime


class MemoryRelationshipListResponse(MemoBase):
    relationships: list[MemoryRelationshipItem]


class EntityCardRequest(MemoBase):
    profile_id: str | None = None
    entity_id: uuid.UUID | None = None
    name: str | None = None
    entity_type: str | None = None
    fact_limit: int = Field(default=8, ge=1, le=50)
    decision_limit: int = Field(default=8, ge=1, le=50)
    relationship_limit: int = Field(default=40, ge=1, le=200)


class EntityMergeSuggestionsRequest(MemoBase):
    profile_id: str | None = None
    query: str | None = None
    entity_type: str | None = None
    limit: int = Field(default=50, ge=1, le=200)


class EntityMergeCandidateItem(MemoBase):
    entity_id: uuid.UUID
    name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    status: str


class EntityMergeSuggestionItem(MemoBase):
    source: EntityMergeCandidateItem
    target: EntityMergeCandidateItem
    confidence: float
    reasons: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    shared_tokens: list[str] = Field(default_factory=list)


class EntityMergeSuggestionsResponse(MemoBase):
    suggestions: list[EntityMergeSuggestionItem]
    total: int


class EntityMergeReviewRequest(MemoBase):
    profile_id: str | None = None
    action: Literal["merge", "reject"]
    source_entity_id: uuid.UUID
    target_entity_id: uuid.UUID
    reason: str | None = None


class EntityMergeReviewResponse(MemoBase):
    action: str
    source_entity_id: uuid.UUID
    target_entity_id: uuid.UUID
    status: str
    moved_aliases: int = 0
    moved_relationships: int = 0
    event_id: uuid.UUID | None = None


class EntityListRequest(MemoBase):
    profile_id: str | None = None
    query: str | None = None
    entity_type: str | None = None
    project_scope: Literal["all", "project", "global"] = "all"
    project: str | None = None
    topic: str | None = None
    status: str | None = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class EntityListItem(MemoBase):
    entity_id: uuid.UUID
    name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    status: str
    description: str | None = None
    projects: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    source_count: int
    relationship_count: int
    confidence: float
    last_updated: datetime | None = None


class EntityListResponse(MemoBase):
    entities: list[EntityListItem]
    total: int


class EntityCardMemoryItem(MemoBase):
    id: uuid.UUID
    kind: str
    text: str
    project: str | None = None
    topic: str | None = None
    status: str
    confidence: float
    source_event_ids: list[uuid.UUID] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class EntityCardRelationshipItem(MemoBase):
    id: uuid.UUID
    source_type: str
    source_id: uuid.UUID
    relationship_type: str
    target_type: str
    target_id: uuid.UUID
    confidence: float
    rationale: str | None = None


class EntityCardResponse(MemoBase):
    entity_id: uuid.UUID
    name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    status: str
    description: str | None = None
    projects: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    latest_facts: list[EntityCardMemoryItem] = Field(default_factory=list)
    active_decisions: list[EntityCardMemoryItem] = Field(default_factory=list)
    related_secrets: list[EntityCardMemoryItem] = Field(default_factory=list)
    relationships: list[EntityCardRelationshipItem] = Field(default_factory=list)
    source_count: int
    confidence: float
    last_updated: datetime | None = None
    conflict_count: int
    warnings: list[str] = Field(default_factory=list)
