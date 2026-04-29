"""Admin search + introspection endpoints. Plan §9.2 / §21 / §25."""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from fastapi import APIRouter, Body, HTTPException, Response, status
from sqlalchemy import func, literal, or_, select

from siqueira_memo.api.deps import AuthDep, ProfileDep, SessionDep
from siqueira_memo.config import get_settings
from siqueira_memo.models import (
    Chunk,
    ChunkEmbeddingBGEM3,
    ChunkEmbeddingMock,
    ChunkEmbeddingOpenAITEL3,
    Decision,
    DecisionSource,
    Entity,
    EntityAlias,
    Fact,
    FactSource,
    MemoryConflict,
    MemoryEvent,
    MemoryRelationship,
    Message,
    SessionSummary,
    TopicSummary,
)
from siqueira_memo.models.constants import (
    RELATIONSHIP_BELONGS_TO_ENTITY,
    STATUS_ACTIVE,
    STATUS_MERGED,
)
from siqueira_memo.schemas.admin import AdminSearchHit, AdminSearchRequest, AdminSearchResponse
from siqueira_memo.schemas.audit import AuditEntrySchema, AuditRequest, AuditResponse
from siqueira_memo.schemas.conflicts import (
    ConflictItem,
    ConflictResolveRequest,
    ConflictResolveResponse,
    ConflictScanResponse,
)
from siqueira_memo.schemas.ingest import GenericEventIn
from siqueira_memo.schemas.memory import (
    EntityCardRequest,
    EntityCardResponse,
    EntityListItem,
    EntityListRequest,
    EntityListResponse,
    EntityMergeReviewRequest,
    EntityMergeReviewResponse,
    EntityMergeSuggestionItem,
    EntityMergeSuggestionsRequest,
    EntityMergeSuggestionsResponse,
    MemoryRelationshipItem,
    MemoryRelationshipListRequest,
    MemoryRelationshipListResponse,
)
from siqueira_memo.services.conflict_service import ConflictService
from siqueira_memo.services.entity_card_service import EntityCardService
from siqueira_memo.services.entity_merge_service import EntityMergeService
from siqueira_memo.services.ingest_service import IngestService
from siqueira_memo.services.memory_version_service import MemoryVersionService
from siqueira_memo.services.relationship_service import RelationshipService
from siqueira_memo.services.retention_service import AuditLog
from siqueira_memo.services.secret_policy import (
    is_secret_metadata,
    masked_preview,
    sanitize_metadata,
    secret_ref,
    secret_value_for_reveal,
)

router = APIRouter(prefix="/v1/admin")


_PREVIEW = 200


def _admin_preview(text: str, metadata: dict[str, Any] | None) -> str:
    preview = masked_preview(text or "", metadata) if is_secret_metadata(metadata) else (text or "")
    return preview[:_PREVIEW]


def _relationship_item(row: MemoryRelationship) -> MemoryRelationshipItem:
    return MemoryRelationshipItem(
        id=row.id,
        profile_id=row.profile_id,
        source_type=row.source_type,
        source_id=row.source_id,
        relationship_type=row.relationship_type,
        target_type=row.target_type,
        target_id=row.target_id,
        confidence=float(row.confidence or 0.0),
        rationale=row.rationale,
        source_event_ids=list(row.source_event_ids or []),
        created_by=row.created_by,
        status=row.status,
        created_at=row.created_at,
    )


def _apply_project_scope(stmt: Any, model: type[Any], payload: AdminSearchRequest) -> Any:
    if payload.project_scope == "global":
        return stmt.where(model.project.is_(None))
    if payload.project_scope == "project":
        if not payload.project:
            return stmt.where(literal(False))
        return stmt.where(model.project == payload.project)
    if payload.project:
        return stmt.where(model.project == payload.project)
    return stmt


def _dedupe_strings(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


@router.post("/entities/merge-suggestions", response_model=EntityMergeSuggestionsResponse)
async def entity_merge_suggestions(
    payload: EntityMergeSuggestionsRequest,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> EntityMergeSuggestionsResponse:
    """Return source-backed ambiguous entity merge suggestions for admin review."""
    svc = EntityMergeService(profile_id=payload.profile_id or profile_id, actor="admin_api")
    suggestions = await svc.suggest(
        session,
        query=payload.query,
        entity_type=payload.entity_type,
        limit=payload.limit,
    )
    return EntityMergeSuggestionsResponse(
        suggestions=[EntityMergeSuggestionItem.model_validate(asdict(item)) for item in suggestions],
        total=len(suggestions),
    )


@router.post("/entities/merge-review", response_model=EntityMergeReviewResponse)
async def entity_merge_review(
    payload: EntityMergeReviewRequest,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> EntityMergeReviewResponse:
    """Apply or reject one admin-reviewed entity merge suggestion."""
    svc = EntityMergeService(profile_id=payload.profile_id or profile_id, actor="admin_api")
    try:
        if payload.action == "merge":
            result = await svc.merge(
                session,
                source_entity_id=payload.source_entity_id,
                target_entity_id=payload.target_entity_id,
                reason=payload.reason,
            )
        else:
            result = await svc.reject(
                session,
                source_entity_id=payload.source_entity_id,
                target_entity_id=payload.target_entity_id,
                reason=payload.reason,
            )
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return EntityMergeReviewResponse.model_validate(asdict(result))


@router.post("/entities/list", response_model=EntityListResponse)
async def list_entities(
    payload: EntityListRequest,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> EntityListResponse:
    """List source-backed entity cards for the admin Entities tab."""
    profile_filter = payload.profile_id or profile_id
    alias_match_ids: list[uuid.UUID] = []
    pattern = f"%{payload.query}%" if payload.query else None
    if pattern:
        alias_match_ids = list(
            (
                await session.execute(
                    select(EntityAlias.entity_id)
                    .where(EntityAlias.profile_id == profile_filter)
                    .where(EntityAlias.status == STATUS_ACTIVE)
                    .where(
                        or_(
                            EntityAlias.alias.ilike(pattern),
                            EntityAlias.alias_normalized.ilike(pattern),
                        )
                    )
                )
            )
            .scalars()
            .all()
        )

    stmt = select(Entity).where(Entity.profile_id == profile_filter)
    if payload.status:
        stmt = stmt.where(Entity.status == payload.status)
    else:
        stmt = stmt.where(Entity.status != STATUS_MERGED)
    if payload.entity_type:
        stmt = stmt.where(Entity.type == payload.entity_type)
    if pattern:
        stmt = stmt.where(
            or_(
                Entity.name.ilike(pattern),
                Entity.name_normalized.ilike(pattern),
                Entity.id.in_(alias_match_ids) if alias_match_ids else literal(False),
            )
        )
    stmt = stmt.order_by(Entity.updated_at.desc(), Entity.name.asc())
    rows = list((await session.execute(stmt)).scalars().all())
    entity_ids = [row.id for row in rows]

    aliases_by_entity: dict[uuid.UUID, list[str]] = defaultdict(list)
    relationships_by_entity: dict[uuid.UUID, list[MemoryRelationship]] = defaultdict(list)
    fact_ids_by_entity: dict[uuid.UUID, set[uuid.UUID]] = defaultdict(set)
    decision_ids_by_entity: dict[uuid.UUID, set[uuid.UUID]] = defaultdict(set)
    projects_by_entity: dict[uuid.UUID, set[str]] = defaultdict(set)
    topics_by_entity: dict[uuid.UUID, set[str]] = defaultdict(set)
    updated_by_entity: dict[uuid.UUID, list[datetime]] = defaultdict(list)
    confidence_by_entity: dict[uuid.UUID, list[float]] = defaultdict(list)

    if entity_ids:
        alias_rows = list(
            (
                await session.execute(
                    select(EntityAlias)
                    .where(EntityAlias.profile_id == profile_filter)
                    .where(EntityAlias.status == STATUS_ACTIVE)
                    .where(EntityAlias.entity_id.in_(entity_ids))
                    .order_by(EntityAlias.created_at.asc(), EntityAlias.alias.asc())
                )
            )
            .scalars()
            .all()
        )
        for alias in alias_rows:
            aliases_by_entity[alias.entity_id].append(alias.alias)

        relationships = list(
            (
                await session.execute(
                    select(MemoryRelationship)
                    .where(MemoryRelationship.profile_id == profile_filter)
                    .where(MemoryRelationship.status == STATUS_ACTIVE)
                    .where(
                        or_(
                            (MemoryRelationship.source_type == "entity")
                            & MemoryRelationship.source_id.in_(entity_ids),
                            (MemoryRelationship.target_type == "entity")
                            & MemoryRelationship.target_id.in_(entity_ids),
                        )
                    )
                )
            )
            .scalars()
            .all()
        )
        for rel in relationships:
            entity_id = rel.source_id if rel.source_type == "entity" else rel.target_id
            relationships_by_entity[entity_id].append(rel)
            if rel.relationship_type == RELATIONSHIP_BELONGS_TO_ENTITY:
                confidence_by_entity[entity_id].append(float(rel.confidence or 0.0))
            if rel.source_type == "fact":
                fact_ids_by_entity[entity_id].add(rel.source_id)
            if rel.target_type == "fact":
                fact_ids_by_entity[entity_id].add(rel.target_id)
            if rel.source_type == "decision":
                decision_ids_by_entity[entity_id].add(rel.source_id)
            if rel.target_type == "decision":
                decision_ids_by_entity[entity_id].add(rel.target_id)

        all_fact_ids = {fact_id for ids in fact_ids_by_entity.values() for fact_id in ids}
        all_decision_ids = {
            decision_id for ids in decision_ids_by_entity.values() for decision_id in ids
        }
        fact_by_id: dict[uuid.UUID, Fact] = {}
        if all_fact_ids:
            fact_by_id = {
                row.id: row
                for row in (
                    await session.execute(
                        select(Fact)
                        .where(Fact.profile_id == profile_filter)
                        .where(Fact.status == STATUS_ACTIVE)
                        .where(Fact.id.in_(all_fact_ids))
                    )
                )
                .scalars()
                .all()
            }
        decision_by_id: dict[uuid.UUID, Decision] = {}
        if all_decision_ids:
            decision_by_id = {
                row.id: row
                for row in (
                    await session.execute(
                        select(Decision)
                        .where(Decision.profile_id == profile_filter)
                        .where(Decision.status == STATUS_ACTIVE)
                        .where(Decision.id.in_(all_decision_ids))
                    )
                )
                .scalars()
                .all()
            }
        for entity_id, fact_ids in fact_ids_by_entity.items():
            for fact_id in fact_ids:
                fact = fact_by_id.get(fact_id)
                if fact is None:
                    continue
                if fact.project:
                    projects_by_entity[entity_id].add(fact.project)
                if fact.topic:
                    topics_by_entity[entity_id].add(fact.topic)
                updated_by_entity[entity_id].append(fact.updated_at or fact.created_at)
        for entity_id, decision_ids in decision_ids_by_entity.items():
            for decision_id in decision_ids:
                decision = decision_by_id.get(decision_id)
                if decision is None:
                    continue
                if decision.project:
                    projects_by_entity[entity_id].add(decision.project)
                if decision.topic:
                    topics_by_entity[entity_id].add(decision.topic)
                updated_by_entity[entity_id].append(decision.updated_at or decision.decided_at)

    items: list[EntityListItem] = []
    for row in rows:
        aliases = _dedupe_strings([row.name, *(row.aliases or []), *aliases_by_entity[row.id]])
        projects = sorted(projects_by_entity[row.id])
        topics = sorted(topics_by_entity[row.id])
        if payload.project_scope == "global" and projects:
            continue
        if payload.project and payload.project not in projects:
            continue
        if payload.topic and payload.topic not in topics:
            continue
        updated_values = [row.updated_at, *updated_by_entity[row.id]]
        confidence_values = confidence_by_entity[row.id]
        items.append(
            EntityListItem(
                entity_id=row.id,
                name=row.name,
                entity_type=row.type,
                aliases=aliases,
                status=row.status,
                description=row.description,
                projects=projects,
                topics=topics,
                source_count=len(fact_ids_by_entity[row.id] | decision_ids_by_entity[row.id]),
                relationship_count=len(relationships_by_entity[row.id]),
                confidence=(
                    sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
                ),
                last_updated=max((value for value in updated_values if value is not None), default=None),
            )
        )

    total = len(items)
    paged = items[payload.offset : payload.offset + payload.limit]
    return EntityListResponse(entities=paged, total=total)


@router.post("/entities/card", response_model=EntityCardResponse)
async def entity_card(
    payload: EntityCardRequest,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> EntityCardResponse:
    svc = EntityCardService(profile_id=payload.profile_id or profile_id)
    try:
        card = await svc.build_card(
            session,
            entity_id=payload.entity_id,
            name=payload.name,
            entity_type=payload.entity_type,
            fact_limit=payload.fact_limit,
            decision_limit=payload.decision_limit,
            relationship_limit=payload.relationship_limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return EntityCardResponse.model_validate(asdict(card))


@router.post("/relationships/list", response_model=MemoryRelationshipListResponse)
async def list_relationships(
    payload: MemoryRelationshipListRequest,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> MemoryRelationshipListResponse:
    svc = RelationshipService(profile_id=payload.profile_id or profile_id, actor="admin_api")
    try:
        rows = await svc.list_for_memory(
            session,
            target_type=payload.target_type,
            target_id=payload.target_id,
            direction=payload.direction,
            include_inactive=payload.include_inactive,
            limit=payload.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return MemoryRelationshipListResponse(relationships=[_relationship_item(row) for row in rows])


@router.post("/search", response_model=AdminSearchResponse)
async def search(
    payload: AdminSearchRequest,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> AdminSearchResponse:
    profile_filter = payload.profile_id or profile_id
    hits: list[AdminSearchHit] = []

    if payload.target_type == "message":
        m_stmt = select(Message).where(Message.profile_id == profile_filter)
        if payload.query:
            pattern = f"%{payload.query}%"
            m_stmt = m_stmt.where(Message.content_redacted.ilike(pattern))
        m_stmt = _apply_project_scope(m_stmt, Message, payload)
        if payload.topic:
            m_stmt = m_stmt.where(Message.topic == payload.topic)
        if payload.since:
            m_stmt = m_stmt.where(Message.created_at >= payload.since)
        if payload.until:
            m_stmt = m_stmt.where(Message.created_at <= payload.until)
        total = (
            await session.execute(select(func.count()).select_from(m_stmt.subquery()))
        ).scalar_one()
        m_stmt = (
            m_stmt.order_by(Message.created_at.desc())
            .offset(payload.offset)
            .limit(payload.limit)
        )
        for m in (await session.execute(m_stmt)).scalars():
            hits.append(
                AdminSearchHit(
                    id=m.id,
                    target_type="message",
                    preview=(m.content_redacted or "")[:_PREVIEW],
                    project=m.project,
                    topic=m.topic,
                    status=None,
                    created_at=m.created_at,
                )
            )
        return AdminSearchResponse(hits=hits, total=total)

    if payload.target_type == "fact":
        f_stmt = select(Fact).where(Fact.profile_id == profile_filter)
        if payload.query:
            pattern = f"%{payload.query}%"
            f_stmt = f_stmt.where(Fact.statement.ilike(pattern))
        f_stmt = _apply_project_scope(f_stmt, Fact, payload)
        if payload.status:
            f_stmt = f_stmt.where(Fact.status == payload.status)
        total = (
            await session.execute(select(func.count()).select_from(f_stmt.subquery()))
        ).scalar_one()
        f_stmt = (
            f_stmt.order_by(Fact.created_at.desc())
            .offset(payload.offset)
            .limit(payload.limit)
        )
        for f in (await session.execute(f_stmt)).scalars():
            hits.append(
                AdminSearchHit(
                    id=f.id,
                    target_type="fact",
                    preview=_admin_preview(f.statement, f.extra_metadata),
                    project=f.project,
                    topic=f.topic,
                    status=f.status,
                    created_at=f.created_at,
                )
            )
        return AdminSearchResponse(hits=hits, total=total)

    if payload.target_type == "decision":
        d_stmt = select(Decision).where(Decision.profile_id == profile_filter)
        if payload.query:
            pattern = f"%{payload.query}%"
            d_stmt = d_stmt.where(Decision.decision.ilike(pattern))
        d_stmt = _apply_project_scope(d_stmt, Decision, payload)
        if payload.topic:
            d_stmt = d_stmt.where(Decision.topic == payload.topic)
        if payload.status:
            d_stmt = d_stmt.where(Decision.status == payload.status)
        total = (
            await session.execute(select(func.count()).select_from(d_stmt.subquery()))
        ).scalar_one()
        d_stmt = (
            d_stmt.order_by(Decision.decided_at.desc())
            .offset(payload.offset)
            .limit(payload.limit)
        )
        for d in (await session.execute(d_stmt)).scalars():
            hits.append(
                AdminSearchHit(
                    id=d.id,
                    target_type="decision",
                    preview=_admin_preview(d.decision, d.extra_metadata),
                    project=d.project,
                    topic=d.topic,
                    status=d.status,
                    created_at=d.decided_at,
                )
            )
        return AdminSearchResponse(hits=hits, total=total)

    if payload.target_type == "summary":
        s_stmt = select(SessionSummary).where(SessionSummary.profile_id == profile_filter)
        if payload.query:
            pattern = f"%{payload.query}%"
            s_stmt = s_stmt.where(SessionSummary.summary_short.ilike(pattern))
        total = (
            await session.execute(select(func.count()).select_from(s_stmt.subquery()))
        ).scalar_one()
        s_stmt = (
            s_stmt.order_by(SessionSummary.created_at.desc())
            .offset(payload.offset)
            .limit(payload.limit)
        )
        for s in (await session.execute(s_stmt)).scalars():
            hits.append(
                AdminSearchHit(
                    id=s.id,
                    target_type="summary",
                    preview=(s.summary_short or "")[:_PREVIEW],
                    project=None,
                    topic=None,
                    status=s.status,
                    created_at=s.created_at,
                )
            )
        return AdminSearchResponse(hits=hits, total=total)

    return AdminSearchResponse(hits=[], total=0)


async def _count_by_project(session: SessionDep, model: type[Any], profile_id: str) -> list[tuple[str, int]]:
    rows = (
        await session.execute(
            select(model.project, func.count())
            .where(model.profile_id == profile_id)
            .where(model.project.is_not(None))
            .group_by(model.project)
        )
    ).all()
    return [(str(project), int(count)) for project, count in rows if project]


@router.post("/projects")
async def projects(
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
    payload: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    """Return project cards with fact/decision/message/summary counts."""
    profile_filter = str(payload.get("profile_id") or profile_id)
    project_map: dict[str, dict[str, Any]] = {}

    def ensure_project(name: str) -> dict[str, Any]:
        if name not in project_map:
            project_map[name] = {
                "project": name,
                "facts": 0,
                "decisions": 0,
                "messages": 0,
                "summaries": 0,
                "topics": {},
            }
        return project_map[name]

    for name, count in await _count_by_project(session, Fact, profile_filter):
        ensure_project(name)["facts"] = count
    for name, count in await _count_by_project(session, Decision, profile_filter):
        ensure_project(name)["decisions"] = count
    for name, count in await _count_by_project(session, Message, profile_filter):
        ensure_project(name)["messages"] = count
    for name, count in await _count_by_project(session, TopicSummary, profile_filter):
        ensure_project(name)["summaries"] = count

    topic_rows = (
        await session.execute(
            select(Fact.project, Fact.topic, func.count())
            .where(Fact.profile_id == profile_filter)
            .where(Fact.project.is_not(None))
            .where(Fact.topic.is_not(None))
            .group_by(Fact.project, Fact.topic)
        )
    ).all()
    for project, topic, count in topic_rows:
        if project and topic:
            topics = ensure_project(str(project))["topics"]
            topics.setdefault(str(topic), {"topic": str(topic), "facts": 0, "decisions": 0})[
                "facts"
            ] = int(count)

    decision_topic_rows = (
        await session.execute(
            select(Decision.project, Decision.topic, func.count())
            .where(Decision.profile_id == profile_filter)
            .where(Decision.project.is_not(None))
            .group_by(Decision.project, Decision.topic)
        )
    ).all()
    for project, topic, count in decision_topic_rows:
        if project and topic:
            topics = ensure_project(str(project))["topics"]
            topics.setdefault(str(topic), {"topic": str(topic), "facts": 0, "decisions": 0})[
                "decisions"
            ] = int(count)

    projects_out = []
    for item in project_map.values():
        item["total"] = item["facts"] + item["decisions"] + item["messages"] + item["summaries"]
        item["topics"] = sorted(item["topics"].values(), key=lambda t: t["topic"])
        projects_out.append(item)
    projects_out.sort(key=lambda p: (-int(p["total"]), str(p["project"])))
    return {"projects": projects_out}


async def _count_rows(session: SessionDep, stmt: Any) -> int:
    return int((await session.execute(stmt)).scalar_one() or 0)


@router.post("/capture")
async def capture_stats(
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
    payload: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    """Return aggressive memory-capture observability counters for the dashboard."""
    settings = get_settings()
    profile_filter = str(payload.get("profile_id") or profile_id)
    since = datetime.now(UTC) - timedelta(days=1)

    raw_turns_today = await _count_rows(
        session,
        select(func.count()).select_from(Message).where(
            Message.profile_id == profile_filter,
            Message.created_at >= since,
        ),
    )
    chunks_created = await _count_rows(
        session,
        select(func.count()).select_from(Chunk).where(Chunk.profile_id == profile_filter),
    )
    embeddings_created = 0
    for embedding_model in (ChunkEmbeddingMock, ChunkEmbeddingOpenAITEL3, ChunkEmbeddingBGEM3):
        embeddings_created += await _count_rows(
            session,
            select(func.count())
            .select_from(embedding_model)
            .join(Chunk, Chunk.id == embedding_model.chunk_id)
            .where(Chunk.profile_id == profile_filter),
        )
    structured_facts = await _count_rows(
        session,
        select(func.count()).select_from(Fact).where(Fact.profile_id == profile_filter),
    )
    structured_decisions = await _count_rows(
        session,
        select(func.count()).select_from(Decision).where(Decision.profile_id == profile_filter),
    )
    skipped_sensitive = await _count_rows(
        session,
        select(func.count()).select_from(Message).where(
            Message.profile_id == profile_filter,
            Message.sensitivity == "sensitive",
        ),
    )
    global_memories = await _count_rows(
        session,
        select(func.count()).select_from(Fact).where(
            Fact.profile_id == profile_filter,
            Fact.project.is_(None),
        ),
    ) + await _count_rows(
        session,
        select(func.count()).select_from(Decision).where(
            Decision.profile_id == profile_filter,
            Decision.project.is_(None),
        ),
    )
    capture_event_types = [
        "capture_classifier_called",
        "capture_classifier_failed",
        "capture_classifier_fallback",
        "capture_candidates_extracted",
        "capture_candidate_auto_saved",
        "capture_classifier_skip",
        "capture_candidate_needs_review",
        "capture_secret_candidate_saved",
    ]
    capture_event_counts: dict[str, int] = {}
    for event_type in capture_event_types:
        capture_event_counts[event_type] = await _count_rows(
            session,
            select(func.count()).select_from(MemoryEvent).where(
                MemoryEvent.profile_id == profile_filter,
                MemoryEvent.event_type == event_type,
                MemoryEvent.created_at >= since,
            ),
        )

    skip_reason_rows = (
        await session.execute(
            select(MemoryEvent.payload)
            .where(
                MemoryEvent.profile_id == profile_filter,
                MemoryEvent.event_type == "capture_classifier_skip",
                MemoryEvent.created_at >= since,
            )
            .order_by(MemoryEvent.created_at.desc())
            .limit(50)
        )
    ).scalars().all()
    skip_reasons: dict[str, int] = {}
    for event_payload in skip_reason_rows:
        reason = str(event_payload.get("reason") or event_payload.get("skipped_reason") or "unknown")
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    fact_rows = (
        await session.execute(
            select(Fact)
            .where(Fact.profile_id == profile_filter, Fact.project.is_(None))
            .order_by(Fact.created_at.desc())
            .limit(5)
        )
    ).scalars().all()
    decision_rows = (
        await session.execute(
            select(Decision)
            .where(Decision.profile_id == profile_filter, Decision.project.is_(None))
            .order_by(Decision.decided_at.desc())
            .limit(5)
        )
    ).scalars().all()
    recent_global_memories: list[dict[str, Any]] = [
        {
            "id": row.id,
            "target_type": "fact",
            "preview": _admin_preview(row.statement, row.extra_metadata),
            "project": row.project,
            "topic": row.topic,
            "status": row.status,
            "created_at": row.created_at,
        }
        for row in fact_rows
    ] + [
        {
            "id": row.id,
            "target_type": "decision",
            "preview": _admin_preview(row.decision, row.extra_metadata),
            "project": row.project,
            "topic": row.topic,
            "status": row.status,
            "created_at": row.decided_at,
        }
        for row in decision_rows
    ]
    recent_global_memories.sort(key=lambda item: item["created_at"], reverse=True)

    return {
        "mode": settings.memory_capture_mode,
        "target_ratio": settings.memory_capture_target_ratio,
        "raw_turns_saved_today": raw_turns_today,
        "chunks_created": chunks_created,
        "embeddings_created": embeddings_created,
        "structured_facts": structured_facts,
        "structured_decisions": structured_decisions,
        "skipped_sensitive": skipped_sensitive,
        "global_memories": global_memories,
        "classifier_calls_today": capture_event_counts["capture_classifier_called"],
        "classifier_failures_today": capture_event_counts["capture_classifier_failed"],
        "fallbacks_today": capture_event_counts["capture_classifier_fallback"],
        "candidate_batches_today": capture_event_counts["capture_candidates_extracted"],
        "candidate_auto_saved_today": capture_event_counts["capture_candidate_auto_saved"],
        "candidate_skipped_today": capture_event_counts["capture_classifier_skip"],
        "candidate_needs_review_today": capture_event_counts["capture_candidate_needs_review"],
        "secret_candidates_saved_today": capture_event_counts["capture_secret_candidate_saved"],
        "skip_reasons": skip_reasons,
        "recent_global_memories": recent_global_memories[:5],
    }


def _fact_to_detail(row: Fact) -> dict[str, Any]:
    secret = is_secret_metadata(row.extra_metadata)
    preview = masked_preview(row.statement, row.extra_metadata) if secret else row.statement
    return {
        "id": row.id,
        "target_type": "fact",
        "subject": masked_preview(row.subject, row.extra_metadata) if secret else row.subject,
        "predicate": row.predicate,
        "object": masked_preview(row.object, row.extra_metadata) if secret else row.object,
        "statement": preview,
        "masked_preview": preview if secret else None,
        "sensitivity": "secret" if secret else str((row.extra_metadata or {}).get("sensitivity") or "internal"),
        "secret_ref": secret_ref(row.extra_metadata),
        "secret_masked": secret,
        "project": row.project,
        "topic": row.topic,
        "status": row.status,
        "confidence": row.confidence,
        "valid_from": row.valid_from,
        "valid_to": row.valid_to,
        "source_event_ids": row.source_event_ids,
        "source_message_ids": row.source_message_ids,
        "superseded_by": row.superseded_by,
        "metadata": sanitize_metadata(row.extra_metadata),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def _decision_to_detail(row: Decision) -> dict[str, Any]:
    secret = is_secret_metadata(row.extra_metadata)
    preview = masked_preview(row.decision, row.extra_metadata) if secret else row.decision
    return {
        "id": row.id,
        "target_type": "decision",
        "topic": row.topic,
        "decision": preview,
        "statement": preview,
        "masked_preview": preview if secret else None,
        "sensitivity": "secret" if secret else str((row.extra_metadata or {}).get("sensitivity") or "internal"),
        "secret_ref": secret_ref(row.extra_metadata),
        "secret_masked": secret,
        "context": masked_preview(row.context, row.extra_metadata) if secret else row.context,
        "rationale": masked_preview(row.rationale, row.extra_metadata) if secret else row.rationale,
        "tradeoffs": row.tradeoffs,
        "options_considered": row.options_considered,
        "project": row.project,
        "status": row.status,
        "reversible": row.reversible,
        "source_event_ids": row.source_event_ids,
        "source_message_ids": row.source_message_ids,
        "superseded_by": row.superseded_by,
        "metadata": sanitize_metadata(row.extra_metadata),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "decided_at": row.decided_at,
    }


async def _source_details(
    session: SessionDep, *, target_type: str, target_id: Any
) -> list[dict[str, Any]]:
    source_rows: list[Any]
    if target_type == "fact":
        source_rows = list(
            (
                await session.execute(select(FactSource).where(FactSource.fact_id == target_id))
            ).scalars().all()
        )
    elif target_type == "decision":
        source_rows = list(
            (
                await session.execute(
                    select(DecisionSource).where(DecisionSource.decision_id == target_id)
                )
            ).scalars().all()
        )
    else:
        return []
    event_ids = [row.event_id for row in source_rows]
    if not event_ids:
        return []
    events = (
        await session.execute(select(MemoryEvent).where(MemoryEvent.id.in_(event_ids)))
    ).scalars().all()
    event_by_id = {event.id: event for event in events}
    details = []
    for source_row in source_rows:
        event = event_by_id.get(source_row.event_id)
        details.append(
            {
                "event_id": source_row.event_id,
                "message_id": source_row.message_id,
                "event_type": event.event_type if event else None,
                "source": event.source if event else None,
                "actor": event.actor if event else None,
                "created_at": event.created_at if event else None,
                "payload": event.payload if event else None,
            }
        )
    return details


def _versioned_target(value: Any) -> str:
    target_type = str(value or "")
    if target_type not in {"fact", "decision"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="target_type must be fact or decision")
    return target_type


def _payload_uuid(value: Any, field: str) -> uuid.UUID:
    if not value:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{field} required")
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"invalid {field}") from exc


def _payload_int(value: Any, field: str) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"invalid {field}") from exc
    if parsed <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{field} must be positive")
    return parsed


@router.post("/versions/diff")
async def version_diff(
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
    payload: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    """Return a field-level diff between two persisted memory versions."""
    profile_filter = str(payload.get("profile_id") or profile_id)
    target_type = cast(Any, _versioned_target(payload.get("target_type")))
    target_id = _payload_uuid(payload.get("target_id"), "target_id")
    try:
        diff = await MemoryVersionService().diff(
            session,
            target_type,
            target_id,
            _payload_int(payload.get("from_version"), "from_version"),
            _payload_int(payload.get("to_version"), "to_version"),
            profile_id=profile_filter,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return {
        "target_type": diff.target_type,
        "target_id": diff.target_id,
        "from_version": diff.from_version,
        "to_version": diff.to_version,
        "changes": diff.changes,
    }


@router.post("/versions/rollback")
async def version_rollback(
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
    payload: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    """Rollback a fact/decision row to a previously recorded version snapshot."""
    profile_filter = str(payload.get("profile_id") or profile_id)
    target_type = cast(Any, _versioned_target(payload.get("target_type")))
    target_id = _payload_uuid(payload.get("target_id"), "target_id")
    try:
        result = await MemoryVersionService().rollback(
            session,
            target_type=target_type,
            target_id=target_id,
            to_version=_payload_int(payload.get("to_version"), "to_version"),
            profile_id=profile_filter,
            actor="admin",
            reason=str(payload.get("reason") or "admin rollback"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {
        "target_type": result.target_type,
        "target_id": result.target_id,
        "rolled_back": result.rolled_back,
        "rollback_to_version": result.rollback_to_version,
        "new_version": result.new_version,
        "event_id": result.event_id,
    }


@router.post("/detail")
async def detail(
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
    payload: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    """Return a full fact/decision/message/summary payload plus provenance."""
    profile_filter = str(payload.get("profile_id") or profile_id)
    target_type = str(payload.get("target_type") or "")
    target_id = payload.get("target_id")
    if not target_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="target_id required")

    if target_type == "fact":
        fact_row = (
            await session.execute(
                select(Fact).where(Fact.id == target_id, Fact.profile_id == profile_filter)
            )
        ).scalar_one_or_none()
        if fact_row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="fact not found")
        return {"item": _fact_to_detail(fact_row), "sources": await _source_details(session, target_type="fact", target_id=fact_row.id)}

    if target_type == "decision":
        decision_row = (
            await session.execute(
                select(Decision).where(Decision.id == target_id, Decision.profile_id == profile_filter)
            )
        ).scalar_one_or_none()
        if decision_row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="decision not found")
        return {"item": _decision_to_detail(decision_row), "sources": await _source_details(session, target_type="decision", target_id=decision_row.id)}

    if target_type == "message":
        message_row = (
            await session.execute(
                select(Message).where(Message.id == target_id, Message.profile_id == profile_filter)
            )
        ).scalar_one_or_none()
        if message_row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="message not found")
        return {
            "item": {
                "id": message_row.id,
                "target_type": "message",
                "session_id": message_row.session_id,
                "platform": message_row.platform,
                "role": message_row.role,
                "content_redacted": message_row.content_redacted,
                "project": message_row.project,
                "topic": message_row.topic,
                "created_at": message_row.created_at,
                "metadata": message_row.extra_metadata,
            },
            "sources": [{"event_id": message_row.event_id, "message_id": message_row.id}],
        }

    if target_type == "summary":
        session_summary = (
            await session.execute(
                select(SessionSummary).where(
                    SessionSummary.id == target_id,
                    SessionSummary.profile_id == profile_filter,
                )
            )
        ).scalar_one_or_none()
        if session_summary is not None:
            return {
                "item": {
                    "id": session_summary.id,
                    "target_type": "summary",
                    "scope": "session",
                    "session_id": session_summary.session_id,
                    "summary_short": session_summary.summary_short,
                    "summary_long": session_summary.summary_long,
                    "facts": session_summary.facts,
                    "decisions": session_summary.decisions,
                    "open_questions": session_summary.open_questions,
                    "status": session_summary.status,
                    "version": session_summary.version,
                    "created_at": session_summary.created_at,
                    "metadata": session_summary.extra_metadata,
                },
                "sources": [
                    {"event_id": event_id, "message_id": None}
                    for event_id in session_summary.source_event_ids
                ],
            }
        topic_summary = (
            await session.execute(
                select(TopicSummary).where(
                    TopicSummary.id == target_id,
                    TopicSummary.profile_id == profile_filter,
                )
            )
        ).scalar_one_or_none()
        if topic_summary is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="summary not found")
        return {
            "item": {
                "id": topic_summary.id,
                "target_type": "summary",
                "scope": "topic",
                "project": topic_summary.project,
                "topic": topic_summary.topic,
                "summary_short": topic_summary.summary_short,
                "summary_long": topic_summary.summary_long,
                "facts": topic_summary.facts,
                "decisions": topic_summary.decisions,
                "status": topic_summary.status,
                "version": topic_summary.version,
                "created_at": topic_summary.created_at,
                "metadata": topic_summary.extra_metadata,
            },
            "sources": [
                {"event_id": event_id, "message_id": None}
                for event_id in topic_summary.source_event_ids
            ],
        }

    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="unsupported target_type")


@router.post("/secrets/reveal")
async def reveal_secret(
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
    payload: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    """Explicitly reveal a secret memory value and write an audit event."""
    profile_filter = str(payload.get("profile_id") or profile_id)
    target_type = str(payload.get("target_type") or "")
    target_id = payload.get("target_id")
    reason = str(payload.get("reason") or "admin reveal")
    if not target_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="target_id required")

    statement = ""
    metadata: dict[str, Any] = {}
    if target_type == "fact":
        fact_row = (
            await session.execute(
                select(Fact).where(Fact.id == target_id, Fact.profile_id == profile_filter)
            )
        ).scalar_one_or_none()
        if fact_row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="fact not found")
        statement = fact_row.statement
        metadata = fact_row.extra_metadata or {}
    elif target_type == "decision":
        decision_row = (
            await session.execute(
                select(Decision).where(Decision.id == target_id, Decision.profile_id == profile_filter)
            )
        ).scalar_one_or_none()
        if decision_row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="decision not found")
        statement = decision_row.decision
        metadata = decision_row.extra_metadata or {}
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="unsupported target_type")

    if not is_secret_metadata(metadata):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="target is not marked secret")
    value = secret_value_for_reveal(statement, metadata)
    if value is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="secret value unavailable")

    await IngestService(profile_id=profile_filter).ingest_event(
        session,
        GenericEventIn(
            profile_id=profile_filter,
            event_type="secret_revealed",
            source="admin_api",
            actor="admin",
            agent_context="primary",
            payload={
                "event_type": "secret_revealed",
                "target_type": target_type,
                "target_id": str(target_id),
                "reason": reason,
                "secret_ref": secret_ref(metadata),
                "masked_preview": masked_preview(statement, metadata),
            },
        ),
    )
    await session.commit()
    return {
        "target_type": target_type,
        "target_id": str(target_id),
        "secret_value": value,
        "masked_preview": masked_preview(statement, metadata),
    }


@router.post("/export")
async def export_markdown(
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
    payload: dict[str, Any] = Body(default_factory=dict),
) -> Response:
    """Export project/topic facts and decisions as Markdown."""
    profile_filter = str(payload.get("profile_id") or profile_id)
    project = payload.get("project")
    topic = payload.get("topic")
    fact_stmt = select(Fact).where(Fact.profile_id == profile_filter)
    decision_stmt = select(Decision).where(Decision.profile_id == profile_filter)
    if project:
        fact_stmt = fact_stmt.where(Fact.project == str(project))
        decision_stmt = decision_stmt.where(Decision.project == str(project))
    if topic:
        fact_stmt = fact_stmt.where(Fact.topic == str(topic))
        decision_stmt = decision_stmt.where(Decision.topic == str(topic))
    facts = (await session.execute(fact_stmt.order_by(Fact.created_at.desc()))).scalars().all()
    decisions = (
        await session.execute(decision_stmt.order_by(Decision.decided_at.desc()))
    ).scalars().all()

    raw_title = str(project or "All Projects")
    title = " ".join(word.capitalize() for word in raw_title.replace("-", " ").split())
    if raw_title == "siqueira-memo":
        title = "Siqueira Memo"
    lines = [f"# {title} Memory Export", "", f"Profile: `{profile_filter}`", ""]
    if topic:
        lines.extend([f"Topic: `{topic}`", ""])
    lines.extend(["## Decisions", ""])
    if decisions:
        for decision_row in decisions:
            lines.extend(
                [
                    f"### {decision_row.topic}",
                    "",
                    masked_preview(decision_row.decision, decision_row.extra_metadata)
                    if is_secret_metadata(decision_row.extra_metadata)
                    else decision_row.decision,
                    "",
                    f"- Status: `{decision_row.status}`",
                    f"- Project: `{decision_row.project or ''}`",
                    f"- Decided at: `{decision_row.decided_at.isoformat()}`",
                    f"- ID: `{decision_row.id}`",
                    "",
                ]
            )
    else:
        lines.extend(["No decisions found.", ""])
    lines.extend(["## Facts", ""])
    if facts:
        for fact_row in facts:
            lines.extend(
                [
                    f"### {fact_row.subject} — {fact_row.predicate}",
                    "",
                    masked_preview(fact_row.statement, fact_row.extra_metadata)
                    if is_secret_metadata(fact_row.extra_metadata)
                    else fact_row.statement,
                    "",
                    f"- Status: `{fact_row.status}`",
                    f"- Project: `{fact_row.project or ''}`",
                    f"- Topic: `{fact_row.topic or ''}`",
                    f"- Confidence: `{fact_row.confidence}`",
                    f"- ID: `{fact_row.id}`",
                    "",
                ]
            )
    else:
        lines.extend(["No facts found.", ""])
    markdown = "\n".join(lines)
    filename = f"siqueira-{str(project or 'memory').replace('/', '-')}.md"
    return Response(
        markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _to_conflict_item(row: MemoryConflict) -> ConflictItem:
    return ConflictItem(
        id=row.id,
        conflict_type=row.conflict_type,
        left_type=row.left_type,
        left_id=row.left_id,
        right_type=row.right_type,
        right_id=row.right_id,
        severity=row.severity,
        status=row.status,
        resolution_hint=row.resolution_hint,
        confidence=float(row.confidence or 0.0),
        created_at=row.created_at,
        resolved_at=row.resolved_at,
    )


@router.post("/conflicts/scan", response_model=ConflictScanResponse)
async def conflicts_scan(
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> ConflictScanResponse:
    svc = ConflictService(profile_id=profile_id)
    rows = await svc.scan(session)
    return ConflictScanResponse(
        detected=len(rows),
        conflicts=[_to_conflict_item(row) for row in rows],
    )


@router.post("/conflicts/list", response_model=ConflictScanResponse)
async def conflicts_list(
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> ConflictScanResponse:
    rows = (
        await session.execute(
            select(MemoryConflict)
            .where(MemoryConflict.profile_id == profile_id)
            .order_by(MemoryConflict.created_at.desc())
        )
    ).scalars().all()
    return ConflictScanResponse(
        detected=len(rows),
        conflicts=[_to_conflict_item(row) for row in rows],
    )


@router.post("/conflicts/resolve", response_model=ConflictResolveResponse)
async def conflicts_resolve(
    payload: ConflictResolveRequest,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> ConflictResolveResponse:
    svc = ConflictService(profile_id=profile_id)
    try:
        row = await svc.resolve_by_supersession(
            session,
            conflict_id=payload.conflict_id,
            kept_id=payload.kept_id,
            dropped_id=payload.dropped_id,
            actor=payload.actor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ConflictResolveResponse(
        id=row.id,
        status=row.status,
        resolution=row.resolution,
        resolved_at=row.resolved_at,
    )


@router.post("/audit", response_model=AuditResponse)
async def audit(
    payload: AuditRequest,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> AuditResponse:
    log = AuditLog(profile_id=payload.profile_id or profile_id)
    entries = await log.fetch_deletion_events(
        session, since=payload.since, limit=payload.limit
    )
    return AuditResponse(
        entries=[
            AuditEntrySchema(
                id=e.id,
                event_type=e.event_type,
                target_type=e.target_type,
                target_id=e.target_id,
                actor=e.actor,
                created_at=e.created_at,
                reason=e.reason,
                mode=e.mode,
            )
            for e in entries
        ]
    )
