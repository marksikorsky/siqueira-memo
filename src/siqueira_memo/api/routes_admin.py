"""Admin search + introspection endpoints. Plan §9.2 / §21 / §25."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Response, status
from sqlalchemy import func, select

from siqueira_memo.api.deps import AuthDep, ProfileDep, SessionDep
from siqueira_memo.models import (
    Decision,
    DecisionSource,
    Fact,
    FactSource,
    MemoryConflict,
    MemoryEvent,
    Message,
    SessionSummary,
    TopicSummary,
)
from siqueira_memo.schemas.admin import AdminSearchHit, AdminSearchRequest, AdminSearchResponse
from siqueira_memo.schemas.audit import AuditEntrySchema, AuditRequest, AuditResponse
from siqueira_memo.schemas.conflicts import (
    ConflictItem,
    ConflictResolveRequest,
    ConflictResolveResponse,
    ConflictScanResponse,
)
from siqueira_memo.services.conflict_service import ConflictService
from siqueira_memo.services.retention_service import AuditLog

router = APIRouter(prefix="/v1/admin")


_PREVIEW = 200


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
        if payload.project:
            m_stmt = m_stmt.where(Message.project == payload.project)
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
        if payload.project:
            f_stmt = f_stmt.where(Fact.project == payload.project)
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
                    preview=(f.statement or "")[:_PREVIEW],
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
        if payload.project:
            d_stmt = d_stmt.where(Decision.project == payload.project)
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
                    preview=(d.decision or "")[:_PREVIEW],
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


def _fact_to_detail(row: Fact) -> dict[str, Any]:
    return {
        "id": row.id,
        "target_type": "fact",
        "subject": row.subject,
        "predicate": row.predicate,
        "object": row.object,
        "statement": row.statement,
        "project": row.project,
        "topic": row.topic,
        "status": row.status,
        "confidence": row.confidence,
        "valid_from": row.valid_from,
        "valid_to": row.valid_to,
        "source_event_ids": row.source_event_ids,
        "source_message_ids": row.source_message_ids,
        "superseded_by": row.superseded_by,
        "metadata": row.extra_metadata,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def _decision_to_detail(row: Decision) -> dict[str, Any]:
    return {
        "id": row.id,
        "target_type": "decision",
        "topic": row.topic,
        "decision": row.decision,
        "statement": row.decision,
        "context": row.context,
        "rationale": row.rationale,
        "tradeoffs": row.tradeoffs,
        "options_considered": row.options_considered,
        "project": row.project,
        "status": row.status,
        "reversible": row.reversible,
        "source_event_ids": row.source_event_ids,
        "source_message_ids": row.source_message_ids,
        "superseded_by": row.superseded_by,
        "metadata": row.extra_metadata,
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
                    decision_row.decision,
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
                    fact_row.statement,
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
