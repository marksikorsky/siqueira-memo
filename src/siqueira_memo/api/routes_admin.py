"""Admin search + introspection endpoints. Plan §9.2 / §21 / §25."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select

from siqueira_memo.api.deps import AuthDep, ProfileDep, SessionDep
from siqueira_memo.models import Decision, Fact, MemoryConflict, Message, SessionSummary
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
