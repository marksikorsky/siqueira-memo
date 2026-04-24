"""Remember/correct/forget/timeline/sources endpoints. Plan §4.4."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from siqueira_memo.api.deps import AuthDep, ProfileDep, SessionDep
from siqueira_memo.models import (
    Decision,
    DecisionSource,
    Fact,
    FactSource,
    SessionSummary,
    TopicSummary,
)
from siqueira_memo.schemas.common import SourceRef
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
from siqueira_memo.services.deletion_service import DeletionService
from siqueira_memo.services.extraction_service import ExtractionService

router = APIRouter(prefix="/v1/memory")


@router.post("/remember", response_model=RememberResponse)
async def remember(
    payload: RememberRequest,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> RememberResponse:
    svc = ExtractionService(profile_id=profile_id)
    try:
        return await svc.remember(session, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/correct", response_model=CorrectResponse)
async def correct(
    payload: CorrectRequest,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> CorrectResponse:
    svc = ExtractionService(profile_id=profile_id)
    return await svc.apply_correction(session, payload)


@router.post("/forget", response_model=ForgetResponse)
async def forget(
    payload: ForgetRequest,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> ForgetResponse:
    svc = DeletionService(profile_id=profile_id)
    return await svc.forget(session, payload)


@router.post("/timeline", response_model=TimelineResponse)
async def timeline(
    payload: TimelineRequest,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> TimelineResponse:
    entries: list[TimelineEntry] = []
    profile_filter = payload.profile_id or profile_id

    decision_stmt = select(Decision).where(Decision.profile_id == profile_filter)
    if payload.project:
        decision_stmt = decision_stmt.where(Decision.project == payload.project)
    if payload.topic:
        decision_stmt = decision_stmt.where(Decision.topic == payload.topic)
    if payload.since:
        decision_stmt = decision_stmt.where(Decision.decided_at >= payload.since)
    if payload.until:
        decision_stmt = decision_stmt.where(Decision.decided_at <= payload.until)
    decision_stmt = decision_stmt.order_by(Decision.decided_at.desc()).limit(payload.limit)
    for d in (await session.execute(decision_stmt)).scalars():
        entries.append(
            TimelineEntry(
                id=d.id,
                kind="decision",
                title=d.topic,
                preview=d.decision[:200],
                status=d.status,
                created_at=d.decided_at,
            )
        )

    fact_stmt = select(Fact).where(Fact.profile_id == profile_filter)
    if payload.project:
        fact_stmt = fact_stmt.where(Fact.project == payload.project)
    if payload.topic:
        fact_stmt = fact_stmt.where(Fact.topic == payload.topic)
    if payload.since:
        fact_stmt = fact_stmt.where(Fact.created_at >= payload.since)
    if payload.until:
        fact_stmt = fact_stmt.where(Fact.created_at <= payload.until)
    fact_stmt = fact_stmt.order_by(Fact.created_at.desc()).limit(payload.limit)
    for f in (await session.execute(fact_stmt)).scalars():
        entries.append(
            TimelineEntry(
                id=f.id,
                kind="fact",
                title=f.predicate,
                preview=f.statement[:200],
                status=f.status,
                created_at=f.created_at,
            )
        )

    entries.sort(key=lambda e: e.created_at, reverse=True)
    return TimelineResponse(entries=entries[: payload.limit])


@router.post("/sources", response_model=SourcesResponse)
async def sources(
    payload: SourcesRequest,
    session: SessionDep,
    profile_id: ProfileDep,
    _token: AuthDep,
) -> SourcesResponse:
    profile_filter = payload.profile_id or profile_id
    refs: list[SourceRef] = []
    if payload.target_type == "fact":
        fact_rows = (
            (
                await session.execute(
                    select(FactSource).where(FactSource.fact_id == payload.target_id)
                )
            )
            .scalars()
            .all()
        )
        for fact_src in fact_rows:
            refs.append(
                SourceRef(
                    event_id=str(fact_src.event_id),
                    message_id=str(fact_src.message_id) if fact_src.message_id else None,
                )
            )
        if not refs:
            fact_row = (
                await session.execute(
                    select(Fact).where(
                        Fact.id == payload.target_id, Fact.profile_id == profile_filter
                    )
                )
            ).scalar_one_or_none()
            if fact_row is not None:
                refs.extend(
                    SourceRef(event_id=str(eid)) for eid in (fact_row.source_event_ids or [])
                )
    elif payload.target_type == "decision":
        decision_rows = (
            (
                await session.execute(
                    select(DecisionSource).where(
                        DecisionSource.decision_id == payload.target_id
                    )
                )
            )
            .scalars()
            .all()
        )
        for decision_src in decision_rows:
            refs.append(
                SourceRef(
                    event_id=str(decision_src.event_id),
                    message_id=str(decision_src.message_id) if decision_src.message_id else None,
                )
            )
        if not refs:
            decision_row = (
                await session.execute(
                    select(Decision).where(
                        Decision.id == payload.target_id,
                        Decision.profile_id == profile_filter,
                    )
                )
            ).scalar_one_or_none()
            if decision_row is not None:
                refs.extend(
                    SourceRef(event_id=str(eid))
                    for eid in (decision_row.source_event_ids or [])
                )
    elif payload.target_type == "summary":
        for summary_model in (SessionSummary, TopicSummary):
            summary_row: Any = (
                await session.execute(
                    select(summary_model).where(
                        summary_model.id == payload.target_id,
                        summary_model.profile_id == profile_filter,
                    )
                )
            ).scalar_one_or_none()
            if summary_row is not None:
                refs.extend(
                    SourceRef(event_id=str(eid))
                    for eid in (summary_row.source_event_ids or [])
                )
                break
    return SourcesResponse(sources=refs)
