"""Remember / correct lifecycle. Plan §18.2.6 / §31.7 / §9.3."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from siqueira_memo.models import Fact, MemoryEvent
from siqueira_memo.models.constants import STATUS_ACTIVE, STATUS_SUPERSEDED
from siqueira_memo.schemas.memory import CorrectRequest, RememberRequest
from siqueira_memo.services.extraction_service import ExtractionService


@pytest.mark.asyncio
async def test_remember_fact_is_idempotent(db, session):
    svc = ExtractionService(profile_id="p1")
    req = RememberRequest(
        kind="fact",
        subject="siqueira-memo",
        predicate="primary_integration",
        object="MemoryProvider plugin",
        statement="Siqueira Memo integrates via MemoryProvider plugin.",
        project="siqueira-memo",
        confidence=0.9,
    )
    first = await svc.remember(session, req)
    second = await svc.remember(session, req)
    assert first.id == second.id
    facts = (await session.execute(select(Fact))).scalars().all()
    active = [f for f in facts if f.status == STATUS_ACTIVE]
    assert len(active) == 1


@pytest.mark.asyncio
async def test_remember_decision_creates_event(db, session):
    svc = ExtractionService(profile_id="p1")
    req = RememberRequest(
        kind="decision",
        statement="Use Hermes MemoryProvider plugin as primary integration",
        topic="memory integration",
        project="siqueira-memo",
    )
    result = await svc.remember(session, req)
    ev = (
        await session.execute(select(MemoryEvent).where(MemoryEvent.id == result.event_id))
    ).scalar_one()
    assert ev.event_type == "decision_recorded"
    assert ev.payload["decision_id"] == str(result.id)


@pytest.mark.asyncio
async def test_correction_supersedes_existing_fact(db, session):
    svc = ExtractionService(profile_id="p1")
    original = await svc.remember(
        session,
        RememberRequest(
            kind="fact",
            subject="shannon",
            predicate="primary_auth",
            object="api_key",
            statement="Shannon primary auth is API key.",
        ),
    )

    replacement = RememberRequest(
        kind="fact",
        subject="shannon",
        predicate="primary_auth",
        object="claude_oauth",
        statement="Shannon primary auth is Claude OAuth token.",
    )
    result = await svc.apply_correction(
        session,
        CorrectRequest(
            target_type="fact",
            target_id=original.id,
            correction_text="actually it's the OAuth token",
            replacement=replacement,
        ),
    )

    row = (await session.execute(select(Fact).where(Fact.id == original.id))).scalar_one()
    assert row.status == STATUS_SUPERSEDED
    assert result.replacement_id is not None
    assert row.superseded_by == result.replacement_id

    invalidated_events = (
        await session.execute(
            select(MemoryEvent).where(MemoryEvent.event_type == "fact_invalidated")
        )
    ).scalars().all()
    assert invalidated_events
