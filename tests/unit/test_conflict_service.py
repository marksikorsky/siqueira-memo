"""Conflict detection and resolution. Plan §21."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from siqueira_memo.models import Decision, Fact, MemoryConflict, MemoryEvent
from siqueira_memo.models.constants import (
    CONFLICT_STATUS_AUTO_RESOLVED,
    CONFLICT_STATUS_OPEN,
    STATUS_ACTIVE,
    STATUS_SUPERSEDED,
)
from siqueira_memo.services.conflict_service import ConflictService


async def _mk_event(session, profile_id: str, event_type: str = "decision_recorded") -> MemoryEvent:
    event = MemoryEvent(
        event_type=event_type,
        source="test",
        actor="test",
        profile_id=profile_id,
        payload={"event_type": event_type},
    )
    session.add(event)
    await session.flush()
    return event


async def _mk_decision(session, profile_id: str, *, topic: str, decision: str, when: datetime | None = None) -> Decision:
    event = await _mk_event(session, profile_id)
    row = Decision(
        profile_id=profile_id,
        topic=topic,
        decision=decision,
        context="ctx",
        rationale="r",
        canonical_key=f"dec-{uuid.uuid4()}",
        status=STATUS_ACTIVE,
        decided_at=when or datetime.now(UTC),
        source_event_ids=[event.id],
    )
    session.add(row)
    await session.flush()
    return row


async def _mk_fact(
    session,
    profile_id: str,
    *,
    subject: str,
    predicate: str,
    obj: str,
    when: datetime | None = None,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    confidence: float = 0.8,
) -> Fact:
    event = await _mk_event(session, profile_id, event_type="fact_extracted")
    row = Fact(
        profile_id=profile_id,
        subject=subject,
        predicate=predicate,
        object=obj,
        statement=f"{subject} {predicate} {obj}",
        canonical_key=f"f-{uuid.uuid4()}",
        status=STATUS_ACTIVE,
        confidence=confidence,
        source_event_ids=[event.id],
        valid_from=valid_from,
        valid_to=valid_to,
    )
    session.add(row)
    await session.flush()
    return row


@pytest.mark.asyncio
async def test_decision_polarity_conflict_detected(db, session):
    profile = "p1"
    svc = ConflictService(profile_id=profile)
    older = await _mk_decision(
        session, profile,
        topic="mcp integration",
        decision="Use MCP as primary integration",
        when=datetime.now(UTC) - timedelta(days=10),
    )
    newer = await _mk_decision(
        session, profile,
        topic="mcp integration",
        decision="Do not use MCP as primary integration",
        when=datetime.now(UTC),
    )
    conflicts = await svc.scan(session)
    assert conflicts
    stored = (await session.execute(select(MemoryConflict))).scalars().all()
    assert stored
    conflict = stored[0]
    assert conflict.conflict_type == "decision_decision"
    # newer is the "right" side (stored more recent)
    ids = {conflict.left_id, conflict.right_id}
    assert older.id in ids and newer.id in ids


@pytest.mark.asyncio
async def test_fact_fact_conflict_on_same_subject_predicate(db, session):
    profile = "p1"
    svc = ConflictService(profile_id=profile)
    await _mk_fact(
        session, profile,
        subject="shannon", predicate="primary_auth", obj="api_key",
        valid_from=datetime.now(UTC) - timedelta(days=30),
    )
    await _mk_fact(
        session, profile,
        subject="shannon", predicate="primary_auth", obj="claude_oauth_token",
        valid_from=datetime.now(UTC) - timedelta(days=1),
    )
    await svc.scan(session)
    stored = (await session.execute(select(MemoryConflict))).scalars().all()
    assert any(c.conflict_type == "fact_fact" for c in stored)


@pytest.mark.asyncio
async def test_temporal_non_overlap_not_conflict(db, session):
    profile = "p1"
    svc = ConflictService(profile_id=profile)
    await _mk_fact(
        session, profile,
        subject="server", predicate="ip", obj="192.168.0.1",
        valid_from=datetime.now(UTC) - timedelta(days=60),
        valid_to=datetime.now(UTC) - timedelta(days=30),
    )
    await _mk_fact(
        session, profile,
        subject="server", predicate="ip", obj="192.168.0.2",
        valid_from=datetime.now(UTC) - timedelta(days=20),
        valid_to=datetime.now(UTC),
    )
    await svc.scan(session)
    stored = (await session.execute(select(MemoryConflict))).scalars().all()
    assert not any(c.conflict_type == "fact_fact" for c in stored)


@pytest.mark.asyncio
async def test_resolve_by_supersession(db, session):
    profile = "p1"
    svc = ConflictService(profile_id=profile)
    older = await _mk_decision(
        session, profile,
        topic="mcp", decision="Use MCP as primary",
        when=datetime.now(UTC) - timedelta(days=5),
    )
    newer = await _mk_decision(
        session, profile,
        topic="mcp", decision="Do not use MCP as primary",
        when=datetime.now(UTC),
    )
    await svc.scan(session)

    stored = (await session.execute(select(MemoryConflict).where(MemoryConflict.status == CONFLICT_STATUS_OPEN))).scalars().one()
    resolved = await svc.resolve_by_supersession(
        session, conflict_id=stored.id, kept_id=newer.id, dropped_id=older.id
    )
    assert resolved.status == CONFLICT_STATUS_AUTO_RESOLVED
    refreshed_older = (await session.execute(select(Decision).where(Decision.id == older.id))).scalar_one()
    assert refreshed_older.status == STATUS_SUPERSEDED
    assert refreshed_older.superseded_by == newer.id


@pytest.mark.asyncio
async def test_candidate_narrowing_caps_pairs(db, session):
    profile = "p1"
    svc = ConflictService(profile_id=profile, max_pairs=3)
    # Seed ten unrelated decisions + two conflicting ones; verify at most max_pairs open conflicts.
    for i in range(10):
        await _mk_decision(session, profile, topic=f"topic-{i}", decision=f"pick option {i}")
    await _mk_decision(session, profile, topic="shared", decision="Use X primary")
    await _mk_decision(session, profile, topic="shared", decision="Do not use X primary")
    await svc.scan(session)
    stored = (await session.execute(select(MemoryConflict))).scalars().all()
    assert len(stored) <= 3


@pytest.mark.asyncio
async def test_correction_supersession_marks_resolved(db, session):
    profile = "p1"
    svc = ConflictService(profile_id=profile)
    older = await _mk_fact(session, profile, subject="s", predicate="p", obj="old")
    newer = await _mk_fact(session, profile, subject="s", predicate="p", obj="new")
    await svc.scan(session)
    conflict = (await session.execute(select(MemoryConflict))).scalars().one()
    result = await svc.record_user_correction(
        session, conflict_id=conflict.id, kept_id=newer.id, dropped_id=older.id
    )
    assert result.status == CONFLICT_STATUS_AUTO_RESOLVED
    assert result.resolved_by == "user_correction"
