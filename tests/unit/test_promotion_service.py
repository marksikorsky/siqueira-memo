"""Candidate-to-active promotion. Plan §31.7 / §33.12."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from siqueira_memo.models import Decision, Fact, MemoryConflict, MemoryEvent
from siqueira_memo.models.constants import (
    CONFLICT_STATUS_NEEDS_REVIEW,
    STATUS_ACTIVE,
    STATUS_CANDIDATE,
    STATUS_DEDUPED,
    STATUS_NEEDS_REVIEW,
)
from siqueira_memo.services.promotion_service import (
    PromotionOutcome,
    PromotionService,
)
from siqueira_memo.utils.canonical import (
    advisory_lock_key,
    decision_canonical_key,
    fact_canonical_key,
)


async def _mk_event(session, profile: str, et: str = "fact_extracted") -> uuid.UUID:
    event = MemoryEvent(
        id=uuid.uuid4(),
        event_type=et,
        source="extractor",
        actor="worker",
        profile_id=profile,
        payload={"event_type": et},
    )
    session.add(event)
    await session.flush()
    return event.id


async def _mk_candidate_fact(
    session, profile: str, *, subject: str, predicate: str, obj: str, confidence: float = 0.85
) -> Fact:
    event_id = await _mk_event(session, profile)
    key = fact_canonical_key(subject, predicate, obj, profile_id=profile)
    fact = Fact(
        profile_id=profile,
        subject=subject,
        predicate=predicate,
        object=obj,
        statement=f"{subject} {predicate} {obj}",
        canonical_key=key,
        status=STATUS_CANDIDATE,
        confidence=confidence,
        source_event_ids=[event_id],
    )
    session.add(fact)
    await session.flush()
    return fact


async def _mk_candidate_decision(
    session, profile: str, *, topic: str, decision: str, confidence: float = 0.85
) -> Decision:
    event_id = await _mk_event(session, profile, et="decision_recorded")
    key = decision_canonical_key(None, topic, decision, profile_id=profile)
    row = Decision(
        profile_id=profile,
        topic=topic,
        decision=decision,
        context="ctx",
        rationale="r",
        canonical_key=key,
        status=STATUS_CANDIDATE,
        decided_at=datetime.now(UTC),
        source_event_ids=[event_id],
        extra_metadata={"confidence": confidence},
    )
    session.add(row)
    await session.flush()
    return row


@pytest.mark.asyncio
async def test_promote_fact_with_no_active_collision(db, session):
    profile = "p1"
    candidate = await _mk_candidate_fact(
        session, profile, subject="siqueira-memo", predicate="primary_integration", obj="plugin"
    )
    svc = PromotionService(profile_id=profile, promotion_threshold=0.6)
    outcome = await svc.promote(session, candidate)
    assert outcome is PromotionOutcome.PROMOTED
    refreshed = (await session.execute(select(Fact).where(Fact.id == candidate.id))).scalar_one()
    assert refreshed.status == STATUS_ACTIVE


@pytest.mark.asyncio
async def test_low_confidence_candidate_stays_candidate(db, session):
    profile = "p1"
    candidate = await _mk_candidate_fact(
        session, profile, subject="s", predicate="p", obj="o", confidence=0.3
    )
    svc = PromotionService(profile_id=profile, promotion_threshold=0.6)
    outcome = await svc.promote(session, candidate)
    assert outcome is PromotionOutcome.BELOW_THRESHOLD
    refreshed = (await session.execute(select(Fact).where(Fact.id == candidate.id))).scalar_one()
    assert refreshed.status == STATUS_CANDIDATE


@pytest.mark.asyncio
async def test_dedupe_when_active_equivalent_exists(db, session):
    profile = "p1"
    active = await _mk_candidate_fact(
        session, profile, subject="s", predicate="p", obj="o"
    )
    active.status = STATUS_ACTIVE
    await session.flush()

    # Second candidate with identical canonical key and extra source.
    dup_event = await _mk_event(session, profile)
    duplicate = Fact(
        profile_id=profile,
        subject="s",
        predicate="p",
        object="o",
        statement="s p o",
        canonical_key=active.canonical_key,
        status=STATUS_CANDIDATE,
        confidence=0.99,
        source_event_ids=[dup_event],
    )
    session.add(duplicate)
    await session.flush()

    svc = PromotionService(profile_id=profile)
    outcome = await svc.promote(session, duplicate)
    assert outcome is PromotionOutcome.DEDUPED
    dup_refreshed = (
        await session.execute(select(Fact).where(Fact.id == duplicate.id))
    ).scalar_one()
    assert dup_refreshed.status == STATUS_DEDUPED
    active_refreshed = (
        await session.execute(select(Fact).where(Fact.id == active.id))
    ).scalar_one()
    # Source events are merged onto the kept active row.
    merged = {str(x) for x in (active_refreshed.source_event_ids or [])}
    assert str(dup_event) in merged


@pytest.mark.asyncio
async def test_conflicting_active_flags_needs_review(db, session):
    profile = "p1"
    # Seed an active fact with the same subject/predicate but different object.
    event_id = await _mk_event(session, profile)
    active_key = fact_canonical_key("s", "p", "old", profile_id=profile)
    active = Fact(
        profile_id=profile,
        subject="s",
        predicate="p",
        object="old",
        statement="s p old",
        canonical_key=active_key,
        status=STATUS_ACTIVE,
        confidence=0.9,
        source_event_ids=[event_id],
    )
    session.add(active)
    await session.flush()

    candidate = await _mk_candidate_fact(
        session, profile, subject="s", predicate="p", obj="new"
    )
    svc = PromotionService(profile_id=profile)
    outcome = await svc.promote(session, candidate)
    assert outcome is PromotionOutcome.NEEDS_REVIEW

    refreshed = (
        await session.execute(select(Fact).where(Fact.id == candidate.id))
    ).scalar_one()
    assert refreshed.status == STATUS_NEEDS_REVIEW

    conflicts = (await session.execute(select(MemoryConflict))).scalars().all()
    assert len(conflicts) == 1
    assert conflicts[0].status == CONFLICT_STATUS_NEEDS_REVIEW
    assert conflicts[0].conflict_type == "fact_fact"


@pytest.mark.asyncio
async def test_promote_decision_path(db, session):
    profile = "p1"
    candidate = await _mk_candidate_decision(
        session, profile, topic="integration", decision="Use Hermes MemoryProvider plugin"
    )
    svc = PromotionService(profile_id=profile, promotion_threshold=0.6)
    outcome = await svc.promote(session, candidate)
    assert outcome is PromotionOutcome.PROMOTED
    refreshed = (
        await session.execute(select(Decision).where(Decision.id == candidate.id))
    ).scalar_one()
    assert refreshed.status == STATUS_ACTIVE


def test_advisory_lock_key_is_deterministic_and_signed():
    key1 = advisory_lock_key("foo")
    key2 = advisory_lock_key("foo")
    assert key1 == key2
    assert -(2**63) <= key1 < 2**63
