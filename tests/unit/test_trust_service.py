"""Trust/source reputation tests. Roadmap Phase 8."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from siqueira_memo.models import Decision, Fact, MemoryConflict, MemoryEvent
from siqueira_memo.models.constants import CONFLICT_STATUS_OPEN, STATUS_ACTIVE, STATUS_SUPERSEDED
from siqueira_memo.schemas.recall import RecallRequest
from siqueira_memo.services.retrieval_service import RetrievalService
from siqueira_memo.services.trust_service import TrustService


@pytest.mark.asyncio
async def test_source_backed_user_confirmed_memory_scores_higher_than_inferred_import(session):
    profile = "p-trust-score"
    source_backed = Fact(
        profile_id=profile,
        subject="Siqueira",
        predicate="trust",
        object="source-backed",
        statement="Siqueira trust scoring favors source-backed user-confirmed facts.",
        canonical_key="source-backed-trust-fact",
        status=STATUS_ACTIVE,
        confidence=0.8,
        source_event_ids=[uuid.uuid4()],
        extractor_name="memory_capture_classifier",
        extra_metadata={"capture_kind": "preference", "confirmed_by": "user"},
        created_at=datetime.now(UTC),
    )
    inferred_import = Fact(
        profile_id=profile,
        subject="Siqueira",
        predicate="trust",
        object="inferred-import",
        statement="Siqueira inferred this trust fact from an imported summary without sources.",
        canonical_key="inferred-import-trust-fact",
        status=STATUS_ACTIVE,
        confidence=0.8,
        extractor_name="summary_import",
        extra_metadata={"source_type": "summary", "inferred": True},
        created_at=datetime.now(UTC),
    )
    session.add_all([source_backed, inferred_import])
    await session.flush()

    svc = TrustService(profile_id=profile)
    trusted = await svc.score_memory(session, "fact", source_backed)
    weak = await svc.score_memory(session, "fact", inferred_import)

    assert trusted.trust_score > weak.trust_score
    assert trusted.factors["source_backed"] > 0.0
    assert trusted.factors["user_confirmed"] > 0.0
    assert weak.factors["source_backed"] == 0.0
    assert weak.factors["summary_or_import_penalty"] < 0.0


@pytest.mark.asyncio
async def test_superseded_and_conflicting_memory_trust_drops(session):
    profile = "p-trust-penalty"
    active = Decision(
        profile_id=profile,
        project="siqueira-memo",
        topic="trust",
        decision="Use trust scoring as a retrieval boost.",
        context="Phase 8",
        rationale="Trusted memories should rank higher.",
        canonical_key="active-trust-decision",
        status=STATUS_ACTIVE,
        decided_at=datetime.now(UTC),
        source_event_ids=[uuid.uuid4()],
        extractor_name="test",
        extra_metadata={"confidence": 0.9, "confirmed_by": "user"},
    )
    superseded = Decision(
        profile_id=profile,
        project="siqueira-memo",
        topic="trust",
        decision="Ignore trust scoring.",
        context="old",
        rationale="obsolete",
        canonical_key="old-trust-decision",
        status=STATUS_SUPERSEDED,
        decided_at=datetime.now(UTC) - timedelta(days=3),
        source_event_ids=[uuid.uuid4()],
        extractor_name="test",
        extra_metadata={"confidence": 0.9, "confirmed_by": "user"},
    )
    session.add_all([active, superseded])
    await session.flush()
    session.add(
        MemoryConflict(
            profile_id=profile,
            conflict_type="contradiction",
            left_type="decision",
            left_id=superseded.id,
            right_type="decision",
            right_id=active.id,
            status=CONFLICT_STATUS_OPEN,
            confidence=0.9,
        )
    )
    await session.flush()

    svc = TrustService(profile_id=profile)
    active_score = await svc.score_memory(session, "decision", active)
    superseded_score = await svc.score_memory(session, "decision", superseded)

    assert active_score.trust_score > superseded_score.trust_score
    assert superseded_score.factors["status_penalty"] < 0.0
    assert superseded_score.factors["open_conflict_penalty"] < 0.0


@pytest.mark.asyncio
async def test_feedback_updates_trust_and_writes_audit_event(session):
    profile = "p-trust-feedback"
    fact = Fact(
        profile_id=profile,
        subject="Trust feedback",
        predicate="supports",
        object="audit",
        statement="Trust feedback writes an audit event.",
        canonical_key="trust-feedback-audit",
        status=STATUS_ACTIVE,
        confidence=0.7,
        extractor_name="test",
    )
    session.add(fact)
    await session.flush()

    svc = TrustService(profile_id=profile, actor="admin-test")
    before = await svc.score_memory(session, "fact", fact)
    updated = await svc.record_feedback(
        session,
        target_type="fact",
        target_id=fact.id,
        feedback="useful",
        reason="admin verified during Phase 8 smoke",
    )
    await session.flush()
    after = await svc.score_memory(session, "fact", fact)

    assert after.trust_score > before.trust_score
    assert updated.extra_metadata["trust_feedback"][-1]["feedback"] == "useful"
    event = (
        await session.execute(
            select(MemoryEvent).where(
                MemoryEvent.profile_id == profile,
                MemoryEvent.event_type == "trust_feedback_recorded",
            )
        )
    ).scalar_one()
    assert event.actor == "admin-test"
    assert event.payload["target_id"] == str(fact.id)
    assert "Trust feedback writes an audit event" not in str(event.payload)


@pytest.mark.asyncio
async def test_retrieval_uses_trust_as_boost_not_absolute_gate(session):
    profile = "p-trust-retrieval"
    trusted = Fact(
        profile_id=profile,
        subject="support email",
        predicate="current",
        object="trusted@example.com",
        statement="The current support email is trusted@example.com.",
        canonical_key="trusted-support-email",
        status=STATUS_ACTIVE,
        confidence=0.75,
        source_event_ids=[uuid.uuid4()],
        extractor_name="test",
        extra_metadata={"confirmed_by": "user"},
        created_at=datetime.now(UTC) - timedelta(days=2),
    )
    weak = Fact(
        profile_id=profile,
        subject="support email",
        predicate="current",
        object="weak@example.com",
        statement="The current support email is weak@example.com.",
        canonical_key="weak-support-email",
        status=STATUS_ACTIVE,
        confidence=0.75,
        extractor_name="summary_import",
        extra_metadata={"source_type": "summary", "inferred": True},
        created_at=datetime.now(UTC),
    )
    session.add_all([weak, trusted])
    await session.flush()

    result = await RetrievalService(profile_id=profile).recall(
        session,
        RecallRequest(query="current support email", types=["facts"], limit=2),
    )

    facts = result.context_pack.facts
    assert [fact.id for fact in facts] == [trusted.id, weak.id]
    assert all("trust_score" in fact.score_breakdown for fact in facts)
    assert facts[0].trust_score > facts[1].trust_score
    assert len(facts) == 2
