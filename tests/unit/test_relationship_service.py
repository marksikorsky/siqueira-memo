"""Relationship graph tests. Roadmap Phase 4."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from siqueira_memo.models import Decision, Fact, MemoryRelationship
from siqueira_memo.models.constants import STATUS_ACTIVE, STATUS_SUPERSEDED
from siqueira_memo.schemas.memory import CorrectRequest, RememberRequest
from siqueira_memo.schemas.recall import RecallRequest
from siqueira_memo.services.conflict_service import ConflictService
from siqueira_memo.services.extraction_service import ExtractionService
from siqueira_memo.services.relationship_service import RelationshipService
from siqueira_memo.services.retrieval_service import RetrievalService


@pytest.mark.asyncio
async def test_relationship_service_creates_and_dedupes_active_relationship(session):
    profile = "p-rel"
    source = Fact(
        profile_id=profile,
        subject="Siqueira",
        predicate="has",
        object="relationship graph",
        statement="Siqueira has a relationship graph.",
        canonical_key="fact-source-relationship-graph",
        status=STATUS_ACTIVE,
        confidence=0.9,
        extractor_name="test",
    )
    target = Decision(
        profile_id=profile,
        project="siqueira-memo",
        topic="memory graph",
        decision="Use memory_relationships for cross-memory links.",
        context="test",
        rationale="Relationships should be first-class.",
        canonical_key="decision-target-memory-graph",
        status=STATUS_ACTIVE,
        decided_at=datetime.now(UTC),
        extractor_name="test",
    )
    session.add_all([source, target])
    await session.flush()

    svc = RelationshipService(profile_id=profile, actor="test")
    first = await svc.create(
        session,
        source_type="fact",
        source_id=source.id,
        relationship_type="related_to",
        target_type="decision",
        target_id=target.id,
        confidence=0.71,
        rationale="seed link",
    )
    second = await svc.create(
        session,
        source_type="fact",
        source_id=source.id,
        relationship_type="related_to",
        target_type="decision",
        target_id=target.id,
        confidence=0.92,
        rationale="stronger evidence",
    )

    assert second.id == first.id
    assert second.confidence == pytest.approx(0.92)
    assert second.rationale == "stronger evidence"
    rows = (await session.execute(select(MemoryRelationship))).scalars().all()
    assert len(rows) == 1
    assert rows[0].created_by == "test"


@pytest.mark.asyncio
async def test_conflict_scan_creates_contradicts_relationship(session):
    profile = "p-conflict-rel"
    older = Decision(
        profile_id=profile,
        project="siqueira-memo",
        topic="mcp integration",
        decision="Use MCP as primary integration.",
        context="old",
        rationale="early assumption",
        canonical_key="old-mcp-primary",
        status=STATUS_ACTIVE,
        decided_at=datetime.now(UTC) - timedelta(days=4),
        extractor_name="test",
    )
    newer = Decision(
        profile_id=profile,
        project="siqueira-memo",
        topic="mcp integration",
        decision="Do not use MCP as primary integration.",
        context="new",
        rationale="MemoryProvider is native.",
        canonical_key="new-mcp-primary",
        status=STATUS_ACTIVE,
        decided_at=datetime.now(UTC),
        extractor_name="test",
    )
    session.add_all([older, newer])
    await session.flush()

    conflicts = await ConflictService(profile_id=profile).scan(session)
    assert conflicts

    rels = (
        await session.execute(
            select(MemoryRelationship).where(
                MemoryRelationship.profile_id == profile,
                MemoryRelationship.relationship_type == "contradicts",
            )
        )
    ).scalars().all()
    assert len(rels) == 1
    assert rels[0].source_type == "decision"
    assert rels[0].target_type == "decision"
    assert {rels[0].source_id, rels[0].target_id} == {older.id, newer.id}


@pytest.mark.asyncio
async def test_correction_replacement_creates_supersedes_relationship(session):
    profile = "p-correction-rel"
    svc = ExtractionService(profile_id=profile, actor="test")
    old = await svc.remember(
        session,
        RememberRequest(
            kind="decision",
            project="siqueira-memo",
            topic="secret policy",
            statement="Never store secrets in Siqueira Memo.",
            rationale="old safety rule",
        ),
    )
    corrected = await svc.apply_correction(
        session,
        CorrectRequest(
            target_type="decision",
            target_id=old.id,
            correction_text="Secrets may be stored when masked and audited.",
            replacement=RememberRequest(
                kind="decision",
                project="siqueira-memo",
                topic="secret policy",
                statement="Store useful secrets as masked, audited secret memories.",
                rationale="Milestone C policy",
            ),
        ),
    )
    assert corrected.replacement_id is not None

    old_row = await session.get(Decision, old.id)
    assert old_row is not None
    assert old_row.status == STATUS_SUPERSEDED
    assert old_row.superseded_by == corrected.replacement_id

    rel = (
        await session.execute(
            select(MemoryRelationship).where(
                MemoryRelationship.profile_id == profile,
                MemoryRelationship.relationship_type == "supersedes",
                MemoryRelationship.source_id == corrected.replacement_id,
                MemoryRelationship.target_id == old.id,
            )
        )
    ).scalar_one_or_none()
    assert rel is not None
    assert rel.source_type == "decision"
    assert rel.target_type == "decision"


@pytest.mark.asyncio
async def test_recall_expands_from_seed_memory_to_related_active_decision(session):
    profile = "p-recall-rel"
    seed = Fact(
        profile_id=profile,
        subject="legacy secret skip policy",
        predicate="status",
        object="superseded",
        statement="The legacy secret skip policy said to skip all secrets.",
        canonical_key="legacy-secret-skip-policy",
        status=STATUS_SUPERSEDED,
        confidence=0.8,
        topic="secret policy",
        extractor_name="test",
    )
    active = Decision(
        profile_id=profile,
        project="siqueira-memo",
        topic="secret policy",
        decision="Store useful secrets as masked, audited secret memories.",
        context="Milestone C",
        rationale="Useful operational memory should not be dropped.",
        canonical_key="active-secret-policy",
        status=STATUS_ACTIVE,
        decided_at=datetime.now(UTC),
        extractor_name="test",
    )
    session.add_all([seed, active])
    await session.flush()
    await RelationshipService(profile_id=profile, actor="test").create(
        session,
        source_type="decision",
        source_id=active.id,
        relationship_type="supersedes",
        target_type="fact",
        target_id=seed.id,
        confidence=0.95,
        rationale="active policy supersedes legacy skip rule",
    )
    await session.flush()

    result = await RetrievalService(profile_id=profile).recall(
        session,
        RecallRequest(
            query="legacy secret skip policy",
            types=["facts", "decisions"],
            mode="balanced",
        ),
    )
    recalled = {d.id: d for d in result.context_pack.decisions}
    assert active.id in recalled
    assert recalled[active.id].retrieval_lane == "graph"
    assert "supersedes" in (recalled[active.id].retrieval_explanation or "")
    assert "active decision" in result.context_pack.answer_context.lower()
