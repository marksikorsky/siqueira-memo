"""Retrieval + context-pack tests. Plan §8 / §5.8 / §33.5."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from siqueira_memo.config import settings_for_tests
from siqueira_memo.models import Chunk, Decision, Fact, MemoryEvent
from siqueira_memo.models.constants import (
    CHUNK_SOURCE_MESSAGE,
    RECALL_MODE_DEEP,
    RECALL_MODE_FAST,
    STATUS_ACTIVE,
)
from siqueira_memo.schemas.recall import RecallRequest
from siqueira_memo.services.context_pack_service import ContextPackShaper
from siqueira_memo.services.embedding_service import MockEmbeddingProvider
from siqueira_memo.services.retrieval_service import RetrievalService


@pytest.mark.asyncio
async def test_recall_returns_decisions_and_facts(session):
    profile = "p1"
    event = MemoryEvent(
        event_type="decision_recorded",
        source="test",
        actor="test",
        profile_id=profile,
        payload={"event_type": "decision_recorded", "decision_id": "x", "canonical_key": "k", "status": "active"},
    )
    session.add(event)
    await session.flush()

    decision = Decision(
        profile_id=profile,
        topic="memory architecture",
        decision="Use Hermes MemoryProvider plugin as primary integration",
        context="conversation",
        rationale="decided vs MCP",
        canonical_key="dec-memory-architecture-primary",
        status=STATUS_ACTIVE,
        decided_at=datetime.now(UTC),
        source_event_ids=[event.id],
        extractor_name="manual",
    )
    fact = Fact(
        profile_id=profile,
        subject="siqueira-memo",
        predicate="primary_integration",
        object="MemoryProvider plugin",
        statement="Siqueira Memo integrates primarily through a Hermes MemoryProvider plugin.",
        canonical_key="fact-siqueira-memo-primary-integration",
        status=STATUS_ACTIVE,
        confidence=0.95,
        source_event_ids=[event.id],
        topic="memory architecture",
        extractor_name="manual",
    )
    session.add_all([decision, fact])
    await session.flush()

    svc = RetrievalService(profile_id=profile, embedding_provider=MockEmbeddingProvider())
    result = await svc.recall(
        session,
        RecallRequest(query="какой primary integration для памяти?", mode="balanced"),
    )

    ids = {str(d.id) for d in result.context_pack.decisions}
    assert str(decision.id) in ids
    assert any(f.status == STATUS_ACTIVE for f in result.context_pack.facts)
    assert result.context_pack.confidence in {"high", "medium"}
    assert result.context_pack.embedding_table == "chunk_embeddings_mock"


@pytest.mark.asyncio
async def test_recall_detects_conflict_between_active_decisions(session):
    profile = "p1"
    event = MemoryEvent(
        event_type="decision_recorded",
        source="test",
        actor="test",
        profile_id=profile,
        payload={"event_type": "decision_recorded", "decision_id": "x", "canonical_key": "k", "status": "active"},
    )
    session.add(event)
    await session.flush()
    older = Decision(
        profile_id=profile,
        topic="mcp integration",
        decision="Use MCP as primary integration",
        context="early exploration",
        rationale="assumed MCP was canonical",
        canonical_key="dec-mcp-use",
        status=STATUS_ACTIVE,
        decided_at=datetime.now(UTC) - timedelta(days=30),
        source_event_ids=[event.id],
        extractor_name="manual",
    )
    newer = Decision(
        profile_id=profile,
        topic="mcp integration",
        decision="Do not use MCP as primary integration",
        context="after review",
        rationale="MemoryProvider is native",
        canonical_key="dec-mcp-no",
        status=STATUS_ACTIVE,
        decided_at=datetime.now(UTC),
        source_event_ids=[event.id],
        extractor_name="manual",
    )
    session.add_all([older, newer])
    await session.flush()

    svc = RetrievalService(profile_id=profile)
    result = await svc.recall(
        session,
        RecallRequest(query="what did we decide about MCP?", include_conflicts=True, mode="balanced"),
    )
    assert result.context_pack.conflicts, "expected polarity conflict between decisions"
    assert result.context_pack.confidence == "low"


@pytest.mark.asyncio
async def test_recall_writes_retrieval_log(session):
    profile = "p1"
    svc = RetrievalService(profile_id=profile)
    await svc.recall(session, RecallRequest(query="nothing here", mode="fast"))
    from sqlalchemy import select

    from siqueira_memo.models import RetrievalLog

    rows = (await session.execute(select(RetrievalLog))).scalars().all()
    assert rows
    assert rows[0].profile_id == profile
    assert rows[0].mode == "fast"


@pytest.mark.asyncio
async def test_recall_ranks_chunks_by_lexical_overlap(session):
    profile = "p1"
    chunk = Chunk(
        profile_id=profile,
        source_type=CHUNK_SOURCE_MESSAGE,
        source_id=uuid.uuid4(),
        chunk_text="Siqueira Memo uses pgvector for hybrid retrieval.",
        token_count=10,
        tokenizer_name="test",
    )
    other = Chunk(
        profile_id=profile,
        source_type=CHUNK_SOURCE_MESSAGE,
        source_id=uuid.uuid4(),
        chunk_text="Coffee preferences: black, no sugar.",
        token_count=6,
        tokenizer_name="test",
    )
    session.add_all([chunk, other])
    await session.flush()

    svc = RetrievalService(profile_id=profile)
    result = await svc.recall(
        session,
        RecallRequest(query="pgvector hybrid retrieval", types=["chunks"], mode="balanced"),
    )
    ids = [str(c.id) for c in result.context_pack.chunks]
    assert ids
    assert ids[0] == str(chunk.id)


def test_context_pack_shaper_drops_deep_mode_payload():
    from siqueira_memo.schemas.recall import ContextPack

    shaper = ContextPackShaper(settings_for_tests())
    pack = ContextPack(
        answer_context="a" * 4000,
        chunks=[],
        decisions=[],
        facts=[],
        summaries=[],
    )
    shaped = shaper.shape_for_prefetch(pack, RECALL_MODE_DEEP)
    assert shaped.chunks == []
    assert "not auto-injected" in " ".join(shaped.warnings)


def test_context_pack_shaper_trims_to_budget():
    from siqueira_memo.schemas.recall import ContextPack, RecallChunk

    shaper = ContextPackShaper(settings_for_tests())
    big_chunks = [
        RecallChunk(
            id=uuid.uuid4(),
            source_type="message",
            source_id=uuid.uuid4(),
            chunk_text=" ".join(["token"] * 400),
            score=0.9,
        )
        for _ in range(10)
    ]
    pack = ContextPack(
        answer_context="short context",
        chunks=big_chunks,
    )
    shaped = shaper.shape_for_prefetch(pack, RECALL_MODE_FAST)
    assert len(shaped.chunks) < len(big_chunks)
    assert shaped.token_estimate <= 1200 + 400  # within fast budget + one trailing chunk tolerance
