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
from siqueira_memo.services.markdown_export import ExportFilter, export_markdown
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


@pytest.mark.asyncio
async def test_recall_excludes_secret_fact_until_explicitly_allowed(session):
    profile = "p1"
    raw_secret = "sk-proj-" + "b" * 40
    event = MemoryEvent(
        event_type="fact_extracted",
        source="test",
        actor="test",
        profile_id=profile,
        payload={"event_type": "fact_extracted"},
    )
    session.add(event)
    await session.flush()
    secret_fact = Fact(
        profile_id=profile,
        subject="OpenAI staging key",
        predicate="stored_as_secret",
        object=raw_secret,
        statement=f"OpenAI staging key is {raw_secret}.",
        canonical_key="fact-secret-openai-staging",
        status=STATUS_ACTIVE,
        confidence=0.98,
        source_event_ids=[event.id],
        topic="secrets",
        extractor_name="manual",
        extra_metadata={
            "sensitivity": "secret",
            "recall_policy": "explicit_or_high_relevance",
            "masked_preview": "OpenAI staging key is sk-proj-...bbbb.",
            "secret_kind": "api_key",
            "secret_value": raw_secret,
        },
    )
    public_fact = Fact(
        profile_id=profile,
        subject="OpenAI staging service",
        predicate="runs_on",
        object="staging cluster",
        statement="OpenAI staging service runs on the staging cluster.",
        canonical_key="fact-openai-staging-service",
        status=STATUS_ACTIVE,
        confidence=0.8,
        source_event_ids=[event.id],
        topic="secrets",
        extractor_name="manual",
    )
    session.add_all([secret_fact, public_fact])
    await session.flush()

    svc = RetrievalService(profile_id=profile)
    normal = await svc.recall(
        session,
        RecallRequest(query="OpenAI staging key", types=["facts"], mode="balanced"),
    )
    assert str(secret_fact.id) not in {str(f.id) for f in normal.context_pack.facts}
    assert raw_secret not in normal.context_pack.answer_context
    assert any("secret" in warning.lower() for warning in normal.context_pack.warnings)

    explicit = await svc.recall(
        session,
        RecallRequest(
            query="OpenAI staging key",
            types=["facts"],
            mode="balanced",
            allow_secret_recall=True,
        ),
    )
    recalled_secret = next(f for f in explicit.context_pack.facts if f.id == secret_fact.id)
    assert recalled_secret.sensitivity == "secret"
    assert recalled_secret.masked_preview == "OpenAI staging key is sk-proj-...bbbb."
    assert raw_secret not in recalled_secret.statement
    assert raw_secret not in explicit.context_pack.answer_context


def test_context_pack_shaper_excludes_secret_items_from_prefetch():
    from siqueira_memo.schemas.recall import ContextPack, RecallFact

    raw_secret = "sk-proj-" + "c" * 40
    shaper = ContextPackShaper(settings_for_tests())
    pack = ContextPack(
        answer_context=f"Known facts:\n- Secret token {raw_secret}",
        facts=[
            RecallFact(
                id=uuid.uuid4(),
                subject="secret token",
                predicate="stored_as_secret",
                object=raw_secret,
                statement=f"Secret token {raw_secret}",
                status=STATUS_ACTIVE,
                confidence=0.9,
                sensitivity="secret",
                masked_preview="Secret token sk-proj-...cccc",
            )
        ],
    )

    shaped = shaper.shape_for_prefetch(pack, RECALL_MODE_FAST)
    assert shaped.facts == []
    assert raw_secret not in shaped.answer_context
    assert any("secret" in warning.lower() for warning in shaped.warnings)


@pytest.mark.asyncio
async def test_markdown_export_masks_secret_facts(session):
    profile = "p1"
    raw_secret = "sk-proj-" + "d" * 40
    event = MemoryEvent(
        event_type="fact_extracted",
        source="test",
        actor="test",
        profile_id=profile,
        payload={"event_type": "fact_extracted"},
    )
    session.add(event)
    await session.flush()
    session.add(
        Fact(
            profile_id=profile,
            subject="OpenAI export key",
            predicate="stored_as_secret",
            object=raw_secret,
            statement=f"OpenAI export key is {raw_secret}.",
            canonical_key="fact-secret-openai-export",
            status=STATUS_ACTIVE,
            confidence=0.98,
            source_event_ids=[event.id],
            project="siqueira-memo",
            topic="secrets",
            extractor_name="manual",
            extra_metadata={
                "sensitivity": "secret",
                "masked_preview": "OpenAI export key is sk-proj-...dddd.",
                "secret_value": raw_secret,
            },
        )
    )
    await session.flush()

    body = await export_markdown(session, ExportFilter(profile_id=profile, project="siqueira-memo"))

    assert raw_secret not in body
    assert "sk-proj-...dddd" in body
    assert "secret" in body.lower()


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
