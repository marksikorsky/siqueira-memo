"""Forget / deletion cascade tests. Plan §9 / §24."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from siqueira_memo.models import Chunk, Decision, Fact, MemoryEvent, Message
from siqueira_memo.models.constants import (
    CHUNK_SOURCE_MESSAGE,
    EVENT_TYPE_MEMORY_DELETED,
    STATUS_ACTIVE,
    STATUS_DELETED,
)
from siqueira_memo.schemas.memory import ForgetRequest
from siqueira_memo.services.deletion_service import DeletionService


async def _seed_message(session, profile_id: str) -> tuple[MemoryEvent, Message, Chunk]:
    event = MemoryEvent(
        event_type="message_received",
        source="test",
        actor="test",
        profile_id=profile_id,
        session_id="s",
        payload={"event_type": "message_received", "message_id": "x", "role": "user", "platform": "cli", "content_hash": "h"},
    )
    session.add(event)
    await session.flush()
    message = Message(
        event_id=event.id,
        profile_id=profile_id,
        session_id="s",
        platform="cli",
        role="user",
        content_raw="Remember: use pgvector.",
        content_redacted="Remember: use pgvector.",
        content_hash="h",
    )
    session.add(message)
    await session.flush()
    chunk = Chunk(
        profile_id=profile_id,
        source_type=CHUNK_SOURCE_MESSAGE,
        source_id=message.id,
        chunk_text="Remember: use pgvector.",
        token_count=5,
        tokenizer_name="t",
    )
    session.add(chunk)
    await session.flush()
    return event, message, chunk


@pytest.mark.asyncio
async def test_forget_fact_soft_marks_deleted(db, session):
    profile = "p1"
    event = MemoryEvent(
        event_type="fact_extracted",
        source="test",
        actor="test",
        profile_id=profile,
        payload={"event_type": "fact_extracted", "fact_id": "x", "canonical_key": "k", "status": "active", "confidence": 1.0},
    )
    session.add(event)
    await session.flush()
    fact = Fact(
        profile_id=profile,
        subject="x",
        predicate="y",
        object="z",
        statement="x y z",
        canonical_key="xyz",
        status=STATUS_ACTIVE,
        source_event_ids=[event.id],
    )
    session.add(fact)
    await session.flush()

    svc = DeletionService(profile_id=profile)
    response = await svc.forget(
        session,
        ForgetRequest(target_type="fact", target_id=fact.id, mode="soft"),
    )
    assert response.invalidated_facts == 1
    refreshed = (await session.execute(select(Fact).where(Fact.id == fact.id))).scalar_one()
    assert refreshed.status == STATUS_DELETED

    # A MemoryEvent audit entry was written.
    audit = (
        await session.execute(
            select(MemoryEvent).where(MemoryEvent.event_type == EVENT_TYPE_MEMORY_DELETED)
        )
    ).scalar_one()
    assert audit.payload["target_id"] == str(fact.id)


@pytest.mark.asyncio
async def test_forget_message_hard_cascades_to_chunks(db, session):
    profile = "p1"
    _event, message, chunk = await _seed_message(session, profile)
    svc = DeletionService(profile_id=profile)
    response = await svc.forget(
        session,
        ForgetRequest(
            target_type="message",
            target_id=message.id,
            mode="hard",
            scrub_raw=True,
        ),
    )
    assert response.removed_chunks == 1
    remaining = (await session.execute(select(Chunk).where(Chunk.id == chunk.id))).scalar_one_or_none()
    assert remaining is None
    updated = (await session.execute(select(Message).where(Message.id == message.id))).scalar_one()
    assert updated.content_raw == "[deleted]"
    assert updated.content_redacted == "[deleted]"


@pytest.mark.asyncio
async def test_forget_decision_hard_removes_row(db, session):
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
        topic="test",
        decision="unused",
        context="test",
        canonical_key="k",
        status=STATUS_ACTIVE,
        decided_at=datetime.now(UTC),
        source_event_ids=[event.id],
    )
    session.add(decision)
    await session.flush()
    decision_id = decision.id

    svc = DeletionService(profile_id=profile)
    await svc.forget(session, ForgetRequest(target_type="decision", target_id=decision_id, mode="hard"))
    remaining = (await session.execute(select(Decision).where(Decision.id == decision_id))).scalar_one_or_none()
    assert remaining is None
