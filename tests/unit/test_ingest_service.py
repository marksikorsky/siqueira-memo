"""Ingest service tests. Plan §4.2 / §3.1 / §3.2 / §7.2."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from siqueira_memo.models import MemoryEvent, Message, ToolEvent
from siqueira_memo.models.constants import (
    EVENT_TYPE_MESSAGE_RECEIVED,
    EVENT_TYPE_TOOL_CALLED,
    ROLE_USER,
)
from siqueira_memo.schemas import (
    HermesAuxCompactionIn,
    MessageIngestIn,
    ToolEventIngestIn,
)
from siqueira_memo.services.ingest_service import IngestService


@pytest.mark.asyncio
async def test_message_ingest_creates_event_and_message(db, session, queue):
    svc = IngestService(queue=queue, profile_id="test-profile")
    payload = MessageIngestIn(
        session_id="s1",
        platform="cli",
        role=ROLE_USER,
        content="Привет, Hermes. Давай решим архитектуру.",
    )
    result = await svc.ingest_message(session, payload)

    assert result.message_id is not None
    assert result.event_id is not None

    event = (await session.execute(select(MemoryEvent).where(MemoryEvent.id == result.event_id))).scalar_one()
    assert event.event_type == EVENT_TYPE_MESSAGE_RECEIVED
    assert event.session_id == "s1"
    assert event.profile_id == "test-profile"

    message = (await session.execute(select(Message).where(Message.id == result.message_id))).scalar_one()
    assert message.content_raw == "Привет, Hermes. Давай решим архитектуру."
    assert message.content_redacted == "Привет, Hermes. Давай решим архитектуру."
    assert message.event_id == event.id
    assert message.role == ROLE_USER


@pytest.mark.asyncio
async def test_message_ingest_redacts_secrets(db, session, queue):
    svc = IngestService(queue=queue, profile_id="test-profile")
    payload = MessageIngestIn(
        session_id="s1",
        platform="cli",
        role=ROLE_USER,
        content=(
            "Забудь этот старый токен sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ),
    )
    result = await svc.ingest_message(session, payload)
    assert result.redactions >= 1

    message = (await session.execute(select(Message).where(Message.id == result.message_id))).scalar_one()
    assert "sk-proj-aaaa" not in message.content_redacted
    # Raw must retain the secret so the audit trail is intact.
    assert "sk-proj-aaaa" in message.content_raw
    assert message.sensitivity in {"elevated", "sensitive", "normal"}


@pytest.mark.asyncio
async def test_message_ingest_is_idempotent_on_dedupe(db, session, queue):
    svc = IngestService(queue=queue, profile_id="test-profile")
    payload = MessageIngestIn(
        session_id="s1",
        platform="cli",
        role=ROLE_USER,
        content="same message",
        source="sync_turn",
    )
    first = await svc.ingest_message(session, payload)
    second = await svc.ingest_message(session, payload)
    assert first.message_id == second.message_id
    assert second.duplicate is True


@pytest.mark.asyncio
async def test_message_ingest_enqueues_async_jobs(db, session, queue):
    svc = IngestService(queue=queue, profile_id="test-profile")
    await svc.ingest_message(
        session,
        MessageIngestIn(
            session_id="s1",
            platform="cli",
            role=ROLE_USER,
            content="short message",
        ),
    )
    pending_jobs = queue.pending()
    assert pending_jobs >= 2  # at least chunk + extraction-gate jobs.


@pytest.mark.asyncio
async def test_message_ingest_respects_agent_context(db, session, queue):
    svc = IngestService(queue=queue, profile_id="test-profile")
    payload = MessageIngestIn(
        session_id="s1",
        platform="cli",
        role=ROLE_USER,
        content="subagent chatter",
        agent_context="subagent",
    )
    result = await svc.ingest_message(session, payload)
    assert result is not None
    # Subagent ingests are stored but with agent_context propagated.
    event = (await session.execute(select(MemoryEvent).where(MemoryEvent.id == result.event_id))).scalar_one()
    assert event.agent_context == "subagent"


@pytest.mark.asyncio
async def test_tool_event_ingest_redacts_input_and_output(db, session, queue):
    svc = IngestService(queue=queue, profile_id="test-profile")
    payload = ToolEventIngestIn(
        session_id="s1",
        tool_name="bash",
        input={"cmd": "echo sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
        output="result token: Bearer abcd1234EFGH5678ijkl9012mnop3456qrst7890",
    )
    result = await svc.ingest_tool_event(session, payload)
    assert result.tool_event_id is not None

    evt = (await session.execute(select(ToolEvent).where(ToolEvent.id == result.tool_event_id))).scalar_one()
    assert "sk-proj-aaaa" not in str(evt.input_redacted)
    assert "abcd1234EFGH" not in (evt.output_redacted or "")
    event = (await session.execute(select(MemoryEvent).where(MemoryEvent.id == evt.event_id))).scalar_one()
    assert event.event_type == EVENT_TYPE_TOOL_CALLED


@pytest.mark.asyncio
async def test_ingest_hermes_auxiliary_compaction(db, session, queue):
    svc = IngestService(queue=queue, profile_id="test-profile")
    result = await svc.ingest_hermes_aux_compaction(
        session,
        HermesAuxCompactionIn(
            session_id="s1",
            summary_text="[CONTEXT COMPACTION — REFERENCE ONLY] summary body",
            source_message_count=42,
        ),
    )
    event = (await session.execute(select(MemoryEvent).where(MemoryEvent.id == result.event_id))).scalar_one()
    assert event.event_type == "hermes_auxiliary_compaction_observed"
    assert event.payload["source_message_count"] == 42
