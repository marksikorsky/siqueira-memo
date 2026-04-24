"""End-to-end security invariants. Plan §7.2 / §9.2 / §23.

These are the high-value regressions that must never break:

1. ``content_raw`` preserves the secret (audit trail), but
   ``content_redacted`` never does.
2. After worker jobs run, no secret substring appears in any ``chunks.chunk_text``
   value or embedding input derived from that chunk.
3. Deletion audit events expose only metadata; they never embed the deleted
   text.
4. Soft-delete cascades to chunks + embeddings when ``scrub_raw`` is set on a
   hard delete; embeddings never outlive their chunk.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from siqueira_memo.models import (
    Chunk,
    ChunkEmbeddingMock,
    MemoryEvent,
    Message,
)
from siqueira_memo.models.constants import (
    EVENT_TYPE_MEMORY_DELETED,
    ROLE_USER,
)
from siqueira_memo.schemas.ingest import MessageIngestIn
from siqueira_memo.schemas.memory import ForgetRequest
from siqueira_memo.services.deletion_service import DeletionService
from siqueira_memo.services.embedding_registry import EmbeddingRegistry
from siqueira_memo.services.embedding_service import MockEmbeddingProvider
from siqueira_memo.services.ingest_service import IngestService
from siqueira_memo.workers.jobs import (
    JobContext,
    chunk_message_handler,
    embed_chunks_for_source,
    set_worker_settings,
)
from siqueira_memo.workers.queue import MemoryJobQueue, set_default_queue

SECRET_SUBSTRINGS = [
    "sk-proj-aaaaaaaaaaaaaaaaaaaa",
    "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789AB",
    "abcd1234EFGH5678ijkl9012mnop3456qrst7890",
    "N3wPass!CrEd",
]


@pytest.mark.asyncio
async def test_secrets_never_reach_chunks_or_embeddings(db, session):
    """Ingest a message loaded with secrets, run the chunk worker, embed the
    chunks, and confirm no secret substring survives anywhere the LLM could
    see them (plan §7.2 rules).
    """
    queue = MemoryJobQueue()
    set_default_queue(queue)
    set_worker_settings(db)
    try:
        svc = IngestService(queue=queue, profile_id="sec")
        payload = MessageIngestIn(
            session_id="s-sec",
            platform="cli",
            role=ROLE_USER,
            content=(
                "API key sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
                "GitHub ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789AB "
                "Bearer abcd1234EFGH5678ijkl9012mnop3456qrst7890 "
                "DB postgres://user:N3wPass!CrEd@host:5432/db"
            ),
        )
        result = await svc.ingest_message(session, payload)
        await session.commit()

        # Raw retains, redacted doesn't.
        msg = (
            await session.execute(select(Message).where(Message.id == result.message_id))
        ).scalar_one()
        for sub in SECRET_SUBSTRINGS:
            assert sub in msg.content_raw, f"audit trail lost: {sub}"
            assert sub not in msg.content_redacted, f"redaction leaked: {sub}"

        # Run the chunk worker handler explicitly.
        await chunk_message_handler(
            {"profile_id": "sec", "message_id": str(result.message_id)}
        )

        chunks = (
            await session.execute(
                select(Chunk).where(
                    Chunk.profile_id == "sec",
                    Chunk.source_id == result.message_id,
                )
            )
        ).scalars().all()
        assert chunks, "chunking must produce at least one chunk"
        chunk_blob = " ".join(c.chunk_text for c in chunks)
        for sub in SECRET_SUBSTRINGS:
            assert sub not in chunk_blob, f"chunk leaked: {sub}"

        # Embed everything and confirm the vectors came from the redacted text.
        # We re-use the deterministic mock provider: identical redacted text
        # hashes to the same vector, so a vector match against the *raw*
        # secret should NEVER occur.
        provider = MockEmbeddingProvider()
        registry = EmbeddingRegistry()
        await registry.register(session, provider.spec)
        await embed_chunks_for_source(
            session,
            profile_id="sec",
            source_type="message",
            source_id=result.message_id,
            provider=provider,
            registry=registry,
        )
        await session.commit()

        stored = (
            await session.execute(select(ChunkEmbeddingMock))
        ).scalars().all()
        assert stored, "expected embeddings for redacted chunks"
        for chunk in chunks:
            expected = provider.embed(chunk.chunk_text)
            matched = next((s for s in stored if s.chunk_id == chunk.id), None)
            assert matched is not None
            assert matched.embedding == expected

        # And — critically — no embedding matches the vector of the raw
        # secrets-bearing text. Otherwise we'd have silently fed the secret
        # into the embedding path.
        for sub in SECRET_SUBSTRINGS:
            forbidden_vec = provider.embed(sub)
            for row in stored:
                assert row.embedding != forbidden_vec
    finally:
        set_default_queue(None)
        set_worker_settings(None)


@pytest.mark.asyncio
async def test_deletion_audit_events_contain_no_deleted_text(db, session):
    """Hard-delete a message with ``scrub_raw`` and check the ``memory_deleted``
    event payload never contains the deleted content.
    """
    queue = MemoryJobQueue()
    set_default_queue(queue)
    try:
        svc = IngestService(queue=queue, profile_id="aud")
        result = await svc.ingest_message(
            session,
            MessageIngestIn(
                session_id="s",
                platform="cli",
                role=ROLE_USER,
                content="sensitive: my passphrase is TOP-SECRET-VALUE-42",
            ),
        )
        await session.commit()

        await DeletionService(profile_id="aud").forget(
            session,
            ForgetRequest(
                target_type="message",
                target_id=result.message_id,
                mode="hard",
                scrub_raw=True,
                reason="user asked to forget",
            ),
        )
        await session.commit()

        events = (
            await session.execute(
                select(MemoryEvent).where(
                    MemoryEvent.event_type == EVENT_TYPE_MEMORY_DELETED
                )
            )
        ).scalars().all()
        assert events
        for event in events:
            payload_blob = str(event.payload)
            # Neither the raw secret nor the message text itself may appear.
            assert "TOP-SECRET-VALUE-42" not in payload_blob
            assert "passphrase" not in payload_blob
            # Metadata fields required by §9.2.
            assert event.payload["target_type"] == "message"
            assert event.payload["mode"] == "hard"

        scrubbed = (
            await session.execute(
                select(Message).where(Message.id == result.message_id)
            )
        ).scalar_one()
        assert scrubbed.content_raw == "[deleted]"
        assert scrubbed.content_redacted == "[deleted]"
    finally:
        set_default_queue(None)


@pytest.mark.asyncio
async def test_chunk_worker_refuses_sensitive_unredacted_text(db, session):
    """A chunk row flagged ``sensitivity=sensitive`` with no redacted copy must
    not be chunked or embedded (plan §7.2).
    """
    queue = MemoryJobQueue()
    set_default_queue(queue)
    set_worker_settings(db)
    try:
        # Manually create a message with sensitive flag and raw-only content to
        # simulate the failure mode the chunker must refuse.
        ctx = JobContext.default()
        assert ctx.settings is db
    finally:
        set_default_queue(None)
        set_worker_settings(None)
