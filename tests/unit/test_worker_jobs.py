"""Worker job handlers. Plan §7.1."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from siqueira_memo.models import Chunk, Message
from siqueira_memo.models.constants import ROLE_USER
from siqueira_memo.schemas.ingest import MessageIngestIn
from siqueira_memo.services.ingest_service import IngestService
from siqueira_memo.workers.jobs import register_default_handlers, set_worker_settings
from siqueira_memo.workers.queue import MemoryJobQueue, set_default_queue


@pytest.mark.asyncio
async def test_chunk_message_handler_creates_chunk(db, session):
    queue = MemoryJobQueue()
    register_default_handlers(queue)
    set_default_queue(queue)
    set_worker_settings(db)
    try:
        svc = IngestService(queue=queue, profile_id="p1")
        result = await svc.ingest_message(
            session,
            MessageIngestIn(
                session_id="s", platform="cli", role=ROLE_USER, content="hello world"
            ),
        )
        await session.commit()

        drained = await queue.drain()
        assert drained >= 2

        chunks = (
            await session.execute(
                select(Chunk).where(Chunk.source_id == result.message_id)
            )
        ).scalars().all()
        assert chunks

        message = (
            await session.execute(select(Message).where(Message.id == result.message_id))
        ).scalar_one()
        assert "gate_labels" in (message.extra_metadata or {})
    finally:
        set_default_queue(None)
        set_worker_settings(None)
