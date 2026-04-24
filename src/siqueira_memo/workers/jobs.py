"""Job handlers registered against the queue.

The in-memory queue is sufficient for unit tests and single-process dev runs.
Production should swap to the Redis queue; job payloads must stay JSON-safe so
both backends behave identically.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.config import Settings, get_settings
from siqueira_memo.db import get_session_factory
from siqueira_memo.logging import get_logger
from siqueira_memo.models import Chunk, Message, ToolEvent
from siqueira_memo.models.constants import (
    CHUNK_SOURCE_MESSAGE,
    CHUNK_SOURCE_TOOL_OUTPUT,
)
from siqueira_memo.services.chunking_service import ChunkingService
from siqueira_memo.services.embedding_registry import EmbeddingRegistry
from siqueira_memo.services.embedding_service import (
    EmbeddingProvider,
    build_embedding_provider,
)
from siqueira_memo.services.extraction_gate import default_gate
from siqueira_memo.workers.queue import JobQueue

_settings_override: Settings | None = None


def set_worker_settings(settings: Settings | None) -> None:
    """Override the settings used by background jobs.

    Tests inject their ephemeral settings here so handlers write to the same
    DB the assertions read from. In production this stays ``None`` and the
    handlers fall back to the cached ``get_settings()``.
    """
    global _settings_override
    _settings_override = settings


def _current_settings() -> Settings:
    return _settings_override or get_settings()

log = get_logger(__name__)


@dataclass
class JobContext:
    settings: Settings
    chunker: ChunkingService

    @classmethod
    def default(cls) -> JobContext:
        return cls(settings=_current_settings(), chunker=ChunkingService())


async def chunk_message_handler(payload: dict[str, Any]) -> None:
    ctx = JobContext.default()
    factory = get_session_factory(ctx.settings)
    message_id = uuid.UUID(payload["message_id"])
    profile_id = payload["profile_id"]
    async with factory() as session:
        message = (
            await session.execute(
                select(Message).where(
                    Message.id == message_id, Message.profile_id == profile_id
                )
            )
        ).scalar_one_or_none()
        if message is None:
            log.debug("chunk_message.missing", extra={"message_id": str(message_id)})
            return
        if message.sensitivity == "sensitive":
            log.debug("chunk_message.skip_sensitive")
            return
        chunks = ctx.chunker.chunk_message(
            message.content_redacted,
            source_id=str(message.id),
            sensitivity=message.sensitivity,
        )
        for chunk in chunks:
            session.add(
                Chunk(
                    profile_id=profile_id,
                    source_type=CHUNK_SOURCE_MESSAGE,
                    source_id=message.id,
                    chunk_text=chunk.chunk_text,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                    tokenizer_name=chunk.tokenizer_name,
                    tokenizer_version=chunk.tokenizer_version,
                    project=message.project,
                    topic=message.topic,
                    entities=list(message.entities or []),
                    sensitivity=message.sensitivity,
                    extra_metadata={**chunk.extra_metadata, **(message.extra_metadata or {})},
                )
            )
        await session.commit()
    log.info(
        "chunk_message.done",
        extra={"message_id": str(message_id), "chunks": len(chunks)},
    )


async def chunk_tool_output_handler(payload: dict[str, Any]) -> None:
    ctx = JobContext.default()
    factory = get_session_factory(ctx.settings)
    tool_event_id = uuid.UUID(payload["tool_event_id"])
    profile_id = payload["profile_id"]
    async with factory() as session:
        tool_event = (
            await session.execute(
                select(ToolEvent).where(
                    ToolEvent.id == tool_event_id, ToolEvent.profile_id == profile_id
                )
            )
        ).scalar_one_or_none()
        if tool_event is None:
            return
        text = tool_event.output_redacted or tool_event.output_summary
        if not text:
            return
        chunks = ctx.chunker.chunk_log(text, source_id=str(tool_event.id))
        for chunk in chunks:
            session.add(
                Chunk(
                    profile_id=profile_id,
                    source_type=CHUNK_SOURCE_TOOL_OUTPUT,
                    source_id=tool_event.id,
                    chunk_text=chunk.chunk_text,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                    tokenizer_name=chunk.tokenizer_name,
                    tokenizer_version=chunk.tokenizer_version,
                    sensitivity=tool_event.sensitivity,
                    extra_metadata={**chunk.extra_metadata},
                )
            )
        await session.commit()


async def extraction_gate_handler(payload: dict[str, Any]) -> None:
    ctx = JobContext.default()
    factory = get_session_factory(ctx.settings)
    message_id = uuid.UUID(payload["message_id"])
    profile_id = payload["profile_id"]
    async with factory() as session:
        message = (
            await session.execute(
                select(Message).where(
                    Message.id == message_id, Message.profile_id == profile_id
                )
            )
        ).scalar_one_or_none()
        if message is None:
            return
        result = default_gate.classify(message.content_redacted, role=message.role)
        message.extra_metadata = {
            **(message.extra_metadata or {}),
            "gate_labels": list(result.labels),
            "gate_confidence": result.confidence,
        }
        await session.commit()


async def embed_chunks_handler(payload: dict[str, Any]) -> None:
    """Embed chunks for a single source (message/tool_event/artifact).

    Payload keys:
    - ``profile_id`` (required)
    - ``source_type`` (required, e.g. "message")
    - ``source_id`` (required, UUID string)
    """
    ctx = JobContext.default()
    factory = get_session_factory(ctx.settings)
    profile_id = payload["profile_id"]
    source_type = payload["source_type"]
    source_id = uuid.UUID(payload["source_id"])
    async with factory() as session:
        provider = build_embedding_provider(ctx.settings)
        registry = EmbeddingRegistry()
        spec = provider.spec
        await registry.register(session, spec)
        embedded = await embed_chunks_for_source(
            session,
            profile_id=profile_id,
            source_type=source_type,
            source_id=source_id,
            provider=provider,
            registry=registry,
        )
        await session.commit()
    log.info(
        "embed_chunks.done",
        extra={
            "profile_id": profile_id,
            "source_id": str(source_id),
            "embedded": embedded,
        },
    )


async def embed_chunks_for_source(
    session: AsyncSession,
    *,
    profile_id: str,
    source_type: str,
    source_id: uuid.UUID,
    provider: EmbeddingProvider,
    registry: EmbeddingRegistry,
) -> int:
    """Embed every unembedded chunk matching ``(profile_id, source_type, source_id)``.

    Returns the number of new embeddings written. Idempotent: relies on the
    per-embedding table's unique ``(chunk_id, model_version)`` constraint to
    skip chunks that already have a current-version embedding.
    """
    from sqlalchemy.exc import IntegrityError

    rows = (
        await session.execute(
            select(Chunk).where(
                Chunk.profile_id == profile_id,
                Chunk.source_type == source_type,
                Chunk.source_id == source_id,
            )
        )
    ).scalars().all()
    embedded = 0
    for chunk in rows:
        if chunk.sensitivity == "sensitive":
            continue
        try:
            async with session.begin_nested():
                vec = provider.embed(chunk.chunk_text)
                await registry.store_embedding(
                    session, chunk_id=chunk.id, vector=vec, spec=provider.spec
                )
                embedded += 1
        except IntegrityError:
            # Already embedded for this model version.
            continue
    return embedded


def register_default_handlers(queue: JobQueue) -> None:
    queue.register("chunk_message", chunk_message_handler)
    queue.register("chunk_tool_output", chunk_tool_output_handler)
    queue.register("extraction_gate", extraction_gate_handler)
    queue.register("embed_chunks", embed_chunks_handler)
