"""Forget / deletion cascade. Plan §9.2 / §24.

Supports two modes:

* ``soft`` — marks the target ``status`` as ``deleted``; derived summaries that
  cite it are marked ``stale`` (plan §24.1). Raw content and embeddings stay.
* ``hard`` — removes derived chunks + embeddings, invalidates facts/decisions
  whose only sources point at the target, marks summaries based on the
  percentage of affected sources. The raw message/tool event/artifact text is
  scrubbed when ``scrub_raw`` is requested; the audit event retains only
  metadata, never deleted content (plan §9.2).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.config import Settings, get_settings
from siqueira_memo.logging import get_logger
from siqueira_memo.models import (
    Chunk,
    ChunkEmbeddingBGEM3,
    ChunkEmbeddingMock,
    ChunkEmbeddingOpenAITEL3,
    Decision,
    DecisionSource,
    Fact,
    FactSource,
    MemoryEvent,
    Message,
    ProjectState,
    SessionSummary,
    TopicSummary,
)
from siqueira_memo.models.constants import (
    EVENT_TYPE_MEMORY_DELETED,
    STATUS_DELETED,
    STATUS_INVALIDATED,
    STATUS_STALE,
)
from siqueira_memo.schemas.memory import ForgetRequest, ForgetResponse

log = get_logger(__name__)


_EMBEDDING_TABLES = (
    ChunkEmbeddingMock,
    ChunkEmbeddingOpenAITEL3,
    ChunkEmbeddingBGEM3,
)


@dataclass
class DeletionService:
    profile_id: str
    settings: Settings | None = None

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()

    # ------------------------------------------------------------------
    # Top level dispatch
    # ------------------------------------------------------------------
    async def forget(
        self, session: AsyncSession, request: ForgetRequest
    ) -> ForgetResponse:
        assert self.settings is not None
        profile_id = request.profile_id or self.profile_id
        target_id = request.target_id
        target_type = request.target_type
        mode = request.mode

        invalidated_facts = 0
        invalidated_decisions = 0
        removed_chunks = 0
        removed_embeddings = 0
        regenerated_summaries = 0

        if target_type == "fact":
            invalidated_facts = await self._invalidate_fact(session, profile_id, target_id, mode)
        elif target_type == "decision":
            invalidated_decisions = await self._invalidate_decision(
                session, profile_id, target_id, mode
            )
        elif target_type == "summary":
            regenerated_summaries = await self._mark_summary_deleted(
                session, profile_id, target_id
            )
        elif target_type == "message":
            result = await self._forget_message(
                session, profile_id, target_id, mode=mode, scrub_raw=request.scrub_raw
            )
            removed_chunks = result["chunks"]
            removed_embeddings = result["embeddings"]
            regenerated_summaries = result["summaries"]
            invalidated_facts += result["facts"]
            invalidated_decisions += result["decisions"]
        elif target_type == "chunk":
            removed = await self._forget_chunk(session, profile_id, target_id, mode)
            removed_chunks = removed["chunks"]
            removed_embeddings = removed["embeddings"]
        elif target_type == "entity":
            invalidated_facts = await self._invalidate_entity_dependent_facts(
                session, profile_id, target_id
            )
        elif target_type == "session":
            result = await self._forget_session(
                session, profile_id, target_id, scrub_raw=request.scrub_raw
            )
            removed_chunks = result["chunks"]
            removed_embeddings = result["embeddings"]
            regenerated_summaries = result["summaries"]
            invalidated_facts = result["facts"]
            invalidated_decisions = result["decisions"]
        else:  # pragma: no cover
            raise ValueError(f"unsupported target_type: {target_type}")

        event_id = uuid.uuid4()
        session.add(
            MemoryEvent(
                id=event_id,
                event_type=EVENT_TYPE_MEMORY_DELETED,
                source="forget_api",
                actor="user",
                profile_id=profile_id,
                payload={
                    "event_type": EVENT_TYPE_MEMORY_DELETED,
                    "target_type": target_type,
                    "target_id": str(target_id),
                    "mode": mode,
                    "reason": request.reason,
                    "removed_chunks": removed_chunks,
                    "removed_embeddings": removed_embeddings,
                },
            )
        )
        await session.flush()

        log.info(
            "forget",
            extra={
                "target_type": target_type,
                "target_id": str(target_id),
                "mode": mode,
                "profile_id": profile_id,
                "invalidated_facts": invalidated_facts,
                "invalidated_decisions": invalidated_decisions,
                "removed_chunks": removed_chunks,
                "removed_embeddings": removed_embeddings,
                "regenerated_summaries": regenerated_summaries,
            },
        )

        return ForgetResponse(
            event_id=event_id,
            target_id=target_id,
            mode=mode,
            invalidated_facts=invalidated_facts,
            invalidated_decisions=invalidated_decisions,
            removed_chunks=removed_chunks,
            removed_embeddings=removed_embeddings,
            regenerated_summaries=regenerated_summaries,
        )

    # ------------------------------------------------------------------
    # Individual target handlers
    # ------------------------------------------------------------------
    async def _invalidate_fact(
        self, session: AsyncSession, profile_id: str, target_id: uuid.UUID, mode: str
    ) -> int:
        row = (
            await session.execute(
                select(Fact).where(Fact.id == target_id, Fact.profile_id == profile_id)
            )
        ).scalar_one_or_none()
        if row is None:
            return 0
        if mode == "hard":
            await session.execute(delete(FactSource).where(FactSource.fact_id == row.id))
            await session.delete(row)
        else:
            row.status = STATUS_DELETED
        return 1

    async def _invalidate_decision(
        self, session: AsyncSession, profile_id: str, target_id: uuid.UUID, mode: str
    ) -> int:
        row = (
            await session.execute(
                select(Decision).where(
                    Decision.id == target_id, Decision.profile_id == profile_id
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return 0
        if mode == "hard":
            await session.execute(
                delete(DecisionSource).where(DecisionSource.decision_id == row.id)
            )
            await session.delete(row)
        else:
            row.status = STATUS_DELETED
        return 1

    async def _mark_summary_deleted(
        self, session: AsyncSession, profile_id: str, target_id: uuid.UUID
    ) -> int:
        for model in (SessionSummary, TopicSummary, ProjectState):
            row: Any = (
                await session.execute(
                    select(model).where(
                        model.id == target_id, model.profile_id == profile_id
                    )
                )
            ).scalar_one_or_none()
            if row is not None:
                row.status = STATUS_DELETED
                return 1
        return 0

    async def _forget_chunk(
        self, session: AsyncSession, profile_id: str, target_id: uuid.UUID, mode: str
    ) -> dict[str, int]:
        chunk = (
            await session.execute(
                select(Chunk).where(Chunk.id == target_id, Chunk.profile_id == profile_id)
            )
        ).scalar_one_or_none()
        if chunk is None:
            return {"chunks": 0, "embeddings": 0}
        removed_embeddings = 0
        for table in _EMBEDDING_TABLES:
            result = await session.execute(
                delete(table).where(table.chunk_id == chunk.id)
            )
            removed_embeddings += getattr(result, "rowcount", 0) or 0
        await session.delete(chunk)
        return {"chunks": 1, "embeddings": removed_embeddings}

    async def _invalidate_entity_dependent_facts(
        self, session: AsyncSession, profile_id: str, target_id: uuid.UUID
    ) -> int:
        # Entities are identified by name; callers usually delete the row itself.
        # We conservatively invalidate facts that reference the entity in their
        # ``extra_metadata["entity_id"]`` field.
        rows = (
            await session.execute(
                select(Fact).where(Fact.profile_id == profile_id, Fact.status == "active")
            )
        ).scalars().all()
        touched = 0
        for row in rows:
            meta = row.extra_metadata or {}
            if str(meta.get("entity_id")) == str(target_id):
                row.status = STATUS_INVALIDATED
                touched += 1
        return touched

    async def _forget_message(
        self,
        session: AsyncSession,
        profile_id: str,
        target_id: uuid.UUID,
        *,
        mode: str,
        scrub_raw: bool,
    ) -> dict[str, int]:
        message = (
            await session.execute(
                select(Message).where(
                    Message.id == target_id, Message.profile_id == profile_id
                )
            )
        ).scalar_one_or_none()
        if message is None:
            return {"chunks": 0, "embeddings": 0, "summaries": 0, "facts": 0, "decisions": 0}

        # Remove derived chunks + embeddings keyed on source_id == message.id.
        chunk_rows = (
            await session.execute(
                select(Chunk).where(
                    Chunk.profile_id == profile_id,
                    Chunk.source_type == "message",
                    Chunk.source_id == message.id,
                )
            )
        ).scalars().all()
        removed_embeddings = 0
        for ch in chunk_rows:
            for table in _EMBEDDING_TABLES:
                result = await session.execute(
                    delete(table).where(table.chunk_id == ch.id)
                )
                removed_embeddings += getattr(result, "rowcount", 0) or 0
            await session.delete(ch)

        # Invalidate derived facts/decisions whose only source is this event.
        facts = (
            await session.execute(
                select(Fact).where(
                    Fact.profile_id == profile_id, Fact.source_message_ids.isnot(None)
                )
            )
        ).scalars().all()
        invalidated_facts = 0
        for fact in facts:
            ids = set(str(x) for x in (fact.source_message_ids or []))
            if str(message.id) in ids:
                ids.discard(str(message.id))
                if not ids:
                    fact.status = STATUS_INVALIDATED
                    invalidated_facts += 1
                else:
                    fact.extra_metadata = {
                        **(fact.extra_metadata or {}),
                        "needs_reverification": True,
                    }
                    fact.source_message_ids = [uuid.UUID(x) for x in ids]

        decisions = (
            await session.execute(
                select(Decision).where(
                    Decision.profile_id == profile_id,
                    Decision.source_message_ids.isnot(None),
                )
            )
        ).scalars().all()
        invalidated_decisions = 0
        for decision in decisions:
            ids = set(str(x) for x in (decision.source_message_ids or []))
            if str(message.id) in ids:
                ids.discard(str(message.id))
                if not ids:
                    decision.status = STATUS_INVALIDATED
                    invalidated_decisions += 1
                else:
                    decision.source_message_ids = [uuid.UUID(x) for x in ids]
                    decision.extra_metadata = {
                        **(decision.extra_metadata or {}),
                        "needs_reverification": True,
                    }

        # Mark summaries that depended on this message stale (plan §24.1 thresholds).
        summaries_touched = 0
        for model in (SessionSummary, TopicSummary, ProjectState):
            rows: list[Any] = list(
                (
                    await session.execute(
                        select(model).where(model.profile_id == profile_id)
                    )
                )
                .scalars()
                .all()
            )
            for row in rows:
                sources = [str(x) for x in (row.source_event_ids or [])]
                if str(message.event_id) in sources and row.status != STATUS_DELETED:
                    assert self.settings is not None
                    affected_ratio = sources.count(str(message.event_id)) / max(1, len(sources))
                    if affected_ratio >= self.settings.summary_invalid_threshold:
                        row.status = STATUS_INVALIDATED
                    else:
                        row.status = STATUS_STALE
                    summaries_touched += 1

        if mode == "hard":
            if scrub_raw:
                message.content_raw = "[deleted]"
                message.content_redacted = "[deleted]"
                message.extra_metadata = {
                    **(message.extra_metadata or {}),
                    "scrubbed": True,
                }
            else:
                message.sensitivity = "sensitive"
                message.extra_metadata = {
                    **(message.extra_metadata or {}),
                    "hidden": True,
                }
        else:
            message.extra_metadata = {
                **(message.extra_metadata or {}),
                "soft_deleted": True,
            }
        return {
            "chunks": len(chunk_rows),
            "embeddings": removed_embeddings,
            "summaries": summaries_touched,
            "facts": invalidated_facts,
            "decisions": invalidated_decisions,
        }

    async def _forget_session(
        self,
        session: AsyncSession,
        profile_id: str,
        session_value: uuid.UUID,
        *,
        scrub_raw: bool,
    ) -> dict[str, int]:
        # ``target_id`` is a UUID in the schema, but ``session_id`` is a text
        # identifier. We treat the stringified UUID as the session_id.
        session_id = str(session_value)
        removed_chunks = 0
        removed_embeddings = 0
        summaries_touched = 0
        messages = (
            await session.execute(
                select(Message).where(
                    Message.profile_id == profile_id, Message.session_id == session_id
                )
            )
        ).scalars().all()
        for m in messages:
            res = await self._forget_message(
                session, profile_id, m.id, mode="hard", scrub_raw=scrub_raw
            )
            removed_chunks += res["chunks"]
            removed_embeddings += res["embeddings"]
            summaries_touched += res["summaries"]
        facts_touched = 0
        decisions_touched = 0
        return {
            "chunks": removed_chunks,
            "embeddings": removed_embeddings,
            "summaries": summaries_touched,
            "facts": facts_touched,
            "decisions": decisions_touched,
        }


def ratio_affected(
    deleted_event_ids: Iterable[uuid.UUID], total_source_ids: Iterable[uuid.UUID]
) -> float:
    deleted = {str(x) for x in deleted_event_ids}
    total = [str(x) for x in total_source_ids]
    if not total:
        return 0.0
    return sum(1 for t in total if t in deleted) / len(total)
