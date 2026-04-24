"""Ingest path: write events, messages, tool events, artifacts durably.

Every method:

1. validates the incoming Pydantic model;
2. creates a single ``memory_events`` row with a discriminated payload;
3. runs redaction **before** any derived content is stored;
4. writes the raw + redacted fields in one SQL transaction;
5. enqueues follow-up jobs (chunking, extraction-gate) non-blockingly.

Redaction counts are logged but never the contents. See plan §3.1 / §7.2 / §10.1.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.config import get_settings
from siqueira_memo.logging import get_logger
from siqueira_memo.models import Artifact, MemoryEvent, Message, ToolEvent
from siqueira_memo.models.constants import (
    AGENT_CONTEXT_PRIMARY,
    EVENT_TYPE_ARTIFACT_CREATED,
    EVENT_TYPE_ASSISTANT_MESSAGE_SENT,
    EVENT_TYPE_BUILTIN_MEMORY_MIRROR,
    EVENT_TYPE_DELEGATION_OBSERVED,
    EVENT_TYPE_HERMES_AUX_COMPACTION_OBSERVED,
    EVENT_TYPE_MESSAGE_RECEIVED,
    EVENT_TYPE_TOOL_CALLED,
    ROLE_ASSISTANT,
    SENSITIVITY_ELEVATED,
    SENSITIVITY_NORMAL,
    SENSITIVITY_SENSITIVE,
)
from siqueira_memo.schemas import (
    ArtifactIngestIn,
    ArtifactIngestOut,
    BuiltinMemoryMirrorIn,
    DelegationObservationIn,
    GenericEventIn,
    GenericEventOut,
    HermesAuxCompactionIn,
    MessageIngestIn,
    MessageIngestOut,
    ToolEventIngestIn,
    ToolEventIngestOut,
)
from siqueira_memo.services.redaction_service import RedactionService
from siqueira_memo.utils.canonical import content_hash
from siqueira_memo.workers.queue import Job, JobQueue, get_default_queue

log = get_logger(__name__)


class IngestService:
    def __init__(
        self,
        *,
        queue: JobQueue | None = None,
        redaction: RedactionService | None = None,
        profile_id: str | None = None,
        actor: str = "hermes",
    ) -> None:
        self.queue = queue or get_default_queue()
        self.redaction = redaction or RedactionService()
        settings = get_settings()
        self.profile_id = profile_id or settings.derive_profile_id()
        self.actor = actor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_profile(self, override: str | None) -> str:
        return override or self.profile_id

    def _should_persist_durably(self, agent_context: str | None) -> bool:
        if agent_context is None:
            return True
        return agent_context in {AGENT_CONTEXT_PRIMARY, "", None}

    def _classify_sensitivity(self, base: str, matches: int) -> str:
        if matches <= 0:
            return base or SENSITIVITY_NORMAL
        if base == SENSITIVITY_SENSITIVE:
            return SENSITIVITY_SENSITIVE
        return SENSITIVITY_ELEVATED

    def _enqueue(self, job: Job) -> None:
        try:
            self.queue.enqueue(job)
        except Exception:  # pragma: no cover
            log.exception("ingest.queue_enqueue_failed", extra={"job": job.name})

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------
    async def ingest_message(
        self, session: AsyncSession, payload: MessageIngestIn
    ) -> MessageIngestOut:
        profile_id = self._resolve_profile(payload.profile_id)
        content_raw = payload.content
        redaction = self.redaction.redact(content_raw)
        c_hash = content_hash(content_raw)

        # Idempotency: dedupe per (profile, session, role, content_hash, source).
        existing = await session.execute(
            select(Message).where(
                Message.profile_id == profile_id,
                Message.session_id == payload.session_id,
                Message.role == payload.role,
                Message.content_hash == c_hash,
                Message.source == payload.source,
            )
        )
        duplicate = existing.scalar_one_or_none()
        if duplicate is not None:
            log.debug(
                "ingest.message.duplicate",
                extra={"message_id": str(duplicate.id), "profile_id": profile_id},
            )
            return MessageIngestOut(
                event_id=duplicate.event_id,
                message_id=duplicate.id,
                duplicate=True,
                redactions=0,
            )

        created_at = payload.created_at or datetime.now(UTC)
        event_type = (
            EVENT_TYPE_ASSISTANT_MESSAGE_SENT
            if payload.role == ROLE_ASSISTANT
            else EVENT_TYPE_MESSAGE_RECEIVED
        )

        message_id = uuid.uuid4()
        event = MemoryEvent(
            id=uuid.uuid4(),
            event_type=event_type,
            source=payload.source,
            actor=self.actor,
            session_id=payload.session_id,
            profile_id=profile_id,
            agent_context=payload.agent_context,
            payload={
                "event_type": event_type,
                "message_id": str(message_id),
                "role": payload.role,
                "platform": payload.platform,
                "content_hash": c_hash,
                "language": payload.language,
            },
            created_at=created_at,
        )
        session.add(event)

        sensitivity = self._classify_sensitivity(payload.sensitivity, redaction.matches)

        message = Message(
            id=message_id,
            event_id=event.id,
            profile_id=profile_id,
            session_id=payload.session_id,
            platform=payload.platform,
            chat_id=payload.chat_id,
            thread_id=payload.thread_id,
            role=payload.role,
            content_raw=content_raw,
            content_redacted=redaction.redacted,
            content_hash=c_hash,
            source=payload.source,
            platform_message_id=payload.platform_message_id,
            language=payload.language,
            project=payload.project,
            topic=payload.topic,
            entities=list(payload.entities),
            sensitivity=sensitivity,
            extra_metadata={
                **payload.metadata,
                "redaction_count": redaction.matches,
                "redaction_kinds": sorted({f.kind for f in redaction.findings}),
            },
            created_at=created_at,
        )
        session.add(message)
        await session.flush()

        log.info(
            "ingest.message",
            extra={
                "event_id": str(event.id),
                "message_id": str(message_id),
                "profile_id": profile_id,
                "session_id": payload.session_id,
                "redactions": redaction.matches,
                "role": payload.role,
            },
        )

        if self._should_persist_durably(payload.agent_context):
            # Chunking job: always, except for flush contexts.
            self._enqueue(
                Job(
                    name="chunk_message",
                    payload={
                        "profile_id": profile_id,
                        "message_id": str(message_id),
                    },
                    dedup_key=f"chunk:{message_id}",
                )
            )
            # Gate job decides whether expensive extraction runs.
            self._enqueue(
                Job(
                    name="extraction_gate",
                    payload={
                        "profile_id": profile_id,
                        "message_id": str(message_id),
                        "session_id": payload.session_id,
                    },
                    dedup_key=f"gate:{message_id}",
                )
            )

        return MessageIngestOut(
            event_id=event.id,
            message_id=message.id,
            duplicate=False,
            redactions=redaction.matches,
        )

    # ------------------------------------------------------------------
    # Tool events
    # ------------------------------------------------------------------
    async def ingest_tool_event(
        self, session: AsyncSession, payload: ToolEventIngestIn
    ) -> ToolEventIngestOut:
        profile_id = self._resolve_profile(payload.profile_id)
        input_redacted, in_count = self.redaction.redact_dict(dict(payload.input))
        output_raw = payload.output or ""
        out_redaction = self.redaction.redact(output_raw) if output_raw else None
        total_matches = in_count + (out_redaction.matches if out_redaction else 0)
        sensitivity = self._classify_sensitivity(payload.sensitivity, total_matches)

        event_id = uuid.uuid4()
        tool_event_id = uuid.uuid4()
        created_at = payload.created_at or datetime.now(UTC)

        event = MemoryEvent(
            id=event_id,
            event_type=EVENT_TYPE_TOOL_CALLED,
            source="live_tool",
            actor=self.actor,
            session_id=payload.session_id,
            profile_id=profile_id,
            agent_context=payload.agent_context,
            payload={
                "event_type": EVENT_TYPE_TOOL_CALLED,
                "tool_event_id": str(tool_event_id),
                "tool_name": payload.tool_name,
            },
            created_at=created_at,
        )
        session.add(event)

        tool_event = ToolEvent(
            id=tool_event_id,
            event_id=event_id,
            profile_id=profile_id,
            session_id=payload.session_id,
            tool_name=payload.tool_name,
            input_raw=dict(payload.input),
            input_redacted=input_redacted,
            output_raw=output_raw or None,
            output_redacted=out_redaction.redacted if out_redaction else None,
            output_summary=None,
            output_pointer=payload.output_pointer,
            output_hash=payload.output_hash,
            output_size_bytes=payload.output_size_bytes,
            exit_status=payload.exit_status,
            artifact_refs=list(payload.artifact_refs),
            sensitivity=sensitivity,
            extra_metadata={
                **payload.metadata,
                "redaction_count": total_matches,
            },
            created_at=created_at,
        )
        session.add(tool_event)
        await session.flush()

        log.info(
            "ingest.tool_event",
            extra={
                "event_id": str(event_id),
                "tool_event_id": str(tool_event_id),
                "tool_name": payload.tool_name,
                "redactions": total_matches,
                "sensitivity": sensitivity,
            },
        )

        if self._should_persist_durably(payload.agent_context):
            self._enqueue(
                Job(
                    name="chunk_tool_output",
                    payload={"profile_id": profile_id, "tool_event_id": str(tool_event_id)},
                    dedup_key=f"chunk_tool:{tool_event_id}",
                )
            )

        return ToolEventIngestOut(
            event_id=event_id,
            tool_event_id=tool_event_id,
            redactions=total_matches,
        )

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    async def ingest_artifact(
        self, session: AsyncSession, payload: ArtifactIngestIn
    ) -> ArtifactIngestOut:
        profile_id = self._resolve_profile(payload.profile_id)
        event_id = uuid.uuid4()
        artifact_id = uuid.uuid4()
        created_at = payload.created_at or datetime.now(UTC)
        event = MemoryEvent(
            id=event_id,
            event_type=EVENT_TYPE_ARTIFACT_CREATED,
            source="artifact",
            actor=self.actor,
            profile_id=profile_id,
            payload={
                "event_type": EVENT_TYPE_ARTIFACT_CREATED,
                "artifact_id": str(artifact_id),
                "path": payload.path,
                "uri": payload.uri,
                "content_hash": payload.content_hash,
            },
            created_at=created_at,
        )
        session.add(event)
        session.add(
            Artifact(
                id=artifact_id,
                event_id=event_id,
                profile_id=profile_id,
                type=payload.type,
                path=payload.path,
                uri=payload.uri,
                content_hash=payload.content_hash,
                summary=payload.summary,
                project=payload.project,
                extra_metadata=dict(payload.metadata),
                created_at=created_at,
            )
        )
        await session.flush()
        return ArtifactIngestOut(event_id=event_id, artifact_id=artifact_id)

    # ------------------------------------------------------------------
    # Generic, delegation, mirror, compaction
    # ------------------------------------------------------------------
    async def ingest_event(
        self, session: AsyncSession, payload: GenericEventIn
    ) -> GenericEventOut:
        profile_id = self._resolve_profile(payload.profile_id)
        event_id = uuid.uuid4()
        ev = MemoryEvent(
            id=event_id,
            event_type=payload.event_type,
            source=payload.source,
            actor=payload.actor,
            session_id=payload.session_id,
            profile_id=profile_id,
            agent_context=payload.agent_context,
            payload=dict(payload.payload),
        )
        session.add(ev)
        await session.flush()
        return GenericEventOut(event_id=event_id)

    async def ingest_delegation(
        self, session: AsyncSession, payload: DelegationObservationIn
    ) -> GenericEventOut:
        profile_id = self._resolve_profile(payload.profile_id)
        event_id = uuid.uuid4()
        session.add(
            MemoryEvent(
                id=event_id,
                event_type=EVENT_TYPE_DELEGATION_OBSERVED,
                source="hermes_delegation",
                actor=self.actor,
                session_id=payload.parent_session_id,
                profile_id=profile_id,
                agent_context=AGENT_CONTEXT_PRIMARY,
                payload={
                    "event_type": EVENT_TYPE_DELEGATION_OBSERVED,
                    "parent_session_id": payload.parent_session_id,
                    "child_session_id": payload.child_session_id,
                    "task": payload.task,
                    "result": payload.result,
                    "toolsets": list(payload.toolsets),
                    "model": payload.model,
                },
            )
        )
        await session.flush()
        return GenericEventOut(event_id=event_id)

    async def ingest_hermes_aux_compaction(
        self, session: AsyncSession, payload: HermesAuxCompactionIn
    ) -> GenericEventOut:
        profile_id = self._resolve_profile(payload.profile_id)
        redaction = self.redaction.redact(payload.summary_text)
        event_id = uuid.uuid4()
        session.add(
            MemoryEvent(
                id=event_id,
                event_type=EVENT_TYPE_HERMES_AUX_COMPACTION_OBSERVED,
                source="hermes_auxiliary_compaction",
                actor="hermes_compressor",
                session_id=payload.session_id,
                profile_id=profile_id,
                payload={
                    "event_type": EVENT_TYPE_HERMES_AUX_COMPACTION_OBSERVED,
                    "summary_text": redaction.redacted,
                    "prefix": payload.prefix,
                    "source_message_count": payload.source_message_count,
                },
            )
        )
        await session.flush()
        return GenericEventOut(event_id=event_id)

    async def ingest_builtin_memory_mirror(
        self, session: AsyncSession, payload: BuiltinMemoryMirrorIn
    ) -> GenericEventOut:
        profile_id = self._resolve_profile(payload.profile_id)
        content = payload.content or ""
        redaction = self.redaction.redact(content)
        c_hash = content_hash(content)
        event_id = uuid.uuid4()
        session.add(
            MemoryEvent(
                id=event_id,
                event_type=EVENT_TYPE_BUILTIN_MEMORY_MIRROR,
                source="hermes_builtin_memory_tool",
                actor="hermes",
                session_id=payload.session_id,
                profile_id=profile_id,
                payload={
                    "event_type": EVENT_TYPE_BUILTIN_MEMORY_MIRROR,
                    "action": payload.action,
                    "target": payload.target,
                    "content_hash": c_hash,
                    "content_redacted": redaction.redacted if content else None,
                    "selector": payload.selector,
                },
            )
        )
        await session.flush()
        return GenericEventOut(event_id=event_id)
