"""Structured extraction orchestrator + manual remember/correct entrypoints.

Full LLM-driven extraction belongs to worker jobs and is not performed inline
on the hot path. This module provides:

* ``ExtractionService.remember_fact`` / ``remember_decision`` — manual
  promotion path used by ``POST /v1/memory/remember`` and the
  ``siqueira_memory_remember`` tool. Enforces the candidate → active lifecycle
  and canonical-key dedupe described in plan §31.7.
* ``ExtractionService.apply_correction`` — handles explicit user corrections:
  invalidates/supersedes prior memories, writes a ``user_correction_received``
  event, and optionally creates a replacement fact/decision.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.logging import get_logger
from siqueira_memo.models import Decision, DecisionSource, Fact, FactSource, MemoryEvent
from siqueira_memo.models.constants import (
    EVENT_TYPE_DECISION_RECORDED,
    EVENT_TYPE_DECISION_SUPERSEDED,
    EVENT_TYPE_FACT_EXTRACTED,
    EVENT_TYPE_FACT_INVALIDATED,
    EVENT_TYPE_USER_CORRECTION,
    STATUS_ACTIVE,
    STATUS_INVALIDATED,
    STATUS_SUPERSEDED,
)
from siqueira_memo.schemas.memory import (
    CorrectRequest,
    CorrectResponse,
    RememberRequest,
    RememberResponse,
)
from siqueira_memo.utils.canonical import decision_canonical_key, fact_canonical_key

log = get_logger(__name__)


@dataclass
class ExtractionService:
    profile_id: str
    actor: str = "user"
    fact_prompt_version: str = "v1"
    decision_prompt_version: str = "v1"
    model_provider: str = "manual"
    model_name: str = "manual"
    schema_version: str = "v1"

    # ------------------------------------------------------------------
    # Remember
    # ------------------------------------------------------------------
    async def remember(
        self, session: AsyncSession, request: RememberRequest
    ) -> RememberResponse:
        profile_id = request.profile_id or self.profile_id
        if request.kind == "fact":
            return await self._remember_fact(session, request, profile_id)
        return await self._remember_decision(session, request, profile_id)

    async def _remember_fact(
        self, session: AsyncSession, request: RememberRequest, profile_id: str
    ) -> RememberResponse:
        subject = request.subject or ""
        predicate = request.predicate or ""
        obj = request.object or ""
        statement = request.statement or f"{subject} {predicate} {obj}".strip()
        if not subject or not predicate or not obj:
            raise ValueError("fact remember requires subject/predicate/object")

        key = fact_canonical_key(
            subject, predicate, obj, project=request.project, profile_id=profile_id
        )
        existing_active = (
            await session.execute(
                select(Fact)
                .where(Fact.profile_id == profile_id)
                .where(Fact.canonical_key == key)
                .where(Fact.status == STATUS_ACTIVE)
            )
        ).scalar_one_or_none()

        superseded: list[uuid.UUID] = []
        if existing_active is not None:
            # Merge sources; do not create duplicate active.
            merged_events = sorted(
                {
                    str(x)
                    for x in (existing_active.source_event_ids or [])
                    + request.source_event_ids
                }
            )
            existing_active.source_event_ids = [uuid.UUID(x) for x in merged_events]
            existing_active.confidence = max(existing_active.confidence, request.confidence)
            existing_active.updated_at = datetime.now(UTC)
            fact_id = existing_active.id
            fact_row = existing_active
            status = STATUS_ACTIVE
        else:
            fact_id = uuid.uuid4()
            fact_row = Fact(
                id=fact_id,
                profile_id=profile_id,
                subject=subject,
                predicate=predicate,
                object=obj,
                statement=statement,
                canonical_key=key,
                project=request.project,
                topic=request.topic,
                confidence=request.confidence,
                status=STATUS_ACTIVE,
                valid_from=request.valid_from,
                valid_to=request.valid_to,
                source_event_ids=list(request.source_event_ids),
                source_message_ids=list(request.source_message_ids),
                extractor_name="manual",
                extractor_version="1",
                prompt_version=self.fact_prompt_version,
                model_provider=self.model_provider,
                model_name=self.model_name,
                source_scope="message",
                schema_version=self.schema_version,
                extra_metadata={
                    **dict(request.metadata),
                    "confidence": request.confidence,
                },
            )
            session.add(fact_row)
            status = STATUS_ACTIVE

        event_id = uuid.uuid4()
        fact_row.source_event_ids = [
            uuid.UUID(x)
            for x in sorted({str(x) for x in (fact_row.source_event_ids or [])} | {str(event_id)})
        ]
        session.add(
            MemoryEvent(
                id=event_id,
                event_type=EVENT_TYPE_FACT_EXTRACTED,
                source="remember_api",
                actor=self.actor,
                session_id=request.session_id,
                profile_id=profile_id,
                payload={
                    "event_type": EVENT_TYPE_FACT_EXTRACTED,
                    "fact_id": str(fact_id),
                    "canonical_key": key,
                    "status": status,
                    "confidence": request.confidence,
                },
            )
        )
        # Flush the event before inserting provenance join rows. Postgres enforces
        # the FK immediately, and without an explicit flush SQLAlchemy can choose
        # an insert order that writes fact_sources/decision_sources first.
        await session.flush()
        # Link sources via the join table.
        for source_event_id in request.source_event_ids:
            session.add(
                FactSource(
                    fact_id=fact_id,
                    event_id=source_event_id,
                    message_id=(request.source_message_ids[0] if request.source_message_ids else None),
                )
            )
        session.add(
            FactSource(fact_id=fact_id, event_id=event_id, message_id=None)
        )
        await session.flush()

        log.info(
            "extraction.remember.fact",
            extra={
                "fact_id": str(fact_id),
                "canonical_key": key,
                "profile_id": profile_id,
            },
        )

        return RememberResponse(
            id=fact_id,
            kind="fact",
            status=status,
            canonical_key=key,
            event_id=event_id,
            superseded=superseded,
        )

    async def _remember_decision(
        self, session: AsyncSession, request: RememberRequest, profile_id: str
    ) -> RememberResponse:
        statement = request.statement or ""
        if not request.topic or not statement:
            raise ValueError("decision remember requires topic and statement")
        key = decision_canonical_key(
            request.project, request.topic, statement, profile_id=profile_id
        )
        existing_active = (
            await session.execute(
                select(Decision)
                .where(Decision.profile_id == profile_id)
                .where(Decision.canonical_key == key)
                .where(Decision.status == STATUS_ACTIVE)
            )
        ).scalar_one_or_none()
        superseded: list[uuid.UUID] = []
        decided_at = datetime.now(UTC)
        if existing_active is not None:
            merged_events = sorted(
                {
                    str(x)
                    for x in (existing_active.source_event_ids or [])
                    + request.source_event_ids
                }
            )
            existing_active.source_event_ids = [uuid.UUID(x) for x in merged_events]
            existing_active.updated_at = datetime.now(UTC)
            decision_id = existing_active.id
            decision_row = existing_active
            status = STATUS_ACTIVE
        else:
            decision_id = uuid.uuid4()
            decision_row = Decision(
                id=decision_id,
                profile_id=profile_id,
                project=request.project,
                topic=request.topic,
                decision=statement,
                context="manual remember",
                options_considered=list(request.options_considered),
                rationale=request.rationale or "",
                tradeoffs=dict(request.tradeoffs),
                canonical_key=key,
                status=STATUS_ACTIVE,
                reversible=request.reversible,
                decided_at=decided_at,
                source_event_ids=list(request.source_event_ids),
                source_message_ids=list(request.source_message_ids),
                extractor_name="manual",
                extractor_version="1",
                prompt_version=self.decision_prompt_version,
                model_provider=self.model_provider,
                model_name=self.model_name,
                source_scope="message",
                schema_version=self.schema_version,
                extra_metadata={
                    **dict(request.metadata),
                    "confidence": request.confidence,
                },
            )
            session.add(decision_row)
            status = STATUS_ACTIVE

        event_id = uuid.uuid4()
        decision_row.source_event_ids = [
            uuid.UUID(x)
            for x in sorted(
                {str(x) for x in (decision_row.source_event_ids or [])} | {str(event_id)}
            )
        ]
        session.add(
            MemoryEvent(
                id=event_id,
                event_type=EVENT_TYPE_DECISION_RECORDED,
                source="remember_api",
                actor=self.actor,
                session_id=request.session_id,
                profile_id=profile_id,
                payload={
                    "event_type": EVENT_TYPE_DECISION_RECORDED,
                    "decision_id": str(decision_id),
                    "canonical_key": key,
                    "status": status,
                },
            )
        )
        # Same ordering guard as facts: make the generated MemoryEvent visible
        # before decision_sources references it.
        await session.flush()
        for source_event_id in request.source_event_ids:
            session.add(
                DecisionSource(
                    decision_id=decision_id,
                    event_id=source_event_id,
                    message_id=(
                        request.source_message_ids[0]
                        if request.source_message_ids
                        else None
                    ),
                )
            )
        session.add(
            DecisionSource(decision_id=decision_id, event_id=event_id, message_id=None)
        )
        await session.flush()

        log.info(
            "extraction.remember.decision",
            extra={
                "decision_id": str(decision_id),
                "canonical_key": key,
                "profile_id": profile_id,
            },
        )

        return RememberResponse(
            id=decision_id,
            kind="decision",
            status=status,
            canonical_key=key,
            event_id=event_id,
            superseded=superseded,
        )

    # ------------------------------------------------------------------
    # Correct
    # ------------------------------------------------------------------
    async def apply_correction(
        self, session: AsyncSession, request: CorrectRequest
    ) -> CorrectResponse:
        profile_id = request.profile_id or self.profile_id
        event_id = uuid.uuid4()
        invalidated: list[uuid.UUID] = []
        superseded: list[uuid.UUID] = []
        replacement_id: uuid.UUID | None = None

        session.add(
            MemoryEvent(
                id=event_id,
                event_type=EVENT_TYPE_USER_CORRECTION,
                source="correct_api",
                actor=self.actor,
                session_id=request.session_id,
                profile_id=profile_id,
                payload={
                    "event_type": EVENT_TYPE_USER_CORRECTION,
                    "correction_text": request.correction_text,
                    "target_type": request.target_type,
                    "target_id": str(request.target_id) if request.target_id else None,
                },
            )
        )

        if request.target_type == "fact" and request.target_id is not None:
            fact = (
                await session.execute(
                    select(Fact).where(
                        Fact.id == request.target_id, Fact.profile_id == profile_id
                    )
                )
            ).scalar_one_or_none()
            if fact is not None:
                fact.status = (
                    STATUS_SUPERSEDED if request.replacement else STATUS_INVALIDATED
                )
                invalidated.append(fact.id)
                session.add(
                    MemoryEvent(
                        id=uuid.uuid4(),
                        event_type=EVENT_TYPE_FACT_INVALIDATED,
                        source="correct_api",
                        actor=self.actor,
                        profile_id=profile_id,
                        session_id=request.session_id,
                        payload={
                            "event_type": EVENT_TYPE_FACT_INVALIDATED,
                            "fact_id": str(fact.id),
                            "reason": request.correction_text,
                        },
                    )
                )

        elif request.target_type == "decision" and request.target_id is not None:
            decision = (
                await session.execute(
                    select(Decision).where(
                        Decision.id == request.target_id, Decision.profile_id == profile_id
                    )
                )
            ).scalar_one_or_none()
            if decision is not None:
                decision.status = (
                    STATUS_SUPERSEDED if request.replacement else STATUS_INVALIDATED
                )
                superseded.append(decision.id)
                session.add(
                    MemoryEvent(
                        id=uuid.uuid4(),
                        event_type=EVENT_TYPE_DECISION_SUPERSEDED,
                        source="correct_api",
                        actor=self.actor,
                        profile_id=profile_id,
                        session_id=request.session_id,
                        payload={
                            "event_type": EVENT_TYPE_DECISION_SUPERSEDED,
                            "decision_id": str(decision.id),
                            "reason": request.correction_text,
                        },
                    )
                )

        if request.replacement is not None:
            replacement_request = request.replacement.model_copy(
                update={
                    "profile_id": profile_id,
                    "session_id": request.session_id,
                    "source_event_ids": list(request.replacement.source_event_ids) + [event_id],
                }
            )
            result = await self.remember(session, replacement_request)
            replacement_id = result.id
            # Link superseded -> replacement.
            if request.target_type == "fact" and invalidated:
                fact_row = (
                    await session.execute(
                        select(Fact).where(Fact.id == invalidated[0])
                    )
                ).scalar_one_or_none()
                if fact_row is not None:
                    fact_row.superseded_by = replacement_id
            if request.target_type == "decision" and superseded:
                decision_row = (
                    await session.execute(
                        select(Decision).where(Decision.id == superseded[0])
                    )
                ).scalar_one_or_none()
                if decision_row is not None:
                    decision_row.superseded_by = replacement_id

        await session.flush()

        log.info(
            "extraction.correction_applied",
            extra={
                "event_id": str(event_id),
                "profile_id": profile_id,
                "target_type": request.target_type,
                "invalidated": [str(x) for x in invalidated],
                "superseded": [str(x) for x in superseded],
                "replacement_id": str(replacement_id) if replacement_id else None,
            },
        )
        return CorrectResponse(
            event_id=event_id,
            invalidated=invalidated,
            superseded=superseded,
            replacement_id=replacement_id,
        )
