"""Entity linking / alias resolution skeleton. Plan §19.

Keeps three paths:

* ``link`` — deterministic alias match or fresh candidate creation.
* ``auto_merge`` — conservative merge when a candidate and an existing active
  entity share a normalised name/type and no conflicting metadata.
* ``needs_review`` — every ambiguous case.

Embedding-based fuzzy linking and LLM-assisted judgment are deliberately out
of scope for this skeleton; the interfaces are in place so workers can be
wired in later without schema churn.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.models import Entity, EntityAlias
from siqueira_memo.models.constants import (
    STATUS_ACTIVE,
    STATUS_CANDIDATE,
    STATUS_NEEDS_REVIEW,
)
from siqueira_memo.utils.canonical import normalize_text

LinkAction = Literal["link", "create_candidate", "auto_merge", "needs_review"]


@dataclass
class LinkResult:
    entity_id: uuid.UUID
    action: LinkAction
    confidence: float
    normalized_name: str


@dataclass
class EntityLinkingService:
    profile_id: str
    auto_merge_confidence: float = 0.95
    review_threshold: float = 0.75

    async def link_or_create(
        self,
        session: AsyncSession,
        *,
        mention: str,
        entity_type: str,
        description: str | None = None,
        source_event_id: uuid.UUID | None = None,
    ) -> LinkResult:
        normalized = normalize_text(mention)
        if not normalized:
            raise ValueError("empty mention")

        alias = (
            await session.execute(
                select(EntityAlias)
                .where(EntityAlias.profile_id == self.profile_id)
                .where(EntityAlias.alias_normalized == normalized)
                .where(EntityAlias.entity_type == entity_type)
                .where(EntityAlias.status == STATUS_ACTIVE)
                .limit(1)
            )
        ).scalar_one_or_none()
        if alias is not None:
            return LinkResult(
                entity_id=alias.entity_id,
                action="link",
                confidence=1.0,
                normalized_name=normalized,
            )

        existing = (
            await session.execute(
                select(Entity)
                .where(Entity.profile_id == self.profile_id)
                .where(Entity.name_normalized == normalized)
                .where(Entity.type == entity_type)
            )
        ).scalar_one_or_none()
        if existing is not None:
            # Same normalised name and type, no conflicting metadata → auto-merge alias.
            self._record_alias(session, existing.id, mention, normalized, entity_type, source_event_id)
            if existing.status != STATUS_ACTIVE:
                existing.status = STATUS_ACTIVE
            return LinkResult(
                entity_id=existing.id,
                action="auto_merge",
                confidence=self.auto_merge_confidence,
                normalized_name=normalized,
            )

        entity_id = uuid.uuid4()
        session.add(
            Entity(
                id=entity_id,
                profile_id=self.profile_id,
                name=mention,
                name_normalized=normalized,
                type=entity_type,
                aliases=[mention],
                description=description,
                status=STATUS_CANDIDATE,
            )
        )
        self._record_alias(session, entity_id, mention, normalized, entity_type, source_event_id)
        await session.flush()
        return LinkResult(
            entity_id=entity_id,
            action="create_candidate",
            confidence=0.6,
            normalized_name=normalized,
        )

    def _record_alias(
        self,
        session: AsyncSession,
        entity_id: uuid.UUID,
        alias: str,
        alias_normalized: str,
        entity_type: str,
        source_event_id: uuid.UUID | None,
    ) -> None:
        session.add(
            EntityAlias(
                id=uuid.uuid4(),
                entity_id=entity_id,
                profile_id=self.profile_id,
                alias=alias,
                alias_normalized=alias_normalized,
                entity_type=entity_type,
                status=STATUS_ACTIVE,
                source_event_ids=[source_event_id] if source_event_id else [],
            )
        )

    async def mark_needs_review(
        self, session: AsyncSession, entity_id: uuid.UUID
    ) -> None:
        row = (
            await session.execute(
                select(Entity).where(Entity.id == entity_id, Entity.profile_id == self.profile_id)
            )
        ).scalar_one_or_none()
        if row is None:
            return
        row.status = STATUS_NEEDS_REVIEW
