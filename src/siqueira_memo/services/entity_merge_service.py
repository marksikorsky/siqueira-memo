"""Source-backed entity merge suggestions and admin review actions."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal, TypeVar

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.models import (
    Entity,
    EntityAlias,
    EntityRelationship,
    MemoryEvent,
    MemoryRelationship,
)
from siqueira_memo.models.constants import (
    STATUS_ACTIVE,
    STATUS_CANDIDATE,
    STATUS_MERGED,
    STATUS_NEEDS_REVIEW,
)
from siqueira_memo.utils.canonical import normalize_text

ReviewAction = Literal["merge", "reject"]
_T = TypeVar("_T", str, uuid.UUID)

_REVIEWABLE_STATUSES = {STATUS_ACTIVE, STATUS_CANDIDATE, STATUS_NEEDS_REVIEW}
_TOKEN_SPLIT = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True)
class EntityMergeCandidate:
    entity_id: uuid.UUID
    name: str
    entity_type: str
    aliases: list[str]
    status: str


@dataclass(frozen=True)
class EntityMergeSuggestion:
    source: EntityMergeCandidate
    target: EntityMergeCandidate
    confidence: float
    reasons: list[str]
    evidence: list[str]
    shared_tokens: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EntityMergeReviewResult:
    action: ReviewAction
    source_entity_id: uuid.UUID
    target_entity_id: uuid.UUID
    status: str
    moved_aliases: int = 0
    moved_relationships: int = 0
    event_id: uuid.UUID | None = None


@dataclass
class EntityMergeService:
    profile_id: str
    actor: str = "system"

    async def suggest(
        self,
        session: AsyncSession,
        *,
        entity_type: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> list[EntityMergeSuggestion]:
        stmt = select(Entity).where(Entity.profile_id == self.profile_id)
        stmt = stmt.where(Entity.status.in_(_REVIEWABLE_STATUSES))
        if entity_type:
            stmt = stmt.where(Entity.type == entity_type)
        if query:
            pattern = f"%{query}%"
            stmt = stmt.where(or_(Entity.name.ilike(pattern), Entity.name_normalized.ilike(pattern)))
        rows = list((await session.execute(stmt.order_by(Entity.type.asc(), Entity.name.asc()))).scalars().all())
        suggestions: list[EntityMergeSuggestion] = []
        for left, right in combinations(rows, 2):
            if left.type != right.type:
                continue
            if self._is_rejected_pair(left, right):
                continue
            suggestion = self._build_suggestion(left, right)
            if suggestion is not None:
                suggestions.append(suggestion)
        suggestions.sort(key=lambda item: (-item.confidence, item.target.name, item.source.name))
        return suggestions[:limit]

    async def merge(
        self,
        session: AsyncSession,
        *,
        source_entity_id: uuid.UUID,
        target_entity_id: uuid.UUID,
        reason: str | None = None,
    ) -> EntityMergeReviewResult:
        source, target = await self._load_pair(session, source_entity_id, target_entity_id)
        if source.type != target.type:
            raise ValueError("entity types differ")
        if source.status == STATUS_MERGED:
            raise ValueError("source entity is already merged")
        if target.status == STATUS_MERGED:
            raise ValueError("target entity is already merged")

        moved_aliases = await self._move_aliases(session, source, target)
        moved_relationships = await self._move_memory_relationships(session, source.id, target.id)
        moved_relationships += await self._move_entity_relationships(session, source.id, target.id)

        target.aliases = _dedupe([*(target.aliases or []), source.name, *(source.aliases or [])])
        source.status = STATUS_MERGED
        source.merged_into = target.id
        source.extra_metadata = {
            **dict(source.extra_metadata or {}),
            "merged_into": str(target.id),
            "merge_reason": reason,
        }
        event_id = self._record_event(
            session,
            "entity_merge_applied",
            source=source,
            target=target,
            reason=reason,
            moved_aliases=moved_aliases,
            moved_relationships=moved_relationships,
        )
        await session.flush()
        return EntityMergeReviewResult(
            action="merge",
            source_entity_id=source.id,
            target_entity_id=target.id,
            status=STATUS_MERGED,
            moved_aliases=moved_aliases,
            moved_relationships=moved_relationships,
            event_id=event_id,
        )

    async def reject(
        self,
        session: AsyncSession,
        *,
        source_entity_id: uuid.UUID,
        target_entity_id: uuid.UUID,
        reason: str | None = None,
    ) -> EntityMergeReviewResult:
        source, target = await self._load_pair(session, source_entity_id, target_entity_id)
        pair_key = _pair_key(source.id, target.id)
        for entity in (source, target):
            metadata = dict(entity.extra_metadata or {})
            rejections = list(metadata.get("merge_rejections") or [])
            if pair_key not in rejections:
                rejections.append(pair_key)
            metadata["merge_rejections"] = rejections
            if reason:
                metadata["last_merge_rejection_reason"] = reason
            entity.extra_metadata = metadata
        event_id = self._record_event(
            session,
            "entity_merge_rejected",
            source=source,
            target=target,
            reason=reason,
            moved_aliases=0,
            moved_relationships=0,
        )
        await session.flush()
        return EntityMergeReviewResult(
            action="reject",
            source_entity_id=source.id,
            target_entity_id=target.id,
            status="rejected",
            event_id=event_id,
        )

    async def _load_pair(
        self, session: AsyncSession, source_entity_id: uuid.UUID, target_entity_id: uuid.UUID
    ) -> tuple[Entity, Entity]:
        if source_entity_id == target_entity_id:
            raise ValueError("source and target entity must differ")
        rows = list(
            (
                await session.execute(
                    select(Entity)
                    .where(Entity.profile_id == self.profile_id)
                    .where(Entity.id.in_([source_entity_id, target_entity_id]))
                )
            )
            .scalars()
            .all()
        )
        by_id = {row.id: row for row in rows}
        source = by_id.get(source_entity_id)
        target = by_id.get(target_entity_id)
        if source is None or target is None:
            raise LookupError("entity pair not found")
        return source, target

    def _build_suggestion(self, left: Entity, right: Entity) -> EntityMergeSuggestion | None:
        left_names = _names(left)
        right_names = _names(right)
        left_compact = {_compact(value) for value in left_names if _compact(value)}
        right_compact = {_compact(value) for value in right_names if _compact(value)}
        shared_compact = sorted(left_compact & right_compact)
        left_tokens = _tokens(left_names)
        right_tokens = _tokens(right_names)
        shared_tokens = sorted((left_tokens & right_tokens) - {"api", "app", "service", "server"})
        reasons: list[str] = []
        evidence: list[str] = []
        confidence = 0.0
        if shared_compact:
            reasons.append("compact_name_match")
            evidence.append(f"compact normalized match: {shared_compact[0]}")
            confidence = max(confidence, 0.95)
        if shared_tokens and min(len(left_tokens), len(right_tokens)) > 0:
            overlap = len(shared_tokens) / max(len(left_tokens | right_tokens), 1)
            if overlap >= 0.34:
                reasons.append("shared_distinctive_tokens")
                evidence.append("shared tokens: " + ", ".join(shared_tokens[:6]))
                confidence = max(confidence, min(0.9, 0.65 + overlap))
        if not reasons or confidence < 0.75:
            return None
        target, source = _choose_target_source(left, right)
        return EntityMergeSuggestion(
            source=_candidate(source),
            target=_candidate(target),
            confidence=round(confidence, 3),
            reasons=reasons,
            evidence=evidence,
            shared_tokens=shared_tokens,
        )

    def _is_rejected_pair(self, left: Entity, right: Entity) -> bool:
        pair_key = _pair_key(left.id, right.id)
        left_rejections = set((left.extra_metadata or {}).get("merge_rejections") or [])
        right_rejections = set((right.extra_metadata or {}).get("merge_rejections") or [])
        return pair_key in left_rejections or pair_key in right_rejections

    async def _move_aliases(self, session: AsyncSession, source: Entity, target: Entity) -> int:
        source_aliases = list(
            (
                await session.execute(
                    select(EntityAlias)
                    .where(EntityAlias.profile_id == self.profile_id)
                    .where(EntityAlias.entity_id == source.id)
                )
            )
            .scalars()
            .all()
        )
        moved = 0
        for alias in source_aliases:
            existing = (
                await session.execute(
                    select(EntityAlias)
                    .where(EntityAlias.profile_id == self.profile_id)
                    .where(EntityAlias.alias_normalized == alias.alias_normalized)
                    .where(EntityAlias.entity_type == alias.entity_type)
                    .where(EntityAlias.entity_id != source.id)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if existing is None:
                alias.entity_id = target.id
                moved += 1
            elif existing.entity_id == target.id:
                alias.status = STATUS_MERGED
        for alias_text in _dedupe([source.name, *(source.aliases or [])]):
            alias_normalized = normalize_text(alias_text)
            if not alias_normalized:
                continue
            existing = (
                await session.execute(
                    select(EntityAlias)
                    .where(EntityAlias.profile_id == self.profile_id)
                    .where(EntityAlias.alias_normalized == alias_normalized)
                    .where(EntityAlias.entity_type == source.type)
                    .where(EntityAlias.status == STATUS_ACTIVE)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if existing is None:
                session.add(
                    EntityAlias(
                        id=uuid.uuid4(),
                        entity_id=target.id,
                        profile_id=self.profile_id,
                        alias=alias_text,
                        alias_normalized=alias_normalized,
                        entity_type=source.type,
                        status=STATUS_ACTIVE,
                    )
                )
                moved += 1
        return moved

    async def _move_memory_relationships(
        self, session: AsyncSession, source_id: uuid.UUID, target_id: uuid.UUID
    ) -> int:
        rows = list(
            (
                await session.execute(
                    select(MemoryRelationship)
                    .where(MemoryRelationship.profile_id == self.profile_id)
                    .where(MemoryRelationship.status == STATUS_ACTIVE)
                    .where(
                        or_(
                            and_(
                                MemoryRelationship.source_type == "entity",
                                MemoryRelationship.source_id == source_id,
                            ),
                            and_(
                                MemoryRelationship.target_type == "entity",
                                MemoryRelationship.target_id == source_id,
                            ),
                        )
                    )
                )
            )
            .scalars()
            .all()
        )
        moved = 0
        for row in rows:
            new_source_id = (
                target_id
                if row.source_type == "entity" and row.source_id == source_id
                else row.source_id
            )
            new_target_id = (
                target_id
                if row.target_type == "entity" and row.target_id == source_id
                else row.target_id
            )
            if row.source_type == "entity" and row.target_type == "entity" and new_source_id == new_target_id:
                row.status = STATUS_MERGED
                moved += 1
                continue
            existing = await self._existing_relationship(session, row, new_source_id, new_target_id)
            if existing is not None and existing.id != row.id:
                existing.confidence = max(float(existing.confidence or 0.0), float(row.confidence or 0.0))
                existing.source_event_ids = _dedupe([*existing.source_event_ids, *row.source_event_ids])
                existing.extra_metadata = {
                    **dict(existing.extra_metadata or {}),
                    **dict(row.extra_metadata or {}),
                }
                row.status = STATUS_MERGED
            else:
                row.source_id = new_source_id
                row.target_id = new_target_id
            moved += 1
        return moved

    async def _existing_relationship(
        self,
        session: AsyncSession,
        row: MemoryRelationship,
        new_source_id: uuid.UUID,
        new_target_id: uuid.UUID,
    ) -> MemoryRelationship | None:
        return (
            await session.execute(
                select(MemoryRelationship).where(
                    MemoryRelationship.profile_id == self.profile_id,
                    MemoryRelationship.source_type == row.source_type,
                    MemoryRelationship.source_id == new_source_id,
                    MemoryRelationship.relationship_type == row.relationship_type,
                    MemoryRelationship.target_type == row.target_type,
                    MemoryRelationship.target_id == new_target_id,
                    MemoryRelationship.status == row.status,
                )
            )
        ).scalar_one_or_none()

    async def _move_entity_relationships(
        self, session: AsyncSession, source_id: uuid.UUID, target_id: uuid.UUID
    ) -> int:
        rows = list(
            (
                await session.execute(
                    select(EntityRelationship)
                    .where(EntityRelationship.profile_id == self.profile_id)
                    .where(EntityRelationship.status == STATUS_ACTIVE)
                    .where(
                        or_(
                            EntityRelationship.source_entity_id == source_id,
                            EntityRelationship.target_entity_id == source_id,
                        )
                    )
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            new_source_id = target_id if row.source_entity_id == source_id else row.source_entity_id
            new_target_id = target_id if row.target_entity_id == source_id else row.target_entity_id
            if new_source_id == new_target_id:
                row.status = STATUS_MERGED
                continue
            row.source_entity_id = new_source_id
            row.target_entity_id = new_target_id
        return len(rows)

    def _record_event(
        self,
        session: AsyncSession,
        event_type: str,
        *,
        source: Entity,
        target: Entity,
        reason: str | None,
        moved_aliases: int,
        moved_relationships: int,
    ) -> uuid.UUID:
        event_id = uuid.uuid4()
        session.add(
            MemoryEvent(
                id=event_id,
                event_type=event_type,
                source="admin_api",
                actor=self.actor,
                session_id=None,
                profile_id=self.profile_id,
                payload={
                    "source_entity_id": str(source.id),
                    "source_name": source.name,
                    "target_entity_id": str(target.id),
                    "target_name": target.name,
                    "reason": reason,
                    "moved_aliases": moved_aliases,
                    "moved_relationships": moved_relationships,
                },
            )
        )
        return event_id


def _names(entity: Entity) -> list[str]:
    return _dedupe([entity.name, entity.name_normalized, *(entity.aliases or [])])


def _candidate(entity: Entity) -> EntityMergeCandidate:
    return EntityMergeCandidate(
        entity_id=entity.id,
        name=entity.name,
        entity_type=entity.type,
        aliases=_dedupe([entity.name, *(entity.aliases or [])]),
        status=entity.status,
    )


def _choose_target_source(left: Entity, right: Entity) -> tuple[Entity, Entity]:
    status_rank = {STATUS_ACTIVE: 0, STATUS_NEEDS_REVIEW: 1, STATUS_CANDIDATE: 2}
    ordered = sorted(
        [left, right],
        key=lambda row: (status_rank.get(row.status, 9), len(row.aliases or []), row.name),
    )
    return ordered[0], ordered[1]


def _compact(value: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]+", "", normalize_text(value))


def _tokens(values: list[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for token in _TOKEN_SPLIT.split(normalize_text(value).replace("-", " ")):
            if len(token) >= 3:
                tokens.add(token)
    return tokens


def _pair_key(left: uuid.UUID, right: uuid.UUID) -> str:
    return ":".join(sorted([str(left), str(right)]))


def _dedupe(values: list[_T]) -> list[_T]:
    seen: set[str] = set()
    result: list[_T] = []
    for value in values:
        key = str(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
