"""Relationship graph service. Roadmap Phase 4."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.models import (
    Decision,
    Entity,
    Fact,
    MemoryRelationship,
    Message,
    SessionSummary,
    TopicSummary,
)
from siqueira_memo.models.constants import ALL_RELATIONSHIP_TYPES, STATUS_ACTIVE

MemoryNodeType = Literal["fact", "decision", "entity", "summary", "message"]
Direction = Literal["incoming", "outgoing", "both"]

_NODE_TYPES: frozenset[str] = frozenset({"fact", "decision", "entity", "summary", "message"})
_MODEL_BY_NODE: dict[str, tuple[type[Any], ...]] = {
    "fact": (Fact,),
    "decision": (Decision,),
    "entity": (Entity,),
    "message": (Message,),
    "summary": (SessionSummary, TopicSummary),
}


@dataclass(frozen=True)
class RelatedMemory:
    target_type: str
    target_id: uuid.UUID
    relationship_type: str
    direction: str
    source_type: str
    source_id: uuid.UUID
    confidence: float
    rationale: str | None

    @property
    def explanation(self) -> str:
        base = (
            f"Returned through {self.relationship_type} relationship "
            f"{self.source_type}:{self.source_id} -> {self.target_type}:{self.target_id}"
        )
        if self.rationale:
            return f"{base}: {self.rationale}"
        return base


@dataclass
class RelationshipService:
    profile_id: str
    actor: str = "system"

    async def create(
        self,
        session: AsyncSession,
        *,
        source_type: str,
        source_id: uuid.UUID,
        relationship_type: str,
        target_type: str,
        target_id: uuid.UUID,
        confidence: float = 1.0,
        rationale: str | None = None,
        source_event_ids: list[uuid.UUID] | None = None,
        status: str = STATUS_ACTIVE,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRelationship:
        source_type = _normalize_type(source_type)
        target_type = _normalize_type(target_type)
        relationship_type = relationship_type.strip().lower()
        self._validate(source_type, target_type, relationship_type, source_id, target_id)
        await self._assert_profile_owned(session, source_type, source_id)
        await self._assert_profile_owned(session, target_type, target_id)
        bounded_confidence = min(1.0, max(0.0, float(confidence)))
        event_ids = _dedupe_uuids(source_event_ids or [])
        existing = await self._load_existing(
            session,
            source_type=source_type,
            source_id=source_id,
            relationship_type=relationship_type,
            target_type=target_type,
            target_id=target_id,
            status=status,
        )
        if existing is not None:
            existing.confidence = max(float(existing.confidence or 0.0), bounded_confidence)
            if rationale:
                existing.rationale = rationale
            existing.source_event_ids = _dedupe_uuids((existing.source_event_ids or []) + event_ids)
            if metadata:
                existing.extra_metadata = {**dict(existing.extra_metadata or {}), **metadata}
            await session.flush()
            return existing

        row = MemoryRelationship(
            id=uuid.uuid4(),
            profile_id=self.profile_id,
            source_type=source_type,
            source_id=source_id,
            relationship_type=relationship_type,
            target_type=target_type,
            target_id=target_id,
            confidence=bounded_confidence,
            rationale=rationale,
            source_event_ids=event_ids,
            created_by=self.actor,
            status=status,
            extra_metadata=dict(metadata or {}),
        )
        session.add(row)
        await session.flush()
        return row

    async def list_for_memory(
        self,
        session: AsyncSession,
        *,
        target_type: str,
        target_id: uuid.UUID,
        direction: Direction = "both",
        include_inactive: bool = False,
        limit: int = 100,
    ) -> list[MemoryRelationship]:
        target_type = _normalize_type(target_type)
        if target_type not in _NODE_TYPES:
            raise ValueError(f"unsupported memory node type: {target_type}")
        clauses = []
        if direction in {"outgoing", "both"}:
            clauses.append(
                and_(
                    MemoryRelationship.source_type == target_type,
                    MemoryRelationship.source_id == target_id,
                )
            )
        if direction in {"incoming", "both"}:
            clauses.append(
                and_(
                    MemoryRelationship.target_type == target_type,
                    MemoryRelationship.target_id == target_id,
                )
            )
        if not clauses:
            raise ValueError(f"unsupported relationship direction: {direction}")
        stmt = select(MemoryRelationship).where(MemoryRelationship.profile_id == self.profile_id)
        if not include_inactive:
            stmt = stmt.where(MemoryRelationship.status == STATUS_ACTIVE)
        stmt = stmt.where(or_(*clauses)).order_by(MemoryRelationship.created_at.desc()).limit(limit)
        return list((await session.execute(stmt)).scalars().all())

    async def expand_related(
        self,
        session: AsyncSession,
        seeds: list[tuple[str, uuid.UUID]],
        *,
        limit: int = 12,
    ) -> list[RelatedMemory]:
        if not seeds or limit <= 0:
            return []
        normalized = [(_normalize_type(t), i) for t, i in seeds if _normalize_type(t) in _NODE_TYPES]
        if not normalized:
            return []
        clauses = []
        for seed_type, seed_id in normalized:
            clauses.append(
                and_(
                    MemoryRelationship.source_type == seed_type,
                    MemoryRelationship.source_id == seed_id,
                )
            )
            clauses.append(
                and_(
                    MemoryRelationship.target_type == seed_type,
                    MemoryRelationship.target_id == seed_id,
                )
            )
        rows = list(
            (
                await session.execute(
                    select(MemoryRelationship)
                    .where(MemoryRelationship.profile_id == self.profile_id)
                    .where(MemoryRelationship.status == STATUS_ACTIVE)
                    .where(or_(*clauses))
                    .order_by(MemoryRelationship.confidence.desc(), MemoryRelationship.created_at.desc())
                    .limit(limit * 3)
                )
            )
            .scalars()
            .all()
        )
        seed_set = {(t, i) for t, i in normalized}
        out: list[RelatedMemory] = []
        seen: set[tuple[str, uuid.UUID]] = set()
        for row in rows:
            candidates: list[tuple[str, uuid.UUID, str]] = []
            if (row.source_type, row.source_id) in seed_set:
                candidates.append((row.target_type, row.target_id, "outgoing"))
            if (row.target_type, row.target_id) in seed_set:
                candidates.append((row.source_type, row.source_id, "incoming"))
            for related_type, related_id, direction in candidates:
                key = (related_type, related_id)
                if key in seen:
                    continue
                if related_type not in {"fact", "decision"}:
                    continue
                seen.add(key)
                out.append(
                    RelatedMemory(
                        target_type=related_type,
                        target_id=related_id,
                        relationship_type=row.relationship_type,
                        direction=direction,
                        source_type=row.source_type,
                        source_id=row.source_id,
                        confidence=float(row.confidence or 0.0),
                        rationale=row.rationale,
                    )
                )
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        return out

    async def _load_existing(
        self,
        session: AsyncSession,
        *,
        source_type: str,
        source_id: uuid.UUID,
        relationship_type: str,
        target_type: str,
        target_id: uuid.UUID,
        status: str,
    ) -> MemoryRelationship | None:
        return (
            await session.execute(
                select(MemoryRelationship).where(
                    MemoryRelationship.profile_id == self.profile_id,
                    MemoryRelationship.source_type == source_type,
                    MemoryRelationship.source_id == source_id,
                    MemoryRelationship.relationship_type == relationship_type,
                    MemoryRelationship.target_type == target_type,
                    MemoryRelationship.target_id == target_id,
                    MemoryRelationship.status == status,
                )
            )
        ).scalar_one_or_none()

    def _validate(
        self,
        source_type: str,
        target_type: str,
        relationship_type: str,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
    ) -> None:
        if source_type not in _NODE_TYPES:
            raise ValueError(f"unsupported relationship source_type: {source_type}")
        if target_type not in _NODE_TYPES:
            raise ValueError(f"unsupported relationship target_type: {target_type}")
        if relationship_type not in ALL_RELATIONSHIP_TYPES:
            raise ValueError(f"unsupported relationship_type: {relationship_type}")
        if source_type == target_type and source_id == target_id:
            raise ValueError("relationship cannot point to itself")

    async def _assert_profile_owned(
        self, session: AsyncSession, node_type: str, node_id: uuid.UUID
    ) -> None:
        models = _MODEL_BY_NODE[node_type]
        for model in models:
            row = (
                await session.execute(
                    select(model.id).where(model.id == node_id, model.profile_id == self.profile_id)
                )
            ).first()
            if row is not None:
                return
        raise ValueError(f"{node_type} {node_id} not found for profile")


def _normalize_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "memory":
        return "fact"
    return normalized


def _dedupe_uuids(values: list[uuid.UUID]) -> list[uuid.UUID]:
    return [uuid.UUID(value) for value in sorted({str(value) for value in values})]
