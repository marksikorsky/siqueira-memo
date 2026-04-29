"""Source-backed entity cards. Roadmap Phase 7.

This service deliberately does not infer personality, intent, or fuzzy truths.
It only assembles what existing entity rows, relationships, facts, and decisions
already support.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.models import Decision, Entity, EntityAlias, Fact, MemoryRelationship
from siqueira_memo.models.constants import (
    RELATIONSHIP_BELONGS_TO_ENTITY,
    RELATIONSHIP_CONTRADICTS,
    RELATIONSHIP_USES_SECRET,
    STATUS_ACTIVE,
    STATUS_SUPERSEDED,
)
from siqueira_memo.services.secret_policy import is_secret_metadata, masked_preview
from siqueira_memo.utils.canonical import normalize_text


@dataclass(frozen=True)
class EntityCardMemory:
    id: uuid.UUID
    kind: str
    text: str
    project: str | None
    topic: str | None
    status: str
    confidence: float
    source_event_ids: list[uuid.UUID] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class EntityCardRelationship:
    id: uuid.UUID
    source_type: str
    source_id: uuid.UUID
    relationship_type: str
    target_type: str
    target_id: uuid.UUID
    confidence: float
    rationale: str | None


@dataclass(frozen=True)
class EntityCard:
    entity_id: uuid.UUID
    name: str
    entity_type: str
    aliases: list[str]
    status: str
    description: str | None
    projects: list[str]
    topics: list[str]
    latest_facts: list[EntityCardMemory]
    active_decisions: list[EntityCardMemory]
    related_secrets: list[EntityCardMemory]
    relationships: list[EntityCardRelationship]
    source_count: int
    confidence: float
    last_updated: datetime | None
    conflict_count: int
    warnings: list[str]


@dataclass
class EntityCardService:
    profile_id: str

    async def build_card(
        self,
        session: AsyncSession,
        *,
        entity_id: uuid.UUID | None = None,
        name: str | None = None,
        entity_type: str | None = None,
        fact_limit: int = 8,
        decision_limit: int = 8,
        relationship_limit: int = 40,
    ) -> EntityCard:
        entity = await self._resolve_entity(
            session, entity_id=entity_id, name=name, entity_type=entity_type
        )
        if entity is None:
            raise ValueError("entity not found")

        aliases = await self._load_aliases(session, entity)
        direct_relationships = await self._load_entity_relationships(
            session, entity.id, limit=relationship_limit
        )
        related_facts = await self._load_related_facts(session, direct_relationships)
        related_decisions = await self._load_related_decisions(session, direct_relationships)
        secret_relationships = await self._load_secret_relationships_for_related_memories(
            session,
            entity.id,
            fact_ids={row.id for row in related_facts},
            decision_ids={row.id for row in related_decisions},
        )
        classification_relationships = _dedupe_relationships(
            [*direct_relationships, *secret_relationships]
        )
        conflict_relationships = await self._load_conflicts(
            session,
            fact_ids={row.id for row in related_facts},
            decision_ids={row.id for row in related_decisions},
            limit=relationship_limit,
        )

        secret_fact_ids = self._secret_target_ids(classification_relationships, kind="fact")
        secret_decision_ids = self._secret_target_ids(classification_relationships, kind="decision")
        fact_items = [
            self._fact_item(row)
            for row in related_facts
            if row.id not in secret_fact_ids and not self._fact_is_secret(row)
        ]
        decision_items = [
            self._decision_item(row)
            for row in related_decisions
            if row.id not in secret_decision_ids and not self._decision_is_secret(row)
        ]
        secret_items = [
            self._secret_fact_item(row, force_secret=row.id in secret_fact_ids)
            for row in related_facts
            if row.id in secret_fact_ids or self._fact_is_secret(row)
        ] + [
            self._secret_decision_item(row, force_secret=row.id in secret_decision_ids)
            for row in related_decisions
            if row.id in secret_decision_ids or self._decision_is_secret(row)
        ]

        active_facts = [item for item in fact_items if item.status == STATUS_ACTIVE][:fact_limit]
        active_decisions = [
            item for item in decision_items if item.status == STATUS_ACTIVE
        ][:decision_limit]
        projects = _sorted_unique(
            item.project for item in [*active_facts, *active_decisions] if item.project
        )
        topics = _sorted_unique(
            item.topic for item in [*active_facts, *active_decisions] if item.topic
        )
        source_count = len(
            {
                event_id
                for item in [*active_facts, *active_decisions, *secret_items]
                for event_id in item.source_event_ids
            }
        )
        graph_confidences = [
            float(row.confidence or 0.0)
            for row in direct_relationships
            if row.relationship_type == RELATIONSHIP_BELONGS_TO_ENTITY
        ]
        last_updated = _max_datetime(
            [
                entity.updated_at,
                *(row.updated_at for row in related_facts),
                *(row.updated_at for row in related_decisions),
            ]
        )
        warnings = self._warnings(
            related_facts=related_facts,
            related_decisions=related_decisions,
            conflict_count=len(conflict_relationships),
        )
        all_relationships = _dedupe_relationships([*direct_relationships, *conflict_relationships])[
            :relationship_limit
        ]
        return EntityCard(
            entity_id=entity.id,
            name=entity.name,
            entity_type=entity.type,
            aliases=aliases,
            status=entity.status,
            description=entity.description,
            projects=projects,
            topics=topics,
            latest_facts=active_facts,
            active_decisions=active_decisions,
            related_secrets=secret_items,
            relationships=[self._relationship_item(row) for row in all_relationships],
            source_count=source_count,
            confidence=_average(graph_confidences),
            last_updated=last_updated,
            conflict_count=len(conflict_relationships),
            warnings=warnings,
        )

    async def _resolve_entity(
        self,
        session: AsyncSession,
        *,
        entity_id: uuid.UUID | None,
        name: str | None,
        entity_type: str | None,
    ) -> Entity | None:
        if entity_id is not None:
            return (
                await session.execute(
                    select(Entity).where(Entity.id == entity_id, Entity.profile_id == self.profile_id)
                )
            ).scalar_one_or_none()
        normalized = normalize_text(name or "")
        if not normalized:
            return None
        stmt = select(EntityAlias).where(
            EntityAlias.profile_id == self.profile_id,
            EntityAlias.alias_normalized == normalized,
            EntityAlias.status == STATUS_ACTIVE,
        )
        if entity_type:
            stmt = stmt.where(EntityAlias.entity_type == entity_type)
        alias = (await session.execute(stmt.limit(1))).scalar_one_or_none()
        if alias is not None:
            return (
                await session.execute(
                    select(Entity).where(
                        Entity.id == alias.entity_id,
                        Entity.profile_id == self.profile_id,
                    )
                )
            ).scalar_one_or_none()
        entity_stmt = select(Entity).where(
            Entity.profile_id == self.profile_id,
            Entity.name_normalized == normalized,
        )
        if entity_type:
            entity_stmt = entity_stmt.where(Entity.type == entity_type)
        return (await session.execute(entity_stmt.limit(1))).scalar_one_or_none()

    async def _load_aliases(self, session: AsyncSession, entity: Entity) -> list[str]:
        rows = list(
            (
                await session.execute(
                    select(EntityAlias)
                    .where(
                        EntityAlias.profile_id == self.profile_id,
                        EntityAlias.entity_id == entity.id,
                        EntityAlias.status == STATUS_ACTIVE,
                    )
                    .order_by(EntityAlias.created_at.asc(), EntityAlias.alias.asc())
                )
            )
            .scalars()
            .all()
        )
        return _dedupe([entity.name, *(entity.aliases or []), *(row.alias for row in rows)])

    async def _load_entity_relationships(
        self, session: AsyncSession, entity_id: uuid.UUID, *, limit: int
    ) -> list[MemoryRelationship]:
        rows = (
            await session.execute(
                select(MemoryRelationship)
                .where(MemoryRelationship.profile_id == self.profile_id)
                .where(MemoryRelationship.status == STATUS_ACTIVE)
                .where(
                    or_(
                        and_(
                            MemoryRelationship.source_type == "entity",
                            MemoryRelationship.source_id == entity_id,
                        ),
                        and_(
                            MemoryRelationship.target_type == "entity",
                            MemoryRelationship.target_id == entity_id,
                        ),
                    )
                )
                .order_by(
                    MemoryRelationship.confidence.desc(), MemoryRelationship.created_at.desc()
                )
                .limit(limit)
            )
        ).scalars()
        return list(rows.all())

    async def _load_secret_relationships_for_related_memories(
        self,
        session: AsyncSession,
        entity_id: uuid.UUID,
        *,
        fact_ids: set[uuid.UUID],
        decision_ids: set[uuid.UUID],
    ) -> list[MemoryRelationship]:
        clauses = []
        if fact_ids:
            clauses.extend(
                [
                    and_(
                        MemoryRelationship.source_type == "fact",
                        MemoryRelationship.source_id.in_(fact_ids),
                    ),
                    and_(
                        MemoryRelationship.target_type == "fact",
                        MemoryRelationship.target_id.in_(fact_ids),
                    ),
                ]
            )
        if decision_ids:
            clauses.extend(
                [
                    and_(
                        MemoryRelationship.source_type == "decision",
                        MemoryRelationship.source_id.in_(decision_ids),
                    ),
                    and_(
                        MemoryRelationship.target_type == "decision",
                        MemoryRelationship.target_id.in_(decision_ids),
                    ),
                ]
            )
        if not clauses:
            return []
        rows = (
            await session.execute(
                select(MemoryRelationship)
                .where(MemoryRelationship.profile_id == self.profile_id)
                .where(MemoryRelationship.status == STATUS_ACTIVE)
                .where(MemoryRelationship.relationship_type == RELATIONSHIP_USES_SECRET)
                .where(or_(*clauses))
                .where(
                    or_(
                        and_(
                            MemoryRelationship.source_type == "entity",
                            MemoryRelationship.source_id == entity_id,
                        ),
                        and_(
                            MemoryRelationship.target_type == "entity",
                            MemoryRelationship.target_id == entity_id,
                        ),
                        MemoryRelationship.source_type.in_(["fact", "decision"]),
                        MemoryRelationship.target_type.in_(["fact", "decision"]),
                    )
                )
            )
        ).scalars()
        return list(rows.all())

    async def _load_related_facts(
        self, session: AsyncSession, relationships: list[MemoryRelationship]
    ) -> list[Fact]:
        ids = self._related_ids(relationships, kind="fact")
        if not ids:
            return []
        return list(
            (
                await session.execute(
                    select(Fact)
                    .where(Fact.profile_id == self.profile_id, Fact.id.in_(ids))
                    .order_by(Fact.updated_at.desc(), Fact.created_at.desc())
                )
            )
            .scalars()
            .all()
        )

    async def _load_related_decisions(
        self, session: AsyncSession, relationships: list[MemoryRelationship]
    ) -> list[Decision]:
        ids = self._related_ids(relationships, kind="decision")
        if not ids:
            return []
        return list(
            (
                await session.execute(
                    select(Decision)
                    .where(Decision.profile_id == self.profile_id, Decision.id.in_(ids))
                    .order_by(Decision.decided_at.desc(), Decision.updated_at.desc())
                )
            )
            .scalars()
            .all()
        )

    async def _load_conflicts(
        self,
        session: AsyncSession,
        *,
        fact_ids: set[uuid.UUID],
        decision_ids: set[uuid.UUID],
        limit: int,
    ) -> list[MemoryRelationship]:
        node_ids = fact_ids | decision_ids
        if not node_ids:
            return []
        rows = (
            await session.execute(
                select(MemoryRelationship)
                .where(MemoryRelationship.profile_id == self.profile_id)
                .where(MemoryRelationship.status == STATUS_ACTIVE)
                .where(MemoryRelationship.relationship_type == RELATIONSHIP_CONTRADICTS)
                .where(
                    or_(
                        MemoryRelationship.source_id.in_(node_ids),
                        MemoryRelationship.target_id.in_(node_ids),
                    )
                )
                .order_by(MemoryRelationship.created_at.desc())
                .limit(limit)
            )
        ).scalars()
        return list(rows.all())

    def _related_ids(
        self, relationships: list[MemoryRelationship], *, kind: str
    ) -> set[uuid.UUID]:
        ids: set[uuid.UUID] = set()
        for row in relationships:
            if row.source_type == kind:
                ids.add(row.source_id)
            if row.target_type == kind:
                ids.add(row.target_id)
        return ids

    def _secret_target_ids(
        self, relationships: list[MemoryRelationship], *, kind: str
    ) -> set[uuid.UUID]:
        ids: set[uuid.UUID] = set()
        for row in relationships:
            if row.relationship_type != RELATIONSHIP_USES_SECRET:
                continue
            if row.source_type == kind:
                ids.add(row.source_id)
            if row.target_type == kind:
                ids.add(row.target_id)
        return ids

    def _fact_item(self, row: Fact) -> EntityCardMemory:
        return EntityCardMemory(
            id=row.id,
            kind="fact",
            text=row.statement,
            project=row.project,
            topic=row.topic,
            status=row.status,
            confidence=float(row.confidence or 0.0),
            source_event_ids=list(row.source_event_ids or []),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def _decision_item(self, row: Decision) -> EntityCardMemory:
        return EntityCardMemory(
            id=row.id,
            kind="decision",
            text=row.decision,
            project=row.project,
            topic=row.topic,
            status=row.status,
            confidence=1.0,
            source_event_ids=list(row.source_event_ids or []),
            created_at=row.decided_at,
            updated_at=row.updated_at,
        )

    def _secret_fact_item(self, row: Fact, *, force_secret: bool = False) -> EntityCardMemory:
        item = self._fact_item(row)
        metadata = {**dict(row.extra_metadata or {})}
        if force_secret:
            metadata["sensitivity"] = "secret"
        return EntityCardMemory(**{**item.__dict__, "text": masked_preview(row.statement, metadata)})

    def _secret_decision_item(
        self, row: Decision, *, force_secret: bool = False
    ) -> EntityCardMemory:
        item = self._decision_item(row)
        metadata = {**dict(row.extra_metadata or {})}
        if force_secret:
            metadata["sensitivity"] = "secret"
        return EntityCardMemory(**{**item.__dict__, "text": masked_preview(row.decision, metadata)})

    def _relationship_item(self, row: MemoryRelationship) -> EntityCardRelationship:
        return EntityCardRelationship(
            id=row.id,
            source_type=row.source_type,
            source_id=row.source_id,
            relationship_type=row.relationship_type,
            target_type=row.target_type,
            target_id=row.target_id,
            confidence=float(row.confidence or 0.0),
            rationale=None,
        )

    def _fact_is_secret(self, row: Fact) -> bool:
        return is_secret_metadata(row.extra_metadata)

    def _decision_is_secret(self, row: Decision) -> bool:
        return is_secret_metadata(row.extra_metadata)

    def _warnings(
        self,
        *,
        related_facts: list[Fact],
        related_decisions: list[Decision],
        conflict_count: int,
    ) -> list[str]:
        warnings: list[str] = []
        if conflict_count:
            warnings.append(f"{conflict_count} conflict relationship(s) touch this entity card")
        related_rows: list[Fact | Decision] = [*related_facts, *related_decisions]
        stale_count = len(
            [row for row in related_rows if row.status != STATUS_ACTIVE or row.superseded_by is not None]
        )
        if stale_count:
            status_hint = (
                "superseded/stale"
                if any(row.status == STATUS_SUPERSEDED for row in related_rows)
                else "stale"
            )
            warnings.append(f"{stale_count} {status_hint} related memory item(s) omitted")
        return warnings


def _dedupe_relationships(rows: list[MemoryRelationship]) -> list[MemoryRelationship]:
    out: list[MemoryRelationship] = []
    seen: set[uuid.UUID] = set()
    for row in rows:
        if row.id in seen:
            continue
        seen.add(row.id)
        out.append(row)
    return out


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = normalize_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _sorted_unique(values: Iterable[str | None]) -> list[str]:
    return sorted({str(value) for value in values if value})


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _max_datetime(values: list[datetime | None]) -> datetime | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None
