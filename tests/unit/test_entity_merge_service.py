"""Entity merge suggestion/review service tests."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from siqueira_memo.models import (
    Entity,
    EntityAlias,
    EntityRelationship,
    Fact,
    MemoryEvent,
    MemoryRelationship,
)
from siqueira_memo.models.constants import (
    RELATIONSHIP_BELONGS_TO_ENTITY,
    STATUS_ACTIVE,
    STATUS_CANDIDATE,
    STATUS_MERGED,
)
from siqueira_memo.services.entity_merge_service import EntityMergeService
from siqueira_memo.utils.canonical import fact_canonical_key, normalize_text


def _entity(
    entity_id: uuid.UUID,
    *,
    profile_id: str = "p1",
    name: str,
    entity_type: str = "api",
    aliases: list[str] | None = None,
    status: str = STATUS_ACTIVE,
) -> Entity:
    return Entity(
        id=entity_id,
        profile_id=profile_id,
        name=name,
        name_normalized=normalize_text(name),
        type=entity_type,
        aliases=aliases or [name],
        status=status,
    )


def _alias(
    entity_id: uuid.UUID,
    *,
    profile_id: str = "p1",
    alias: str,
    entity_type: str = "api",
) -> EntityAlias:
    return EntityAlias(
        id=uuid.uuid4(),
        entity_id=entity_id,
        profile_id=profile_id,
        alias=alias,
        alias_normalized=normalize_text(alias),
        entity_type=entity_type,
        status=STATUS_ACTIVE,
    )


@pytest.mark.asyncio
async def test_suggests_compact_name_duplicates_without_auto_merging(db, session):
    active_id = uuid.uuid4()
    candidate_id = uuid.uuid4()
    other_profile_id = uuid.uuid4()
    session.add_all(
        [
            _entity(active_id, name="Shannon API", aliases=["Shannon API", "Shannon"]),
            _alias(active_id, alias="Shannon"),
            _entity(candidate_id, name="shannon-api", status=STATUS_CANDIDATE),
            _entity(other_profile_id, profile_id="p2", name="ShannonAPI"),
        ]
    )
    await session.flush()

    suggestions = await EntityMergeService(profile_id="p1").suggest(session)

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.target.entity_id == active_id
    assert suggestion.source.entity_id == candidate_id
    assert suggestion.confidence >= 0.9
    assert "compact_name_match" in suggestion.reasons
    assert all(candidate.entity_id != other_profile_id for candidate in [suggestion.source, suggestion.target])

    source = await session.get(Entity, candidate_id)
    target = await session.get(Entity, active_id)
    assert source is not None and source.status == STATUS_CANDIDATE
    assert source.merged_into is None
    assert target is not None and target.status == STATUS_ACTIVE


@pytest.mark.asyncio
async def test_reject_merge_suppresses_future_suggestions_and_records_event(db, session):
    active_id = uuid.uuid4()
    candidate_id = uuid.uuid4()
    session.add_all(
        [
            _entity(active_id, name="Claude Code", entity_type="service"),
            _entity(candidate_id, name="claude-code", entity_type="service"),
        ]
    )
    await session.flush()
    svc = EntityMergeService(profile_id="p1", actor="admin_api")
    assert await svc.suggest(session)

    result = await svc.reject(session, source_entity_id=candidate_id, target_entity_id=active_id, reason="different tools")

    assert result.action == "reject"
    assert await svc.suggest(session) == []
    events = (await session.execute(select(MemoryEvent))).scalars().all()
    assert [event.event_type for event in events] == ["entity_merge_rejected"]
    assert events[0].payload["source_entity_id"] == str(candidate_id)
    assert events[0].payload["target_entity_id"] == str(active_id)


@pytest.mark.asyncio
async def test_merge_rejects_already_merged_source(db, session):
    source_id = uuid.uuid4()
    target_id = uuid.uuid4()
    prior_target_id = uuid.uuid4()
    session.add_all(
        [
            _entity(source_id, name="Shannon API old", status=STATUS_MERGED),
            _entity(target_id, name="Shannon API", status=STATUS_ACTIVE),
            _entity(prior_target_id, name="Shannon API prior", status=STATUS_ACTIVE),
        ]
    )
    await session.flush()
    source = await session.get(Entity, source_id)
    assert source is not None
    source.merged_into = prior_target_id

    with pytest.raises(ValueError, match="source entity is already merged"):
        await EntityMergeService(profile_id="p1").merge(
            session,
            source_entity_id=source_id,
            target_entity_id=target_id,
            reason="stale review",
        )

    unchanged_source = await session.get(Entity, source_id)
    assert unchanged_source is not None
    assert unchanged_source.merged_into == prior_target_id


@pytest.mark.asyncio
async def test_merge_creates_target_alias_for_source_name_without_existing_alias_row(db, session):
    source_id = uuid.uuid4()
    target_id = uuid.uuid4()
    session.add_all(
        [
            _entity(source_id, name="Shannon API old", aliases=["Shannon API old"], status=STATUS_CANDIDATE),
            _entity(target_id, name="Shannon API", aliases=["Shannon API"], status=STATUS_ACTIVE),
        ]
    )
    await session.flush()

    await EntityMergeService(profile_id="p1").merge(
        session,
        source_entity_id=source_id,
        target_entity_id=target_id,
        reason="same entity",
    )

    aliases = (await session.execute(select(EntityAlias))).scalars().all()
    assert [(alias.entity_id, alias.alias_normalized) for alias in aliases] == [
        (target_id, "shannon api old")
    ]


@pytest.mark.asyncio
async def test_apply_merge_marks_source_moves_aliases_and_relationships(db, session):
    target_id = uuid.uuid4()
    source_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    rel_id = uuid.uuid4()
    entity_memory_rel_id = uuid.uuid4()
    self_memory_rel_id = uuid.uuid4()
    entity_rel_id = uuid.uuid4()
    self_rel_id = uuid.uuid4()
    sibling_id = uuid.uuid4()
    session.add_all(
        [
            _entity(target_id, name="Shannon API", aliases=["Shannon API"], status=STATUS_ACTIVE),
            _alias(target_id, alias="Shannon API"),
            _entity(source_id, name="shannon-api", aliases=["shannon-api"], status=STATUS_CANDIDATE),
            _alias(source_id, alias="shannon-api"),
            _entity(sibling_id, name="Shannon Gateway", aliases=["Shannon Gateway"], status=STATUS_ACTIVE),
            Fact(
                id=fact_id,
                profile_id="p1",
                subject="shannon-api",
                predicate="uses",
                object="gateway",
                statement="shannon-api uses gateway.",
                canonical_key=fact_canonical_key("shannon-api", "uses", "gateway"),
                confidence=0.9,
                status=STATUS_ACTIVE,
            ),
            MemoryRelationship(
                id=rel_id,
                profile_id="p1",
                source_type="fact",
                source_id=fact_id,
                relationship_type=RELATIONSHIP_BELONGS_TO_ENTITY,
                target_type="entity",
                target_id=source_id,
                confidence=0.82,
                rationale="source entity edge",
                created_by="test",
                status=STATUS_ACTIVE,
            ),
            MemoryRelationship(
                id=entity_memory_rel_id,
                profile_id="p1",
                source_type="entity",
                source_id=source_id,
                relationship_type="uses",
                target_type="entity",
                target_id=sibling_id,
                confidence=0.82,
                rationale="source to sibling",
                created_by="test",
                status=STATUS_ACTIVE,
            ),
            MemoryRelationship(
                id=self_memory_rel_id,
                profile_id="p1",
                source_type="entity",
                source_id=source_id,
                relationship_type="same_as",
                target_type="entity",
                target_id=target_id,
                confidence=0.82,
                rationale="source to target",
                created_by="test",
                status=STATUS_ACTIVE,
            ),
            EntityRelationship(
                id=entity_rel_id,
                profile_id="p1",
                source_entity_id=source_id,
                relation="uses",
                target_entity_id=sibling_id,
                confidence=0.8,
                status=STATUS_ACTIVE,
            ),
            EntityRelationship(
                id=self_rel_id,
                profile_id="p1",
                source_entity_id=source_id,
                relation="same_as",
                target_entity_id=target_id,
                confidence=0.8,
                status=STATUS_ACTIVE,
            ),
        ]
    )
    await session.flush()

    result = await EntityMergeService(profile_id="p1", actor="admin_api").merge(
        session,
        source_entity_id=source_id,
        target_entity_id=target_id,
        reason="same compact name",
    )

    assert result.action == "merge"
    source = await session.get(Entity, source_id)
    target = await session.get(Entity, target_id)
    assert source is not None and source.status == STATUS_MERGED
    assert source.merged_into == target_id
    assert target is not None
    assert target.aliases == ["Shannon API", "shannon-api"]

    moved_aliases = (
        await session.execute(select(EntityAlias).where(EntityAlias.entity_id == target_id))
    ).scalars().all()
    assert sorted(alias.alias for alias in moved_aliases) == ["Shannon API", "shannon-api"]
    rel = await session.get(MemoryRelationship, rel_id)
    assert rel is not None and rel.target_id == target_id
    entity_memory_rel = await session.get(MemoryRelationship, entity_memory_rel_id)
    assert entity_memory_rel is not None
    assert entity_memory_rel.source_id == target_id
    assert entity_memory_rel.target_id == sibling_id
    assert entity_memory_rel.status == STATUS_ACTIVE
    self_memory_rel = await session.get(MemoryRelationship, self_memory_rel_id)
    assert self_memory_rel is not None
    assert self_memory_rel.status == STATUS_MERGED
    assert self_memory_rel.source_id == source_id
    assert self_memory_rel.target_id == target_id
    entity_rel = await session.get(EntityRelationship, entity_rel_id)
    assert entity_rel is not None
    assert entity_rel.source_entity_id == target_id
    assert entity_rel.target_entity_id == sibling_id
    assert entity_rel.status == STATUS_ACTIVE
    self_rel = await session.get(EntityRelationship, self_rel_id)
    assert self_rel is not None
    assert self_rel.status == STATUS_MERGED
    assert self_rel.source_entity_id == source_id
    assert self_rel.target_entity_id == target_id

    events = (await session.execute(select(MemoryEvent))).scalars().all()
    assert [event.event_type for event in events] == ["entity_merge_applied"]
