"""Entity card service tests. Roadmap Phase 7."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from siqueira_memo.models import Decision, Entity, EntityAlias, Fact, MemoryRelationship
from siqueira_memo.models.constants import (
    RELATIONSHIP_BELONGS_TO_ENTITY,
    RELATIONSHIP_CONTRADICTS,
    RELATIONSHIP_USES_SECRET,
    STATUS_ACTIVE,
    STATUS_SUPERSEDED,
)
from siqueira_memo.services.entity_card_service import EntityCardService
from siqueira_memo.utils.canonical import decision_canonical_key, fact_canonical_key, normalize_text

_now = datetime(2026, 4, 29, tzinfo=UTC)


def _entity(*, name: str = "Shannon API", entity_type: str = "api", profile_id: str = "p1") -> Entity:
    return Entity(
        id=uuid.uuid4(),
        profile_id=profile_id,
        name=name,
        name_normalized=normalize_text(name),
        type=entity_type,
        aliases=[name, "Shannon"],
        description="Source-backed API entity.",
        status=STATUS_ACTIVE,
        created_at=_now,
        updated_at=_now,
    )


def _alias(entity: Entity, alias: str) -> EntityAlias:
    return EntityAlias(
        id=uuid.uuid4(),
        entity_id=entity.id,
        profile_id=entity.profile_id,
        alias=alias,
        alias_normalized=normalize_text(alias),
        entity_type=entity.type,
        status=STATUS_ACTIVE,
        created_at=_now,
    )


def _fact(
    statement: str,
    *,
    profile_id: str = "p1",
    project: str | None = "shannon",
    topic: str | None = "api",
    status: str = STATUS_ACTIVE,
    event_ids: list[uuid.UUID] | None = None,
    metadata: dict[str, object] | None = None,
) -> Fact:
    return Fact(
        id=uuid.uuid4(),
        profile_id=profile_id,
        subject="Shannon API",
        predicate="has fact",
        object=statement,
        statement=statement,
        canonical_key=fact_canonical_key("Shannon API", "has fact", statement),
        project=project,
        topic=topic,
        confidence=0.91,
        status=status,
        source_event_ids=event_ids or [],
        extra_metadata=metadata or {},
        created_at=_now,
        updated_at=_now,
    )


def _decision(
    decision: str,
    *,
    profile_id: str = "p1",
    project: str | None = "shannon",
    topic: str = "api",
    event_ids: list[uuid.UUID] | None = None,
) -> Decision:
    return Decision(
        id=uuid.uuid4(),
        profile_id=profile_id,
        project=project,
        topic=topic,
        decision=decision,
        context="Source-backed decision context.",
        rationale="Because it was explicitly decided.",
        canonical_key=decision_canonical_key(project, topic, decision),
        status=STATUS_ACTIVE,
        reversible=True,
        decided_at=_now - timedelta(minutes=5),
        source_event_ids=event_ids or [],
        created_at=_now - timedelta(minutes=5),
        updated_at=_now - timedelta(minutes=5),
    )


def _relationship(
    *,
    entity: Entity,
    source_type: str,
    source_id: uuid.UUID,
    relationship_type: str = RELATIONSHIP_BELONGS_TO_ENTITY,
    target_type: str = "entity",
    target_id: uuid.UUID | None = None,
    confidence: float = 0.88,
) -> MemoryRelationship:
    return MemoryRelationship(
        id=uuid.uuid4(),
        profile_id=entity.profile_id,
        source_type=source_type,
        source_id=source_id,
        relationship_type=relationship_type,
        target_type=target_type,
        target_id=target_id or entity.id,
        confidence=confidence,
        rationale="source-backed test edge",
        created_by="test",
        status=STATUS_ACTIVE,
        created_at=_now,
        updated_at=_now,
    )


@pytest.mark.asyncio
async def test_entity_card_resolves_alias_and_summarises_source_backed_rows(db, session):
    entity = _entity()
    event_a = uuid.uuid4()
    event_b = uuid.uuid4()
    fact = _fact("Shannon API runs behind the private gateway.", event_ids=[event_a])
    decision = _decision("Use Shannon API as the routing boundary.", event_ids=[event_b])
    session.add_all(
        [
            entity,
            _alias(entity, "shannon"),
            fact,
            decision,
            _relationship(entity=entity, source_type="fact", source_id=fact.id),
            _relationship(entity=entity, source_type="decision", source_id=decision.id),
        ]
    )
    await session.flush()

    card = await EntityCardService(profile_id="p1").build_card(
        session, name="shannon", entity_type="api"
    )

    assert card.entity_id == entity.id
    assert card.name == "Shannon API"
    assert card.aliases == ["Shannon API", "Shannon"]
    assert card.projects == ["shannon"]
    assert card.topics == ["api"]
    assert [item.text for item in card.latest_facts] == [fact.statement]
    assert [item.text for item in card.active_decisions] == [decision.decision]
    assert card.source_count == 2
    assert card.confidence == pytest.approx(0.88)
    assert card.last_updated == _now
    assert all("psycholog" not in item.text.lower() for item in card.latest_facts)


@pytest.mark.asyncio
async def test_entity_card_marks_conflicts_and_stale_related_memories(db, session):
    entity = _entity()
    stale_fact = _fact("Old Shannon API deployment fact.", status=STATUS_SUPERSEDED)
    active_fact = _fact("Current Shannon API deployment fact.")
    session.add_all(
        [
            entity,
            stale_fact,
            active_fact,
            _relationship(entity=entity, source_type="fact", source_id=stale_fact.id),
            _relationship(entity=entity, source_type="fact", source_id=active_fact.id),
            _relationship(
                entity=entity,
                source_type="fact",
                source_id=active_fact.id,
                relationship_type=RELATIONSHIP_CONTRADICTS,
                target_type="fact",
                target_id=stale_fact.id,
            ),
        ]
    )
    await session.flush()

    card = await EntityCardService(profile_id="p1").build_card(session, entity_id=entity.id)

    assert [item.text for item in card.latest_facts] == [active_fact.statement]
    assert card.conflict_count == 1
    assert any("conflict" in warning.lower() for warning in card.warnings)
    assert any("superseded" in warning.lower() for warning in card.warnings)


@pytest.mark.asyncio
async def test_entity_card_masks_related_secret_memories(db, session):
    entity = _entity()
    secret_fact = _fact(
        "Shannon production credential is stored in vault item X.",
        metadata={"sensitivity": "secret", "masked_preview": "Shannon credential: ***"},
    )
    session.add_all(
        [
            entity,
            secret_fact,
            _relationship(
                entity=entity,
                source_type="entity",
                source_id=entity.id,
                relationship_type=RELATIONSHIP_USES_SECRET,
                target_type="fact",
                target_id=secret_fact.id,
            ),
        ]
    )
    await session.flush()

    card = await EntityCardService(profile_id="p1").build_card(session, entity_id=entity.id)

    assert card.related_secrets
    assert card.related_secrets[0].text == "Shannon credential: ***"
    assert secret_fact.statement not in card.related_secrets[0].text
    assert card.latest_facts == []


@pytest.mark.asyncio
async def test_entity_card_treats_uses_secret_relationship_as_secret_without_metadata(db, session):
    entity = _entity()
    relationship_only_secret = _fact("Shannon credential lives in the operator vault item.")
    session.add_all(
        [
            entity,
            relationship_only_secret,
            _relationship(
                entity=entity,
                source_type="entity",
                source_id=entity.id,
                relationship_type=RELATIONSHIP_USES_SECRET,
                target_type="fact",
                target_id=relationship_only_secret.id,
            ),
        ]
    )
    await session.flush()

    card = await EntityCardService(profile_id="p1").build_card(session, entity_id=entity.id)

    assert card.latest_facts == []
    assert card.related_secrets
    assert relationship_only_secret.statement not in card.related_secrets[0].text
    assert card.related_secrets[0].text == "[SECRET_MASKED]"


@pytest.mark.asyncio
async def test_entity_card_forces_relationship_secret_over_non_secret_metadata(db, session):
    entity = _entity()
    secret_fact = _fact(
        "Shannon credential lives in the operator vault item.",
        metadata={"sensitivity": "internal"},
    )
    session.add_all(
        [
            entity,
            secret_fact,
            _relationship(
                entity=entity,
                source_type="entity",
                source_id=entity.id,
                relationship_type=RELATIONSHIP_USES_SECRET,
                target_type="fact",
                target_id=secret_fact.id,
            ),
        ]
    )
    await session.flush()

    card = await EntityCardService(profile_id="p1").build_card(session, entity_id=entity.id)

    assert card.latest_facts == []
    assert card.related_secrets
    assert secret_fact.statement not in card.related_secrets[0].text


@pytest.mark.asyncio
async def test_entity_card_secret_classification_ignores_display_relationship_limit(db, session):
    entity = _entity()
    secret_fact = _fact("Shannon credential lives in the operator vault item.")
    visible_normal_edge = _relationship(
        entity=entity,
        source_type="fact",
        source_id=secret_fact.id,
        relationship_type=RELATIONSHIP_BELONGS_TO_ENTITY,
        confidence=0.99,
    )
    hidden_secret_edge = _relationship(
        entity=entity,
        source_type="entity",
        source_id=entity.id,
        relationship_type=RELATIONSHIP_USES_SECRET,
        target_type="fact",
        target_id=secret_fact.id,
        confidence=0.01,
    )
    session.add_all([entity, secret_fact, visible_normal_edge, hidden_secret_edge])
    await session.flush()

    card = await EntityCardService(profile_id="p1").build_card(
        session, entity_id=entity.id, relationship_limit=1
    )

    assert card.latest_facts == []
    assert card.related_secrets
    assert secret_fact.statement not in card.related_secrets[0].text
    assert len(card.relationships) == 1


@pytest.mark.asyncio
async def test_entity_card_relationships_do_not_return_raw_rationale(db, session):
    entity = _entity()
    secret_fact = _fact("Shannon credential lives in the operator vault item.")
    rel = _relationship(
        entity=entity,
        source_type="entity",
        source_id=entity.id,
        relationship_type=RELATIONSHIP_USES_SECRET,
        target_type="fact",
        target_id=secret_fact.id,
    )
    rel.rationale = "raw rationale copies the operator vault item detail"
    session.add_all([entity, secret_fact, rel])
    await session.flush()

    card = await EntityCardService(profile_id="p1").build_card(session, entity_id=entity.id)

    assert card.relationships
    assert all(item.rationale is None for item in card.relationships)


@pytest.mark.asyncio
async def test_entity_card_is_profile_scoped(db, session):
    entity = _entity(profile_id="other-profile")
    session.add(entity)
    await session.flush()

    with pytest.raises(ValueError, match="entity not found"):
        await EntityCardService(profile_id="p1").build_card(session, entity_id=entity.id)
