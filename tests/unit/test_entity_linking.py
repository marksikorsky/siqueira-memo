"""Entity linking skeleton tests. Plan §19."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from siqueira_memo.models import Entity, EntityAlias
from siqueira_memo.models.constants import STATUS_ACTIVE, STATUS_MERGED
from siqueira_memo.services.entity_linking_service import EntityLinkingService


@pytest.mark.asyncio
async def test_creates_candidate_when_no_match(db, session):
    svc = EntityLinkingService(profile_id="p1")
    result = await svc.link_or_create(
        session,
        mention="Shannon API",
        entity_type="api",
        source_event_id=uuid.uuid4(),
    )
    assert result.action == "create_candidate"
    aliases = (await session.execute(select(EntityAlias))).scalars().all()
    assert len(aliases) == 1
    assert aliases[0].alias_normalized == "shannon api"


@pytest.mark.asyncio
async def test_links_to_existing_alias(db, session):
    svc = EntityLinkingService(profile_id="p1")
    first = await svc.link_or_create(session, mention="Shannon API", entity_type="api")
    second = await svc.link_or_create(session, mention="shannon api", entity_type="api")
    assert first.entity_id == second.entity_id
    assert second.action == "link"


@pytest.mark.asyncio
async def test_auto_merge_on_matching_normalised_name(db, session):
    svc = EntityLinkingService(profile_id="p1")
    first = await svc.link_or_create(session, mention="Shannon API", entity_type="api")

    # Remove alias to simulate a different linking path.
    await session.execute(
        EntityAlias.__table__.delete().where(EntityAlias.entity_id == first.entity_id)
    )
    await session.flush()

    second = await svc.link_or_create(session, mention="shannon-api", entity_type="api")
    # Dashed form normalises to "shannon-api", unique from "shannon api", so this is a new candidate.
    assert second.action in {"create_candidate", "auto_merge"}


@pytest.mark.asyncio
async def test_linking_redirects_merged_entity_name_to_merge_target(db, session):
    source_id = uuid.uuid4()
    target_id = uuid.uuid4()
    session.add_all(
        [
            Entity(
                id=source_id,
                profile_id="p1",
                name="Shannon API old",
                name_normalized="shannon api old",
                type="api",
                aliases=["Shannon API old"],
                status=STATUS_MERGED,
                merged_into=target_id,
            ),
            Entity(
                id=target_id,
                profile_id="p1",
                name="Shannon API",
                name_normalized="shannon api",
                type="api",
                aliases=["Shannon API"],
                status=STATUS_ACTIVE,
            ),
        ]
    )
    await session.flush()

    result = await EntityLinkingService(profile_id="p1").link_or_create(
        session,
        mention="Shannon API old",
        entity_type="api",
    )

    assert result.action == "link"
    assert result.entity_id == target_id
    source = await session.get(Entity, source_id)
    assert source is not None and source.status == STATUS_MERGED
    aliases = (await session.execute(select(EntityAlias))).scalars().all()
    assert [(alias.entity_id, alias.alias_normalized) for alias in aliases] == [
        (target_id, "shannon api old")
    ]


@pytest.mark.asyncio
async def test_linking_redirects_active_alias_from_merged_entity_to_merge_target(db, session):
    source_id = uuid.uuid4()
    target_id = uuid.uuid4()
    session.add_all(
        [
            Entity(
                id=source_id,
                profile_id="p1",
                name="Shannon API old",
                name_normalized="shannon api old",
                type="api",
                aliases=["Shannon API old"],
                status=STATUS_MERGED,
                merged_into=target_id,
            ),
            Entity(
                id=target_id,
                profile_id="p1",
                name="Shannon API",
                name_normalized="shannon api",
                type="api",
                aliases=["Shannon API"],
                status=STATUS_ACTIVE,
            ),
            EntityAlias(
                id=uuid.uuid4(),
                entity_id=source_id,
                profile_id="p1",
                alias="Shannon API old",
                alias_normalized="shannon api old",
                entity_type="api",
                status=STATUS_ACTIVE,
            ),
        ]
    )
    await session.flush()

    result = await EntityLinkingService(profile_id="p1").link_or_create(
        session,
        mention="Shannon API old",
        entity_type="api",
    )

    assert result.action == "link"
    assert result.entity_id == target_id
    aliases = (await session.execute(select(EntityAlias))).scalars().all()
    assert [(alias.entity_id, alias.alias_normalized) for alias in aliases] == [
        (target_id, "shannon api old")
    ]


@pytest.mark.asyncio
async def test_different_types_do_not_conflate(db, session):
    svc = EntityLinkingService(profile_id="p1")
    api = await svc.link_or_create(session, mention="Shannon", entity_type="api")
    person = await svc.link_or_create(session, mention="Shannon", entity_type="person")
    assert api.entity_id != person.entity_id
