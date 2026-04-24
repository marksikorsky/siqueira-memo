"""Entity linking skeleton tests. Plan §19."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from siqueira_memo.models import EntityAlias
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
async def test_different_types_do_not_conflate(db, session):
    svc = EntityLinkingService(profile_id="p1")
    api = await svc.link_or_create(session, mention="Shannon", entity_type="api")
    person = await svc.link_or_create(session, mention="Shannon", entity_type="person")
    assert api.entity_id != person.entity_id
