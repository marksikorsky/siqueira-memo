"""Admin entity card API integration tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import httpx
import pytest
import pytest_asyncio
from pydantic import SecretStr

from siqueira_memo.config import settings_for_tests
from siqueira_memo.db import (
    create_all_for_tests,
    dispose_engines,
    drop_all_for_tests,
    get_session_factory,
)
from siqueira_memo.main import create_app
from siqueira_memo.models import Entity, EntityAlias, Fact, MemoryRelationship
from siqueira_memo.models.constants import RELATIONSHIP_BELONGS_TO_ENTITY, STATUS_ACTIVE
from siqueira_memo.utils.canonical import fact_canonical_key, normalize_text
from siqueira_memo.workers.queue import MemoryJobQueue, set_default_queue

_now = datetime(2026, 4, 29, tzinfo=UTC)


@pytest_asyncio.fixture
async def api_client():
    settings = settings_for_tests(
        admin_password=SecretStr("test-admin-password"),
        admin_session_secret=SecretStr("test-admin-session-secret"),
    )
    await create_all_for_tests(settings)
    set_default_queue(MemoryJobQueue())
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client, settings
    await drop_all_for_tests(settings)
    await dispose_engines()
    set_default_queue(None)


def _auth(settings) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.api_token.get_secret_value()}"}


@pytest.mark.asyncio
async def test_admin_entity_card_endpoint_returns_source_backed_card(api_client):
    client, settings = api_client
    entity_id = uuid.uuid4()
    fact_id = uuid.uuid4()
    event_id = uuid.uuid4()
    factory = get_session_factory(settings)
    async with factory() as session:
        session.add_all(
            [
                Entity(
                    id=entity_id,
                    profile_id="p1",
                    name="Shannon API",
                    name_normalized=normalize_text("Shannon API"),
                    type="api",
                    aliases=["Shannon API"],
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                EntityAlias(
                    id=uuid.uuid4(),
                    entity_id=entity_id,
                    profile_id="p1",
                    alias="shannon",
                    alias_normalized=normalize_text("shannon"),
                    entity_type="api",
                    status=STATUS_ACTIVE,
                    created_at=_now,
                ),
                Fact(
                    id=fact_id,
                    profile_id="p1",
                    subject="Shannon API",
                    predicate="runs behind",
                    object="private gateway",
                    statement="Shannon API runs behind the private gateway.",
                    canonical_key=fact_canonical_key("Shannon API", "runs behind", "private gateway"),
                    project="shannon",
                    topic="api",
                    confidence=0.92,
                    status=STATUS_ACTIVE,
                    source_event_ids=[event_id],
                    created_at=_now,
                    updated_at=_now,
                ),
                MemoryRelationship(
                    id=uuid.uuid4(),
                    profile_id="p1",
                    source_type="fact",
                    source_id=fact_id,
                    relationship_type=RELATIONSHIP_BELONGS_TO_ENTITY,
                    target_type="entity",
                    target_id=entity_id,
                    confidence=0.9,
                    rationale="source-backed edge detail is internal only",
                    created_by="test",
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
            ]
        )
        await session.commit()

    resp = await client.post(
        "/v1/admin/entities/card",
        headers=_auth(settings),
        json={"profile_id": "p1", "name": "shannon", "entity_type": "api"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entity_id"] == str(entity_id)
    assert body["projects"] == ["shannon"]
    assert body["topics"] == ["api"]
    assert body["source_count"] == 1
    assert body["latest_facts"][0]["text"] == "Shannon API runs behind the private gateway."
    assert body["relationships"][0]["rationale"] is None


@pytest.mark.asyncio
async def test_admin_entity_card_endpoint_is_profile_scoped(api_client):
    client, settings = api_client
    entity_id = uuid.uuid4()
    factory = get_session_factory(settings)
    async with factory() as session:
        session.add(
            Entity(
                id=entity_id,
                profile_id="other-profile",
                name="Other API",
                name_normalized=normalize_text("Other API"),
                type="api",
                aliases=["Other API"],
                status=STATUS_ACTIVE,
                created_at=_now,
                updated_at=_now,
            )
        )
        await session.commit()

    resp = await client.post(
        "/v1/admin/entities/card",
        headers=_auth(settings),
        json={"profile_id": "p1", "entity_id": str(entity_id)},
    )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_admin_entities_list_returns_profile_scoped_project_entities(api_client):
    client, settings = api_client
    shannon_entity_id = uuid.uuid4()
    clawik_entity_id = uuid.uuid4()
    other_entity_id = uuid.uuid4()
    shannon_fact_id = uuid.uuid4()
    clawik_fact_id = uuid.uuid4()
    factory = get_session_factory(settings)
    async with factory() as session:
        session.add_all(
            [
                Entity(
                    id=shannon_entity_id,
                    profile_id="p1",
                    name="Shannon API",
                    name_normalized=normalize_text("Shannon API"),
                    type="api",
                    aliases=["Shannon API"],
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                EntityAlias(
                    id=uuid.uuid4(),
                    entity_id=shannon_entity_id,
                    profile_id="p1",
                    alias="shannon",
                    alias_normalized=normalize_text("shannon"),
                    entity_type="api",
                    status=STATUS_ACTIVE,
                    created_at=_now,
                ),
                Entity(
                    id=clawik_entity_id,
                    profile_id="p1",
                    name="Clawik Admin",
                    name_normalized=normalize_text("Clawik Admin"),
                    type="service",
                    aliases=["Clawik Admin"],
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                Entity(
                    id=other_entity_id,
                    profile_id="other-profile",
                    name="Other API",
                    name_normalized=normalize_text("Other API"),
                    type="api",
                    aliases=["Other API"],
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                Fact(
                    id=shannon_fact_id,
                    profile_id="p1",
                    subject="Shannon API",
                    predicate="belongs to",
                    object="Shannon",
                    statement="Shannon API belongs to Shannon.",
                    canonical_key=fact_canonical_key("Shannon API", "belongs to", "Shannon"),
                    project="shannon",
                    topic="api",
                    confidence=0.9,
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                Fact(
                    id=clawik_fact_id,
                    profile_id="p1",
                    subject="Clawik Admin",
                    predicate="belongs to",
                    object="Clawik",
                    statement="Clawik Admin belongs to Clawik.",
                    canonical_key=fact_canonical_key("Clawik Admin", "belongs to", "Clawik"),
                    project="clawik",
                    topic="admin",
                    confidence=0.9,
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                MemoryRelationship(
                    id=uuid.uuid4(),
                    profile_id="p1",
                    source_type="fact",
                    source_id=shannon_fact_id,
                    relationship_type=RELATIONSHIP_BELONGS_TO_ENTITY,
                    target_type="entity",
                    target_id=shannon_entity_id,
                    confidence=0.9,
                    rationale="source-backed edge detail is internal only",
                    created_by="test",
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                MemoryRelationship(
                    id=uuid.uuid4(),
                    profile_id="p1",
                    source_type="fact",
                    source_id=clawik_fact_id,
                    relationship_type=RELATIONSHIP_BELONGS_TO_ENTITY,
                    target_type="entity",
                    target_id=clawik_entity_id,
                    confidence=0.9,
                    rationale="source-backed edge detail is internal only",
                    created_by="test",
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
            ]
        )
        await session.commit()

    resp = await client.post(
        "/v1/admin/entities/list",
        headers=_auth(settings),
        json={"profile_id": "p1", "project": "shannon", "limit": 20},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 1
    assert [item["name"] for item in body["entities"]] == ["Shannon API"]
    item = body["entities"][0]
    assert item["entity_id"] == str(shannon_entity_id)
    assert item["entity_type"] == "api"
    assert item["aliases"] == ["Shannon API", "shannon"]
    assert item["projects"] == ["shannon"]
    assert item["topics"] == ["api"]
    assert item["source_count"] == 1
    assert item["relationship_count"] == 1

    profile_scoped = await client.post(
        "/v1/admin/entities/list",
        headers=_auth(settings),
        json={"profile_id": "p1", "query": "Other", "limit": 20},
    )
    assert profile_scoped.status_code == 200, profile_scoped.text
    assert profile_scoped.json()["entities"] == []


@pytest.mark.asyncio
async def test_admin_entities_list_filters_across_all_relationships_not_display_limit(api_client):
    client, settings = api_client
    entity_id = uuid.uuid4()
    first_fact_id = uuid.uuid4()
    target_fact_id = uuid.uuid4()
    factory = get_session_factory(settings)
    async with factory() as session:
        session.add_all(
            [
                Entity(
                    id=entity_id,
                    profile_id="p1",
                    name="Multi Project Entity",
                    name_normalized=normalize_text("Multi Project Entity"),
                    type="service",
                    aliases=["Multi Project Entity"],
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                Fact(
                    id=first_fact_id,
                    profile_id="p1",
                    subject="Multi Project Entity",
                    predicate="belongs to",
                    object="Old Project",
                    statement="Multi Project Entity belongs to Old Project.",
                    canonical_key=fact_canonical_key(
                        "Multi Project Entity", "belongs to", "Old Project"
                    ),
                    project="old-project",
                    topic="old-topic",
                    confidence=0.9,
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                Fact(
                    id=target_fact_id,
                    profile_id="p1",
                    subject="Multi Project Entity",
                    predicate="belongs to",
                    object="Target Project",
                    statement="Multi Project Entity belongs to Target Project.",
                    canonical_key=fact_canonical_key(
                        "Multi Project Entity", "belongs to", "Target Project"
                    ),
                    project="target-project",
                    topic="target-topic",
                    confidence=0.8,
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                MemoryRelationship(
                    id=uuid.uuid4(),
                    profile_id="p1",
                    source_type="fact",
                    source_id=first_fact_id,
                    relationship_type=RELATIONSHIP_BELONGS_TO_ENTITY,
                    target_type="entity",
                    target_id=entity_id,
                    confidence=0.99,
                    rationale="first edge must not hide later memberships",
                    created_by="test",
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                MemoryRelationship(
                    id=uuid.uuid4(),
                    profile_id="p1",
                    source_type="fact",
                    source_id=target_fact_id,
                    relationship_type=RELATIONSHIP_BELONGS_TO_ENTITY,
                    target_type="entity",
                    target_id=entity_id,
                    confidence=0.1,
                    rationale="low confidence target project still scopes the entity",
                    created_by="test",
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
            ]
        )
        await session.commit()

    resp = await client.post(
        "/v1/admin/entities/list",
        headers=_auth(settings),
        json={"profile_id": "p1", "project": "target-project", "topic": "target-topic"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 1
    item = body["entities"][0]
    assert item["name"] == "Multi Project Entity"
    assert item["projects"] == ["old-project", "target-project"]
    assert item["topics"] == ["old-topic", "target-topic"]
    assert item["source_count"] == 2
    assert item["relationship_count"] == 2


@pytest.mark.asyncio
async def test_admin_entities_list_paginates_after_full_filtered_total(api_client):
    client, settings = api_client
    factory = get_session_factory(settings)
    async with factory() as session:
        session.add_all(
            [
                Entity(
                    id=uuid.uuid4(),
                    profile_id="p1",
                    name=f"Bulk Entity {index:03d}",
                    name_normalized=normalize_text(f"Bulk Entity {index:03d}"),
                    type="service",
                    aliases=[f"Bulk Entity {index:03d}"],
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                )
                for index in range(601)
            ]
        )
        await session.commit()

    resp = await client.post(
        "/v1/admin/entities/list",
        headers=_auth(settings),
        json={"profile_id": "p1", "query": "Bulk Entity", "offset": 550, "limit": 1},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 601
    assert [item["name"] for item in body["entities"]] == ["Bulk Entity 550"]


@pytest.mark.asyncio
async def test_admin_entities_list_global_scope_only_returns_global_entities(api_client):
    client, settings = api_client
    global_entity_id = uuid.uuid4()
    project_entity_id = uuid.uuid4()
    global_fact_id = uuid.uuid4()
    project_fact_id = uuid.uuid4()
    factory = get_session_factory(settings)
    async with factory() as session:
        session.add_all(
            [
                Entity(
                    id=global_entity_id,
                    profile_id="p1",
                    name="Global Entity",
                    name_normalized=normalize_text("Global Entity"),
                    type="service",
                    aliases=["Global Entity"],
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                Entity(
                    id=project_entity_id,
                    profile_id="p1",
                    name="Project Entity",
                    name_normalized=normalize_text("Project Entity"),
                    type="service",
                    aliases=["Project Entity"],
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                Fact(
                    id=global_fact_id,
                    profile_id="p1",
                    subject="Global Entity",
                    predicate="is",
                    object="global",
                    statement="Global Entity is global.",
                    canonical_key=fact_canonical_key("Global Entity", "is", "global"),
                    project=None,
                    topic="shared",
                    confidence=0.9,
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                Fact(
                    id=project_fact_id,
                    profile_id="p1",
                    subject="Project Entity",
                    predicate="belongs to",
                    object="Project",
                    statement="Project Entity belongs to Project.",
                    canonical_key=fact_canonical_key("Project Entity", "belongs to", "Project"),
                    project="project-x",
                    topic="shared",
                    confidence=0.9,
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                MemoryRelationship(
                    id=uuid.uuid4(),
                    profile_id="p1",
                    source_type="fact",
                    source_id=global_fact_id,
                    relationship_type=RELATIONSHIP_BELONGS_TO_ENTITY,
                    target_type="entity",
                    target_id=global_entity_id,
                    confidence=0.9,
                    rationale="global relation",
                    created_by="test",
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
                MemoryRelationship(
                    id=uuid.uuid4(),
                    profile_id="p1",
                    source_type="fact",
                    source_id=project_fact_id,
                    relationship_type=RELATIONSHIP_BELONGS_TO_ENTITY,
                    target_type="entity",
                    target_id=project_entity_id,
                    confidence=0.9,
                    rationale="project relation",
                    created_by="test",
                    status=STATUS_ACTIVE,
                    created_at=_now,
                    updated_at=_now,
                ),
            ]
        )
        await session.commit()

    resp = await client.post(
        "/v1/admin/entities/list",
        headers=_auth(settings),
        json={"profile_id": "p1", "project_scope": "global", "limit": 20},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 1
    assert [item["name"] for item in body["entities"]] == ["Global Entity"]
