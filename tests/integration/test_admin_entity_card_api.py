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
