"""Admin conflict + audit endpoints."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select

from siqueira_memo.config import settings_for_tests
from siqueira_memo.db import (
    create_all_for_tests,
    dispose_engines,
    drop_all_for_tests,
    get_session_factory,
)
from siqueira_memo.main import create_app
from siqueira_memo.models import MemoryEvent
from siqueira_memo.workers.queue import MemoryJobQueue, set_default_queue


@pytest_asyncio.fixture
async def api_client():
    settings = settings_for_tests()
    await create_all_for_tests(settings)
    set_default_queue(MemoryJobQueue())
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            yield client, settings
    await drop_all_for_tests(settings)
    await dispose_engines()
    set_default_queue(None)


def _auth(settings) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.api_token.get_secret_value()}"}


@pytest.mark.asyncio
async def test_conflicts_scan_and_resolve_end_to_end(api_client):
    client, settings = api_client
    auth = _auth(settings)
    # Seed two contradictory decisions via the public API.
    for stmt in (
        "Use MCP as primary integration",
        "Do not use MCP as primary integration",
    ):
        resp = await client.post(
            "/v1/memory/remember",
            headers=auth,
            json={
                "kind": "decision",
                "statement": stmt,
                "topic": "mcp integration",
                "project": "siqueira-memo",
            },
        )
        assert resp.status_code == 200

    scan = await client.post("/v1/admin/conflicts/scan", headers=auth, json={})
    assert scan.status_code == 200, scan.text
    body = scan.json()
    assert body["detected"] >= 1
    conflict = body["conflicts"][0]

    resolve = await client.post(
        "/v1/admin/conflicts/resolve",
        headers=auth,
        json={
            "conflict_id": conflict["id"],
            "kept_id": conflict["right_id"],
            "dropped_id": conflict["left_id"],
            "actor": "auto",
        },
    )
    assert resolve.status_code == 200, resolve.text
    assert resolve.json()["status"] == "auto_resolved"


@pytest.mark.asyncio
async def test_audit_endpoint_returns_metadata_only(api_client):
    client, settings = api_client
    auth = _auth(settings)
    # Remember then delete a fact so an audit entry is produced.
    fact_resp = await client.post(
        "/v1/memory/remember",
        headers=auth,
        json={
            "kind": "fact",
            "subject": "x",
            "predicate": "y",
            "object": "z",
            "statement": "x y z",
        },
    )
    assert fact_resp.status_code == 200
    await client.post(
        "/v1/memory/forget",
        headers=auth,
        json={"target_type": "fact", "target_id": fact_resp.json()["id"], "mode": "soft"},
    )

    resp = await client.post("/v1/admin/audit", headers=auth, json={"limit": 10})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    events = body["entries"]
    kinds = [e["event_type"] for e in events]
    assert "memory_deleted" in kinds
    # Audit must never return raw content.
    for entry in events:
        assert "content" not in entry
        assert "content_raw" not in entry


@pytest.mark.asyncio
async def test_admin_masks_secret_detail_and_reveal_is_audited(api_client):
    client, settings = api_client
    auth = _auth(settings)
    raw_secret = "sk-proj-" + "e" * 40
    remember = await client.post(
        "/v1/memory/remember",
        headers=auth,
        json={
            "kind": "fact",
            "subject": "OpenAI admin key",
            "predicate": "stored_as_secret",
            "object": raw_secret,
            "statement": f"OpenAI admin key is {raw_secret}.",
            "project": "siqueira-memo",
            "topic": "secrets",
            "metadata": {
                "sensitivity": "secret",
                "masked_preview": "OpenAI admin key is sk-proj-...eeee.",
                "secret_kind": "api_key",
                "secret_value": raw_secret,
            },
        },
    )
    assert remember.status_code == 200, remember.text
    fact_id = remember.json()["id"]

    detail = await client.post(
        "/v1/admin/detail",
        headers=auth,
        json={"target_type": "fact", "target_id": fact_id},
    )
    assert detail.status_code == 200, detail.text
    detail_text = detail.text
    assert raw_secret not in detail_text
    assert "sk-proj-...eeee" in detail_text
    assert detail.json()["item"]["secret_masked"] is True

    reveal = await client.post(
        "/v1/admin/secrets/reveal",
        headers=auth,
        json={"target_type": "fact", "target_id": fact_id, "reason": "integration-test"},
    )
    assert reveal.status_code == 200, reveal.text
    assert reveal.json()["secret_value"] == raw_secret

    factory = get_session_factory(settings)
    async with factory() as session:
        audit_events = (
            await session.execute(
                select(MemoryEvent).where(MemoryEvent.event_type == "secret_revealed")
            )
        ).scalars().all()
    assert audit_events
    assert raw_secret not in str(audit_events[0].payload)
