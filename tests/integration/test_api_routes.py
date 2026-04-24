"""End-to-end API integration tests via ASGI transport."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from siqueira_memo.config import settings_for_tests
from siqueira_memo.db import (
    create_all_for_tests,
    dispose_engines,
    drop_all_for_tests,
)
from siqueira_memo.main import create_app
from siqueira_memo.workers.queue import MemoryJobQueue, set_default_queue


@pytest_asyncio.fixture
async def api_client():
    settings = settings_for_tests()
    await create_all_for_tests(settings)
    set_default_queue(MemoryJobQueue())
    app = create_app(settings)
    # httpx.ASGITransport does not run FastAPI's lifespan — drive it manually
    # so app.state (prompt parity hashes, etc.) is populated.
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
async def test_healthz_is_unauthenticated(api_client):
    client, _settings = api_client
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["env"] == "test"


@pytest.mark.asyncio
async def test_readyz_reports_sqlite(api_client):
    client, _settings = api_client
    resp = await client.get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    # Plan §31.11 / §31.13 surfaces.
    assert body["partitions"]["required"] is False
    assert body["partitions"]["missing_current"] == []
    assert body["prompt_parity"]["ok"] is True
    assert len(body["prompt_parity"]["canonical_hash"]) == 64


@pytest.mark.asyncio
async def test_ingest_requires_auth(api_client):
    client, _settings = api_client
    resp = await client.post(
        "/v1/ingest/message",
        json={"session_id": "s", "role": "user", "content": "hi"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_full_roundtrip(api_client):
    client, settings = api_client
    auth = _auth(settings)

    # ingest
    ingest = await client.post(
        "/v1/ingest/message",
        headers=auth,
        json={
            "session_id": "s-rt",
            "platform": "cli",
            "role": "user",
            "content": "We decided to use Hermes MemoryProvider plugin as primary integration.",
        },
    )
    assert ingest.status_code == 200, ingest.text
    ingest_body = ingest.json()

    # remember
    remember = await client.post(
        "/v1/memory/remember",
        headers=auth,
        json={
            "kind": "decision",
            "statement": "Use Hermes MemoryProvider plugin as primary integration",
            "topic": "memory integration",
            "project": "siqueira-memo",
            "source_event_ids": [ingest_body["event_id"]],
        },
    )
    assert remember.status_code == 200, remember.text
    remember_body = remember.json()
    decision_id = remember_body["id"]

    # recall
    recall = await client.post(
        "/v1/recall",
        headers=auth,
        json={"query": "what did we decide about memory integration?", "mode": "balanced"},
    )
    assert recall.status_code == 200, recall.text
    recall_body = recall.json()
    assert recall_body["context_pack"]["decisions"], recall_body
    decision_ids = [d["id"] for d in recall_body["context_pack"]["decisions"]]
    assert decision_id in decision_ids

    # sources
    sources = await client.post(
        "/v1/memory/sources",
        headers=auth,
        json={"target_type": "decision", "target_id": decision_id},
    )
    assert sources.status_code == 200
    assert sources.json()["sources"]

    # timeline
    timeline = await client.post(
        "/v1/memory/timeline",
        headers=auth,
        json={"project": "siqueira-memo"},
    )
    assert timeline.status_code == 200
    entries = timeline.json()["entries"]
    assert any(e["kind"] == "decision" for e in entries)

    # admin search
    search = await client.post(
        "/v1/admin/search",
        headers=auth,
        json={"query": "Hermes", "target_type": "decision"},
    )
    assert search.status_code == 200
    assert search.json()["total"] >= 1

    # forget (soft)
    forget = await client.post(
        "/v1/memory/forget",
        headers=auth,
        json={"target_type": "decision", "target_id": decision_id, "mode": "soft"},
    )
    assert forget.status_code == 200
    assert forget.json()["invalidated_decisions"] == 1


@pytest.mark.asyncio
async def test_correct_via_api(api_client):
    client, settings = api_client
    auth = _auth(settings)
    first = await client.post(
        "/v1/memory/remember",
        headers=auth,
        json={
            "kind": "fact",
            "subject": "shannon",
            "predicate": "primary_auth",
            "object": "api_key",
            "statement": "Shannon auth is api key",
        },
    )
    assert first.status_code == 200
    fact_id = first.json()["id"]

    corr = await client.post(
        "/v1/memory/correct",
        headers=auth,
        json={
            "target_type": "fact",
            "target_id": fact_id,
            "correction_text": "actually it's the OAuth token",
            "replacement": {
                "kind": "fact",
                "subject": "shannon",
                "predicate": "primary_auth",
                "object": "claude_oauth",
                "statement": "Shannon auth is Claude OAuth token",
            },
        },
    )
    assert corr.status_code == 200, corr.text
    body = corr.json()
    assert body["invalidated"] == [fact_id]
    assert body["replacement_id"]


@pytest.mark.asyncio
async def test_hermes_compaction_ingest(api_client):
    client, settings = api_client
    auth = _auth(settings)
    resp = await client.post(
        "/v1/ingest/hermes-compaction",
        headers=auth,
        json={
            "session_id": "s-aux",
            "summary_text": "[CONTEXT COMPACTION — REFERENCE ONLY] body",
            "source_message_count": 24,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["event_id"]
