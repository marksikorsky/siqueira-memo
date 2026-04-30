"""End-to-end API integration tests via ASGI transport."""

from __future__ import annotations

from http import cookies

import httpx
import pytest
import pytest_asyncio
from pydantic import SecretStr

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
    settings = settings_for_tests(
        admin_password=SecretStr("test-admin-password"),
        admin_session_secret=SecretStr("test-admin-session-secret"),
    )
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
async def test_recall_api_allow_secret_recall_still_returns_masked_secret_records(api_client):
    client, settings = api_client
    auth = _auth(settings)
    raw_secret = "sk-proj-" + "r" * 40
    remember = await client.post(
        "/v1/memory/remember",
        headers=auth,
        json={
            "kind": "fact",
            "subject": "OpenAI recall API key",
            "predicate": "stored_as_secret",
            "object": raw_secret,
            "statement": f"OpenAI recall API key is {raw_secret}.",
            "topic": "secrets",
            "metadata": {
                "sensitivity": "secret",
                "masked_preview": "OpenAI recall API key is sk-proj-...rrrr.",
                "secret_value": raw_secret,
            },
        },
    )
    assert remember.status_code == 200, remember.text

    recall = await client.post(
        "/v1/recall",
        headers=auth,
        json={
            "query": "OpenAI recall API key",
            "types": ["facts"],
            "mode": "balanced",
            "allow_secret_recall": True,
        },
    )
    assert recall.status_code == 200, recall.text
    body_text = recall.text
    body = recall.json()
    assert raw_secret not in body_text
    assert "sk-proj-...rrrr" in body_text
    assert body["context_pack"]["facts"]
    assert body["context_pack"]["facts"][0]["secret_masked"] is True


@pytest.mark.asyncio
async def test_ingest_requires_auth(api_client):
    client, _settings = api_client
    resp = await client.post(
        "/v1/ingest/message",
        json={"session_id": "s", "role": "user", "content": "hi"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_admin_ui_uses_form_login_instead_of_basic_auth_popup(api_client):
    client, _settings = api_client
    resp = await client.get("/admin", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/admin/login"
    assert "www-authenticate" not in resp.headers

    login = await client.get("/admin/login")
    assert login.status_code == 200
    assert login.headers["content-type"].startswith("text/html")
    assert "www-authenticate" not in login.headers
    assert "Siqueira Memo Admin" in login.text
    assert '<form method="post" action="/admin/login"' in login.text
    assert 'name="password"' in login.text

    wrong = await client.post(
        "/admin/login",
        data={"password": "wrong-password"},
        follow_redirects=False,
    )
    assert wrong.status_code == 401
    assert "Invalid password" in wrong.text
    assert "www-authenticate" not in wrong.headers

    ok = await client.post(
        "/admin/login",
        data={"password": "test-admin-password"},
        follow_redirects=False,
    )
    assert ok.status_code == 303
    assert ok.headers["location"] == "/admin"
    morsel = cookies.SimpleCookie(ok.headers["set-cookie"])["siqueira_admin_session"]
    assert morsel["httponly"]
    assert morsel["samesite"].lower() == "lax"

    dashboard = await client.get("/admin")
    assert dashboard.status_code == 200
    assert dashboard.headers["content-type"].startswith("text/html")

    session_api = await client.post("/v1/admin/projects", json={"profile_id": "default"})
    assert session_api.status_code == 200, session_api.text

    html = dashboard.text
    assert "Siqueira Memo" in html
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert "linear-gradient" not in html.lower()
    assert "react" not in html.lower()
    assert "@media (max-width: 720px)" in html
    assert "localStorage" in html
    assert 'value="siqueira-memo"' in html
    assert "Siqueira project" in html
    assert "loadDefaultProject" in html
    assert "Project overview" in html
    assert "Detail drawer" in html
    assert "Recall playground" in html
    assert "Memory Capture" in html
    assert "project-scope" in html
    assert "Global" in html
    assert "Conflicts" in html
    assert "Audit" in html
    assert "Export Markdown" in html
    assert "bottom-nav" in html
    assert "safe-area-inset-bottom" in html
    assert "/v1/admin/projects" in html
    assert "/v1/admin/capture" in html
    assert "/v1/admin/detail" in html
    assert "/v1/admin/export" in html
    assert "/v1/recall" in html
    assert "/v1/admin/conflicts/scan" in html
    assert "/v1/admin/audit" in html
    assert "/v1/admin/search" in html
    assert "/v1/memory/timeline" in html
    assert "/v1/memory/sources" in html
    assert "/v1/admin/relationships/list" in html
    assert "/v1/admin/entities/list" in html
    assert "/v1/admin/entities/card" in html
    assert "/v1/admin/entities/merge-suggestions" in html
    assert "/v1/admin/entities/merge-review" in html
    assert "Merge suggestions" in html
    assert "loadEntityMergeSuggestions" in html
    assert "reviewEntityMerge" in html
    assert "Entities" in html
    assert "loadEntities" in html
    assert "openEntityCard" in html
    assert "Relationship Graph" in html
    assert "relationship-badge" in html
    assert "trust-filter" in html
    assert "trust-badge" in html
    assert "/v1/admin/trust/feedback" in html
    assert "--bg: #fbfaf8" in html

    logout = await client.post("/admin/logout", follow_redirects=False)
    assert logout.status_code == 303
    assert logout.headers["location"] == "/admin/login"
    cleared = cookies.SimpleCookie(logout.headers["set-cookie"])["siqueira_admin_session"]
    assert cleared.value == ""
    assert int(cleared["max-age"]) == 0


@pytest.mark.asyncio
async def test_admin_projects_detail_and_export(api_client):
    client, settings = api_client
    auth = _auth(settings)
    fact = await client.post(
        "/v1/memory/remember",
        headers=auth,
        json={
            "kind": "fact",
            "subject": "Siqueira UI",
            "predicate": "has",
            "object": "project overview",
            "statement": "Siqueira UI has a project overview dashboard.",
            "project": "siqueira-memo",
            "topic": "admin-ui",
        },
    )
    assert fact.status_code == 200, fact.text
    fact_id = fact.json()["id"]
    decision = await client.post(
        "/v1/memory/remember",
        headers=auth,
        json={
            "kind": "decision",
            "statement": "Use a zero-build mobile admin UI for Siqueira.",
            "topic": "admin-ui",
            "project": "siqueira-memo",
            "rationale": "Small internal tool; npm would be needless overhead.",
        },
    )
    assert decision.status_code == 200, decision.text

    projects = await client.post(
        "/v1/admin/projects",
        headers=auth,
        json={"profile_id": "default"},
    )
    assert projects.status_code == 200, projects.text
    siqueira = next(p for p in projects.json()["projects"] if p["project"] == "siqueira-memo")
    assert siqueira["facts"] >= 1
    assert siqueira["decisions"] >= 1
    assert any(t["topic"] == "admin-ui" for t in siqueira["topics"])

    detail = await client.post(
        "/v1/admin/detail",
        headers=auth,
        json={"target_type": "fact", "target_id": fact_id},
    )
    assert detail.status_code == 200, detail.text
    detail_body = detail.json()
    assert detail_body["item"]["id"] == fact_id
    assert detail_body["item"]["statement"] == "Siqueira UI has a project overview dashboard."
    assert detail_body["sources"]

    export = await client.post(
        "/v1/admin/export",
        headers=auth,
        json={"project": "siqueira-memo", "format": "markdown"},
    )
    assert export.status_code == 200, export.text
    assert export.headers["content-type"].startswith("text/markdown")
    assert "# Siqueira Memo Memory Export" in export.text
    assert "Siqueira UI has a project overview dashboard." in export.text

    rel = await client.post(
        "/v1/memory/relationships/create",
        headers=auth,
        json={
            "source_type": "fact",
            "source_id": fact_id,
            "relationship_type": "related_to",
            "target_type": "decision",
            "target_id": decision.json()["id"],
            "confidence": 0.88,
            "rationale": "UI fact is related to the admin UI decision.",
        },
    )
    assert rel.status_code == 200, rel.text
    assert rel.json()["relationship_type"] == "related_to"

    rels = await client.post(
        "/v1/admin/relationships/list",
        headers=auth,
        json={"target_type": "fact", "target_id": fact_id},
    )
    assert rels.status_code == 200, rels.text
    assert rels.json()["relationships"]


@pytest.mark.asyncio
async def test_admin_search_supports_global_scope_and_capture_stats(api_client):
    client, settings = api_client
    auth = _auth(settings)
    global_fact = await client.post(
        "/v1/memory/remember",
        headers=auth,
        json={
            "kind": "fact",
            "subject": "Mark",
            "predicate": "memory policy",
            "object": "save almost everything useful",
            "statement": "Mark wants Siqueira to save almost everything useful globally.",
            "project": None,
            "topic": "memory-write-policy",
        },
    )
    assert global_fact.status_code == 200, global_fact.text
    project_fact = await client.post(
        "/v1/memory/remember",
        headers=auth,
        json={
            "kind": "fact",
            "subject": "Siqueira",
            "predicate": "has",
            "object": "capture dashboard",
            "statement": "Siqueira has a capture dashboard.",
            "project": "siqueira-memo",
            "topic": "capture",
        },
    )
    assert project_fact.status_code == 200, project_fact.text

    global_search = await client.post(
        "/v1/admin/search",
        headers=auth,
        json={"target_type": "fact", "project_scope": "global", "limit": 20},
    )
    assert global_search.status_code == 200, global_search.text
    global_hits = global_search.json()["hits"]
    assert global_hits
    assert all(hit["project"] is None for hit in global_hits)
    assert any("almost everything useful" in hit["preview"] for hit in global_hits)

    project_search = await client.post(
        "/v1/admin/search",
        headers=auth,
        json={
            "target_type": "fact",
            "project_scope": "project",
            "project": "siqueira-memo",
            "limit": 20,
        },
    )
    assert project_search.status_code == 200, project_search.text
    project_hits = project_search.json()["hits"]
    assert project_hits
    assert all(hit["project"] == "siqueira-memo" for hit in project_hits)

    capture = await client.post(
        "/v1/admin/capture",
        headers=auth,
        json={"profile_id": "default"},
    )
    assert capture.status_code == 200, capture.text
    capture_body = capture.json()
    assert capture_body["mode"] == "aggressive"
    assert capture_body["structured_facts"] >= 2
    assert capture_body["global_memories"] >= 1
    assert "recent_global_memories" in capture_body


@pytest.mark.asyncio
async def test_admin_search_returns_trust_badges_and_filters_low_trust(api_client):
    client, settings = api_client
    auth = _auth(settings)
    ingest = await client.post(
        "/v1/ingest/message",
        headers=auth,
        json={
            "session_id": "trust-search",
            "platform": "cli",
            "role": "user",
            "content": "Mark confirmed the trusted support contact.",
        },
    )
    assert ingest.status_code == 200, ingest.text
    trusted = await client.post(
        "/v1/memory/remember",
        headers=auth,
        json={
            "kind": "fact",
            "subject": "support contact trusted",
            "predicate": "email",
            "object": "trusted@example.com",
            "statement": "The support contact email is trusted@example.com.",
            "topic": "trust-search",
            "source_event_ids": [ingest.json()["event_id"]],
            "metadata": {"confirmed_by": "user"},
        },
    )
    assert trusted.status_code == 200, trusted.text
    weak = await client.post(
        "/v1/memory/remember",
        headers=auth,
        json={
            "kind": "fact",
            "subject": "support contact weak",
            "predicate": "email",
            "object": "weak@example.com",
            "statement": "The support contact email is weak@example.com.",
            "topic": "trust-search",
            "metadata": {"source_type": "summary", "inferred": True},
            "confidence": 0.1,
        },
    )
    assert weak.status_code == 200, weak.text

    all_hits = await client.post(
        "/v1/admin/search",
        headers=auth,
        json={"target_type": "fact", "query": "support contact email", "limit": 20},
    )
    assert all_hits.status_code == 200, all_hits.text
    hits = all_hits.json()["hits"]
    assert {hit["trust_label"] for hit in hits} >= {"high", "low"}
    assert all("trust_score" in hit and "trust_explanation" in hit for hit in hits)

    low = await client.post(
        "/v1/admin/search",
        headers=auth,
        json={
            "target_type": "fact",
            "query": "support contact email",
            "trust_filter": "low_trust",
            "limit": 20,
        },
    )
    assert low.status_code == 200, low.text
    low_body = low.json()
    assert low_body["total"] == 1
    assert low_body["hits"][0]["id"] == weak.json()["id"]
    assert low_body["hits"][0]["trust_label"] in {"low", "very_low"}


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
async def test_admin_version_diff_and_rollback(api_client):
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
            "statement": "Shannon auth is API key",
        },
    )
    assert first.status_code == 200, first.text
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

    diff = await client.post(
        "/v1/admin/versions/diff",
        headers=auth,
        json={
            "target_type": "fact",
            "target_id": fact_id,
            "from_version": 1,
            "to_version": 2,
        },
    )
    assert diff.status_code == 200, diff.text
    diff_body = diff.json()
    assert diff_body["target_id"] == fact_id
    assert diff_body["changes"]["status"] == {"from": "active", "to": "superseded"}
    assert diff_body["changes"]["superseded_by"]["to"] == corr.json()["replacement_id"]

    rollback = await client.post(
        "/v1/admin/versions/rollback",
        headers=auth,
        json={
            "target_type": "fact",
            "target_id": fact_id,
            "to_version": 1,
            "reason": "operator test rollback",
        },
    )
    assert rollback.status_code == 200, rollback.text
    rollback_body = rollback.json()
    assert rollback_body["rolled_back"] is True
    assert rollback_body["new_version"] == 3

    detail = await client.post(
        "/v1/admin/detail",
        headers=auth,
        json={"target_type": "fact", "target_id": fact_id},
    )
    assert detail.status_code == 200, detail.text
    item = detail.json()["item"]
    assert item["status"] == "active"
    assert item["superseded_by"] is None
    assert item["statement"] == "Shannon auth is API key"


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
