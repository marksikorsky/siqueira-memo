"""Hermes MemoryProvider plugin tests. Plan §32.7 / §33."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from siqueira_memo.config import settings_for_tests
from siqueira_memo.db import create_all_for_tests, dispose_engines, drop_all_for_tests
from siqueira_memo.hermes_provider.provider import SiqueiraMemoProvider
from siqueira_memo.hermes_provider.tools import TOOL_NAMES, tool_schemas
from siqueira_memo.models import Chunk, Decision, Fact, MemoryEvent, Message, SessionSummary
from siqueira_memo.models.constants import STATUS_ACTIVE
from siqueira_memo.workers.jobs import register_default_handlers, set_worker_settings
from siqueira_memo.workers.queue import MemoryJobQueue, set_default_queue


@pytest.fixture
async def provider():
    settings = settings_for_tests()
    await create_all_for_tests(settings)
    queue = MemoryJobQueue()
    set_default_queue(queue)
    set_worker_settings(settings)

    prov = SiqueiraMemoProvider()
    prov._settings = settings
    prov.initialize(session_id="test-session", hermes_home="/tmp/hermes", agent_context="primary")
    try:
        yield prov, settings, queue
    finally:
        set_worker_settings(None)
        set_default_queue(None)
        await drop_all_for_tests(settings)
        await dispose_engines()


def test_is_available_is_local():
    assert SiqueiraMemoProvider().is_available() in {True, False}


def test_tool_schemas_are_siqueira_prefixed():
    schemas = tool_schemas()
    names = {t["name"] for t in schemas}
    assert names == set(TOOL_NAMES)
    for name in names:
        assert name.startswith("siqueira_memory_")


def test_tool_schemas_are_strict():
    for tool in tool_schemas():
        assert tool["parameters"]["type"] == "object"
        assert tool["parameters"].get("additionalProperties") is False
        # Hermes MemoryProvider expects OpenAI-style `parameters`; keep
        # `input_schema` as a compatibility alias only.
        assert tool["input_schema"] == tool["parameters"]


def test_remember_tool_schema_exposes_required_arguments():
    remember = next(t for t in tool_schemas() if t["name"] == "siqueira_memory_remember")
    assert remember["parameters"]["required"] == ["kind", "statement"]
    assert "kind" in remember["parameters"]["properties"]
    assert "statement" in remember["parameters"]["properties"]


def test_recall_tool_schema_defaults_to_trusted_internal_secret_recall():
    recall = next(t for t in tool_schemas() if t["name"] == "siqueira_memory_recall")
    allow_secret = recall["parameters"]["properties"]["allow_secret_recall"]
    assert allow_secret["default"] is True
    assert "trusted" in allow_secret["description"].lower()


@pytest.mark.asyncio
async def test_handle_tool_call_returns_json_string(provider):
    prov, _settings, _queue = provider
    raw = prov.handle_tool_call("siqueira_memory_recall", {"query": "anything", "mode": "fast"})
    data = json.loads(raw)
    assert data["ok"] is True
    assert data["tool"] == "siqueira_memory_recall"
    assert data["result"]["mode"] == "fast"


@pytest.mark.asyncio
async def test_handle_tool_call_defaults_to_trusted_internal_raw_secret_recall(provider):
    prov, settings, _queue = provider
    raw_secret = "sk-proj-" + "h" * 40
    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        event = MemoryEvent(
            event_type="fact_extracted",
            source="test",
            actor="assistant",
            profile_id=prov._profile_id,
            payload={"event_type": "fact_extracted", "fact_id": str(uuid.uuid4())},
        )
        session.add(event)
        await session.flush()
        session.add(
            Fact(
                profile_id=prov._profile_id,
                subject="OpenAI provider key",
                predicate="stored_as_secret",
                object=raw_secret,
                statement=f"OpenAI provider key is {raw_secret}.",
                canonical_key="fact-secret-openai-provider",
                status=STATUS_ACTIVE,
                confidence=0.99,
                source_event_ids=[event.id],
                topic="secrets",
                extractor_name="manual",
                created_at=datetime.now(UTC),
                extra_metadata={
                    "sensitivity": "secret",
                    "masked_preview": "OpenAI provider key is sk-proj-...hhhh.",
                    "secret_value": raw_secret,
                },
            )
        )
        await session.commit()

    raw = prov.handle_tool_call("siqueira_memory_recall", {"query": "OpenAI provider key", "mode": "balanced"})
    data = json.loads(raw)
    assert data["ok"] is True
    payload = json.dumps(data["result"])
    assert raw_secret in payload


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_json(provider):
    prov, _settings, _queue = provider
    raw = prov.handle_tool_call("unknown_tool", {})
    data = json.loads(raw)
    assert data["ok"] is False


@pytest.mark.asyncio
async def test_remember_tool_persists_fact(provider):
    prov, settings, _queue = provider
    raw = prov.handle_tool_call(
        "siqueira_memory_remember",
        {
            "kind": "fact",
            "subject": "siqueira-memo",
            "predicate": "primary_integration",
            "object": "MemoryProvider plugin",
            "statement": "Siqueira Memo is Hermes MemoryProvider plugin.",
            "confidence": 0.95,
        },
    )
    data = json.loads(raw)
    assert data["ok"] is True
    assert data["result"]["kind"] == "fact"

    from siqueira_memo.db import get_session_factory
    factory = get_session_factory(settings)
    async with factory() as session:
        events = (
            await session.execute(
                select(MemoryEvent).where(MemoryEvent.event_type == "fact_extracted")
            )
        ).scalars().all()
        assert events


@pytest.mark.asyncio
async def test_sync_turn_is_non_blocking(provider):
    prov, _settings, queue = provider
    prov.sync_turn("hello assistant", "hi user", session_id="s1")
    assert queue.pending() >= 1


@pytest.mark.asyncio
async def test_sync_turn_handler_persists_redacted_messages_chunks_and_memory(provider):
    prov, settings, queue = provider
    register_default_handlers(queue)

    prov.sync_turn(
        "Марк хочет чтобы Siqueira сохраняла почти всё полезное обсуждение.",
        "Решение: включаем aggressive memory capture для Siqueira.",
        session_id="s-aggressive",
    )
    drained = await queue.drain()
    assert drained >= 3

    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        messages = (
            await session.execute(
                select(Message)
                .where(Message.session_id == "s-aggressive")
                .order_by(Message.created_at.asc())
            )
        ).scalars().all()
        assert [m.role for m in messages] == ["user", "assistant"]
        assert all(m.content_redacted for m in messages)
        assert all(m.project is None for m in messages)

        chunks = (
            await session.execute(
                select(Chunk).where(Chunk.profile_id == prov._profile_id)  # noqa: SLF001
            )
        ).scalars().all()
        assert len(chunks) >= 2

        facts = (
            await session.execute(
                select(Fact).where(Fact.profile_id == prov._profile_id)  # noqa: SLF001
            )
        ).scalars().all()
        decisions = (
            await session.execute(
                select(Decision).where(Decision.profile_id == prov._profile_id)  # noqa: SLF001
            )
        ).scalars().all()
        assert any("aggressive memory capture" in d.decision for d in decisions)
        assert facts or decisions


@pytest.mark.asyncio
async def test_sync_turn_promotes_useful_link_based_architecture_analysis(provider):
    prov, settings, queue = provider
    register_default_handlers(queue)

    prov.sync_turn(
        "https://github.com/example/research-system\nhttps://arxiv.org/abs/2601.00001\nИзучи, можно ли использовать в нашем проекте?",
        "Вердикт: использовать стоит как отдельный research layer. Архитектура: добавить background worker, curated read-only tools, source-backed claims, conflict scan, and avoid unsafe local exec.",
        session_id="s-link-analysis",
    )
    drained = await queue.drain()
    assert drained >= 3

    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        decisions = (
            await session.execute(
                select(Decision)
                .where(Decision.profile_id == prov._profile_id)  # noqa: SLF001
                .where(Decision.decision.ilike("%research layer%"))
            )
        ).scalars().all()
        assert decisions


@pytest.mark.asyncio
async def test_sync_turn_uses_llm_classifier_for_non_marker_useful_memory(provider, monkeypatch):
    import siqueira_memo.workers.jobs as jobs
    from siqueira_memo.services.memory_capture_classifier import MemoryCaptureDecision

    prov, settings, queue = provider
    register_default_handlers(queue)
    set_worker_settings(settings.model_copy(update={"memory_capture_llm_enabled": True}))

    def fake_classify(*_args, **_kwargs):
        return MemoryCaptureDecision(
            save=True,
            kind="fact",
            statement="Mark's cold standby host for quick rollback is node amber-17 at 10.44.0.17.",
            subject="cold standby host",
            predicate="is",
            object="node amber-17 at 10.44.0.17",
            project="infrastructure",
            topic="servers",
            confidence=0.91,
            rationale="Stable infrastructure inventory useful for future operations.",
        )

    monkeypatch.setattr(jobs, "classify_turn_memory", fake_classify)
    prov.sync_turn(
        "на всякий: холодный резерв для быстрого отката — amber-17, адрес 10.44.0.17",
        "Принял.",
        session_id="s-llm-useful",
    )
    await queue.drain()

    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        facts = (
            await session.execute(
                select(Fact)
                .where(Fact.profile_id == prov._profile_id)  # noqa: SLF001
                .where(Fact.statement.ilike("%amber-17%"))
            )
        ).scalars().all()
        assert facts
        assert facts[0].project == "infrastructure"
        assert facts[0].topic == "servers"


@pytest.mark.asyncio
async def test_sync_turn_respects_llm_classifier_ignore(provider, monkeypatch):
    import siqueira_memo.workers.jobs as jobs
    from siqueira_memo.services.memory_capture_classifier import MemoryCaptureDecision

    prov, settings, queue = provider
    register_default_handlers(queue)
    set_worker_settings(settings.model_copy(update={"memory_capture_llm_enabled": True}))

    monkeypatch.setattr(
        jobs,
        "classify_turn_memory",
        lambda *_args, **_kwargs: MemoryCaptureDecision(
            save=False,
            kind="fact",
            statement="",
            confidence=0.99,
            rationale="Casual acknowledgement/no durable content.",
        ),
    )
    prov.sync_turn("ок", "👍", session_id="s-llm-ignore")
    await queue.drain()

    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        facts = (
            await session.execute(select(Fact).where(Fact.profile_id == prov._profile_id))  # noqa: SLF001
        ).scalars().all()
        decisions = (
            await session.execute(select(Decision).where(Decision.profile_id == prov._profile_id))  # noqa: SLF001
        ).scalars().all()
        audit_events = (
            await session.execute(
                select(MemoryEvent).where(MemoryEvent.event_type == "capture_classifier_skip")
            )
        ).scalars().all()
        assert not facts
        assert not decisions
        assert audit_events
        assert audit_events[0].payload["reason"] == "Casual acknowledgement/no durable content."


@pytest.mark.asyncio
async def test_sync_turn_persists_multiple_llm_candidates_and_audits(provider, monkeypatch):
    import siqueira_memo.workers.jobs as jobs
    from siqueira_memo.schemas.memory_capture import MemoryCandidate, MemoryCaptureResult

    prov, settings, queue = provider
    register_default_handlers(queue)
    set_worker_settings(settings.model_copy(update={"memory_capture_llm_enabled": True}))

    monkeypatch.setattr(
        jobs,
        "classify_turn_memory",
        lambda *_args, **_kwargs: MemoryCaptureResult(
            classifier_model="test-capture-model",
            prompt_version="capture-v2-test",
            candidates=[
                MemoryCandidate(
                    action="auto_save",
                    kind="fact",
                    statement="Amber-17 is Mark's cold standby host at 10.44.0.17.",
                    subject="amber-17",
                    predicate="role",
                    object="cold standby host at 10.44.0.17",
                    project="infrastructure",
                    topic="servers",
                    confidence=0.94,
                    importance=0.92,
                    rationale="Stable host inventory.",
                ),
                MemoryCandidate(
                    action="auto_save",
                    kind="decision",
                    statement="Use source-backed capture audit before expanding retrieval fusion.",
                    project="siqueira-memo",
                    topic="roadmap",
                    confidence=0.91,
                    importance=0.9,
                    rationale="Explicit roadmap ordering.",
                ),
                MemoryCandidate(
                    action="skip_noise",
                    kind="fact",
                    statement="Assistant acknowledged the instruction.",
                    confidence=0.8,
                    importance=0.1,
                    rationale="No durable content.",
                ),
            ],
        ),
    )

    prov.sync_turn(
        "Запомни: amber-17 — cold standby. Решение: сначала capture audit.",
        "Принял.",
        session_id="s-multi-candidate",
    )
    await queue.drain()

    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        facts = (
            await session.execute(
                select(Fact)
                .where(Fact.profile_id == prov._profile_id)  # noqa: SLF001
                .where(Fact.statement.ilike("%Amber-17%"))
            )
        ).scalars().all()
        decisions = (
            await session.execute(
                select(Decision)
                .where(Decision.profile_id == prov._profile_id)  # noqa: SLF001
                .where(Decision.decision.ilike("%capture audit%"))
            )
        ).scalars().all()
        audit_events = (
            await session.execute(
                select(MemoryEvent).where(
                    MemoryEvent.session_id == "s-multi-candidate",
                    MemoryEvent.event_type.in_(
                        [
                            "capture_classifier_called",
                            "capture_candidates_extracted",
                            "capture_candidate_auto_saved",
                            "capture_classifier_skip",
                        ]
                    ),
                )
            )
        ).scalars().all()

        assert facts
        assert decisions
        assert any(e.event_type == "capture_candidates_extracted" and e.payload["candidate_count"] == 3 for e in audit_events)
        assert sum(1 for e in audit_events if e.event_type == "capture_candidate_auto_saved") == 2
        assert any(e.event_type == "capture_classifier_skip" for e in audit_events)


@pytest.mark.asyncio
async def test_sync_turn_secret_candidate_is_tagged_redacted_and_not_dropped(provider, monkeypatch):
    import siqueira_memo.workers.jobs as jobs
    from siqueira_memo.schemas.memory_capture import MemoryCandidate, MemoryCaptureResult

    prov, settings, queue = provider
    register_default_handlers(queue)
    set_worker_settings(settings.model_copy(update={"memory_capture_llm_enabled": True}))

    raw_secret = "sk-proj-" + "a" * 40
    monkeypatch.setattr(
        jobs,
        "classify_turn_memory",
        lambda *_args, **_kwargs: MemoryCaptureResult(
            classifier_model="test-capture-model",
            prompt_version="capture-v2-test",
            candidates=[
                MemoryCandidate(
                    action="auto_save",
                    kind="secret",
                    statement=f"OpenAI key for staging tests is {raw_secret}.",
                    subject="OpenAI staging key",
                    predicate="stored_as_secret",
                    object="staging tests credential",
                    project="siqueira-memo",
                    topic="secrets",
                    confidence=0.96,
                    importance=0.85,
                    sensitivity="secret",
                    risk="high",
                    rationale="Operational credential; store masked until explicit secret policy lands.",
                )
            ],
        ),
    )

    prov.sync_turn("Запомни staging OpenAI key", "Сохранил masked.", session_id="s-secret-candidate")
    await queue.drain()

    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        facts = (
            await session.execute(
                select(Fact)
                .where(Fact.profile_id == prov._profile_id)  # noqa: SLF001
                .where(Fact.topic == "secrets")
            )
        ).scalars().all()
        secret_audit_events = (
            await session.execute(
                select(MemoryEvent).where(
                    MemoryEvent.session_id == "s-secret-candidate",
                    MemoryEvent.event_type == "capture_secret_candidate_saved",
                )
            )
        ).scalars().all()

        assert facts
        assert facts[0].extra_metadata["capture_kind"] == "secret"
        assert facts[0].extra_metadata["sensitivity"] == "secret"
        assert raw_secret not in facts[0].statement
        assert "[SECRET_REF:" in facts[0].statement
        assert secret_audit_events


@pytest.mark.asyncio
async def test_sync_turn_audits_llm_fallback_when_classifier_returns_none(provider, monkeypatch):
    import siqueira_memo.workers.jobs as jobs

    prov, settings, queue = provider
    register_default_handlers(queue)
    set_worker_settings(settings.model_copy(update={"memory_capture_llm_enabled": True}))
    monkeypatch.setattr(jobs, "classify_turn_memory", lambda *_args, **_kwargs: None)

    prov.sync_turn(
        "Решение: fallback audit должен сохраняться при отказе classifier.",
        "Ок.",
        session_id="s-fallback-audit",
    )
    await queue.drain()

    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        decisions = (
            await session.execute(select(Decision).where(Decision.profile_id == prov._profile_id))  # noqa: SLF001
        ).scalars().all()
        fallback_events = (
            await session.execute(
                select(MemoryEvent).where(
                    MemoryEvent.session_id == "s-fallback-audit",
                    MemoryEvent.event_type == "capture_classifier_fallback",
                )
            )
        ).scalars().all()

        assert decisions
        assert fallback_events
        assert fallback_events[0].payload["fallback_reason"] == "classifier_unavailable_or_failed"


@pytest.mark.asyncio
async def test_sync_turn_promotes_tailscale_server_inventory(provider):
    prov, settings, queue = provider
    register_default_handlers(queue)

    prov.sync_turn(
        "Скинь мне список серверов с таилскейл",
        "Вот Linux-серверы в твоём Tailscale: vmi3206734/hermes 100.98.80.48 online; personal 100.83.150.74 online; clawik 100.84.141.32 online; draft 100.74.70.18 online; DNS suffix tail4e3571.ts.net.",
        session_id="s-tailscale-inventory",
    )
    drained = await queue.drain()
    assert drained >= 3

    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        messages = (
            await session.execute(
                select(Message)
                .where(Message.session_id == "s-tailscale-inventory")
                .order_by(Message.created_at.asc())
            )
        ).scalars().all()
        assert [m.project for m in messages] == ["infrastructure", "infrastructure"]
        assert [m.topic for m in messages] == ["tailscale", "tailscale"]

        facts = (
            await session.execute(
                select(Fact)
                .where(Fact.profile_id == prov._profile_id)  # noqa: SLF001
                .where(Fact.statement.ilike("%tail4e3571.ts.net%"))
            )
        ).scalars().all()
        assert facts
        assert facts[0].project == "infrastructure"
        assert facts[0].topic == "tailscale"


@pytest.mark.asyncio
async def test_queue_prefetch_warms_context_cache(provider):
    prov, _settings, queue = provider
    register_default_handlers(queue)
    raw = prov.handle_tool_call(
        "siqueira_memory_remember",
        {
            "kind": "fact",
            "subject": "siqueira-memo",
            "predicate": "captures",
            "object": "aggressive memory capture",
            "statement": "Siqueira Memo captures aggressive memory turns.",
            "confidence": 0.95,
        },
    )
    assert json.loads(raw)["ok"] is True

    prov.queue_prefetch("aggressive memory capture", session_id="s1")
    assert queue.pending() >= 1
    await queue.drain()

    cached = prov.prefetch("aggressive memory capture", session_id="s1")
    assert "prefetch cache cold" not in " ".join(cached.get("warnings", []))
    assert cached["facts"]


@pytest.mark.asyncio
async def test_on_memory_write_mirrors(provider):
    prov, settings, queue = provider
    register_default_handlers(queue)
    prov.on_memory_write("add", "memory", "Mark prefers concise answers")
    assert queue.pending() >= 1
    await queue.drain()

    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        events = (
            await session.execute(
                select(MemoryEvent).where(MemoryEvent.event_type == "builtin_memory_mirror")
            )
        ).scalars().all()
        assert events


@pytest.mark.asyncio
async def test_on_delegation_records(provider):
    prov, settings, queue = provider
    register_default_handlers(queue)
    prov.on_delegation("research X", "summary of X", child_session_id="child-1")
    assert queue.pending() >= 1
    await queue.drain()

    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        events = (
            await session.execute(
                select(MemoryEvent).where(MemoryEvent.event_type == "delegation_observed")
            )
        ).scalars().all()
        assert events


@pytest.mark.asyncio
async def test_on_pre_compress_records_hook_event(provider):
    prov, settings, queue = provider
    register_default_handlers(queue)
    ret = prov.on_pre_compress([{"role": "user", "content": "blah"}])
    assert ret == ""
    assert queue.pending() >= 1
    await queue.drain()

    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        events = (
            await session.execute(
                select(MemoryEvent).where(MemoryEvent.event_type == "pre_compress_extract")
            )
        ).scalars().all()
        assert events


@pytest.mark.asyncio
async def test_on_pre_compress_persists_redacted_transcript_summary(provider):
    prov, settings, queue = provider
    register_default_handlers(queue)

    prov.on_pre_compress(
        [
            {"role": "user", "content": "надо сохранить Tail compaction secret sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
            {
                "role": "assistant",
                "content": "Решение: при compaction сохранять transcript tail, tool results и summary.",
            },
            {"role": "tool", "content": "Adapter health OK at https://shella.app/health"},
        ]
    )
    await queue.drain()

    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        events = (
            await session.execute(
                select(MemoryEvent).where(MemoryEvent.event_type == "pre_compress_extract")
            )
        ).scalars().all()
        assert events
        payload = events[0].payload
        assert payload["transcript_tail_count"] == 3
        tail_text = json.dumps(payload["transcript_tail"], ensure_ascii=False)
        assert "sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in tail_text
        assert "shella.app/health" in tail_text

        summaries = (
            await session.execute(
                select(SessionSummary).where(SessionSummary.session_id == "test-session")
            )
        ).scalars().all()
        assert summaries
        assert "compaction transcript tail" in summaries[0].summary_short.lower()
        assert "shella.app/health" in summaries[0].summary_long


@pytest.mark.asyncio
async def test_on_session_end_creates_session_summary(provider):
    prov, settings, queue = provider
    register_default_handlers(queue)
    prov.sync_turn("remember this useful session detail", "captured useful answer", session_id="session-end")
    prov.on_session_end([])
    await queue.drain()

    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        summaries = (
            await session.execute(
                select(SessionSummary).where(SessionSummary.session_id == "test-session")
            )
        ).scalars().all()
        assert summaries


@pytest.mark.asyncio
async def test_on_session_end_uses_supplied_messages_when_db_session_empty(provider):
    prov, settings, queue = provider
    register_default_handlers(queue)

    prov.on_session_end(
        [
            {"role": "user", "content": "Проверь почему giant session under-captured."},
            {
                "role": "assistant",
                "content": "Диагноз: sync_turn не видит tool-heavy work; нужен compression transcript capture.",
            },
        ]
    )
    await queue.drain()

    from siqueira_memo.db import get_session_factory

    factory = get_session_factory(settings)
    async with factory() as session:
        summaries = (
            await session.execute(
                select(SessionSummary).where(SessionSummary.session_id == "test-session")
            )
        ).scalars().all()
        assert summaries
        assert "tool-heavy work" in summaries[0].summary_long


def test_is_hermes_auxiliary_compaction():
    assert SiqueiraMemoProvider.is_hermes_auxiliary_compaction(
        "[CONTEXT COMPACTION — REFERENCE ONLY]\nGoal: ..."
    )
    assert SiqueiraMemoProvider.is_hermes_auxiliary_compaction(
        "[CONTEXT COMPACTION] body"
    )
    assert not SiqueiraMemoProvider.is_hermes_auxiliary_compaction(
        "regular user message"
    )


def test_system_prompt_block_is_returned():
    prov = SiqueiraMemoProvider()
    text = prov.system_prompt_block()
    assert "Siqueira Memo" in text
    assert "precedence" in text.lower()


@pytest.mark.asyncio
async def test_register_calls_ctx_register_memory_provider():
    import importlib.util
    from pathlib import Path

    plugin_path = (
        Path(__file__).resolve().parents[2]
        / "plugins"
        / "memory"
        / "siqueira-memo"
        / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location("siqueira_memo_plugin", plugin_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    recorded: list = []

    class Ctx:
        def register_memory_provider(self, provider):
            recorded.append(provider)

    module.register(Ctx())
    assert len(recorded) == 1
    assert recorded[0].name == "siqueira-memo"


@pytest.mark.asyncio
async def test_run_uses_stable_event_loop_when_called_from_async_context():
    from siqueira_memo.hermes_provider import provider as provider_module

    seen_loop: asyncio.AbstractEventLoop | None = None

    async def bind_to_first_loop() -> str:
        nonlocal seen_loop
        loop = asyncio.get_running_loop()
        if seen_loop is None:
            seen_loop = loop
            return "ok"
        assert loop is seen_loop
        return "ok"

    assert provider_module._run(bind_to_first_loop()) == "ok"
    assert provider_module._run(bind_to_first_loop()) == "ok"


@pytest.mark.asyncio
async def test_forget_tool_returns_event_id(provider):
    prov, settings, _queue = provider
    remember_raw = prov.handle_tool_call(
        "siqueira_memory_remember",
        {
            "kind": "fact",
            "subject": "x",
            "predicate": "y",
            "object": "z",
            "statement": "x y z",
        },
    )
    remember = json.loads(remember_raw)
    fact_id = remember["result"]["id"]
    forget_raw = prov.handle_tool_call(
        "siqueira_memory_forget",
        {"target_type": "fact", "target_id": fact_id, "mode": "soft"},
    )
    forget = json.loads(forget_raw)
    assert forget["ok"] is True
    assert forget["result"]["invalidated_facts"] == 1
