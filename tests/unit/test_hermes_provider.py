"""Hermes MemoryProvider plugin tests. Plan §32.7 / §33."""

from __future__ import annotations

import asyncio
import json

import pytest
from sqlalchemy import select

from siqueira_memo.config import settings_for_tests
from siqueira_memo.db import create_all_for_tests, dispose_engines, drop_all_for_tests
from siqueira_memo.hermes_provider.provider import SiqueiraMemoProvider
from siqueira_memo.hermes_provider.tools import TOOL_NAMES, tool_schemas
from siqueira_memo.models import Chunk, Decision, Fact, MemoryEvent, Message
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
        assert tool["input_schema"]["type"] == "object"
        assert tool["input_schema"].get("additionalProperties") is False


@pytest.mark.asyncio
async def test_handle_tool_call_returns_json_string(provider):
    prov, _settings, _queue = provider
    raw = prov.handle_tool_call("siqueira_memory_recall", {"query": "anything", "mode": "fast"})
    data = json.loads(raw)
    assert data["ok"] is True
    assert data["tool"] == "siqueira_memory_recall"
    assert data["result"]["mode"] == "fast"


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
async def test_queue_prefetch_enqueues(provider):
    prov, _settings, queue = provider
    prov.queue_prefetch("test query", session_id="s1")
    assert queue.pending() >= 1


@pytest.mark.asyncio
async def test_on_memory_write_mirrors(provider):
    prov, _settings, queue = provider
    prov.on_memory_write("add", "memory", "Mark prefers concise answers")
    assert queue.pending() >= 1


@pytest.mark.asyncio
async def test_on_delegation_records(provider):
    prov, _settings, queue = provider
    prov.on_delegation("research X", "summary of X", child_session_id="child-1")
    assert queue.pending() >= 1


@pytest.mark.asyncio
async def test_on_pre_compress_enqueues_but_returns_empty_string(provider):
    prov, _settings, queue = provider
    ret = prov.on_pre_compress([{"role": "user", "content": "blah"}])
    assert ret == ""
    assert queue.pending() >= 1


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
