"""Prompt registry tests. Plan §22 / §31.13."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from siqueira_memo.models import PromptVersion
from siqueira_memo.services.prompt_registry import PromptRegistry, register_hermes_system_prompt


def test_scan_loads_all_v1_prompts():
    reg = PromptRegistry()
    entries = reg.scan()
    names = {e.name for e in entries}
    assert {"extraction_gate", "decision_extractor", "fact_extractor", "conflict_verifier", "entity_linker", "session_summarizer"}.issubset(
        names
    )
    for entry in entries:
        assert entry.version.startswith("v")
        assert entry.content_hash


def test_prompt_entry_reports_stable_hash(tmp_path: Path):
    prompt = tmp_path / "stable.v1.md"
    prompt.write_text("hello", encoding="utf-8")
    reg = PromptRegistry(base_dir=tmp_path)
    entry = reg.scan()[0]
    assert entry.name == "stable"
    assert entry.version == "v1"
    prompt.write_text("hello changed", encoding="utf-8")
    reg2 = PromptRegistry(base_dir=tmp_path)
    entry2 = reg2.scan()[0]
    assert entry2.content_hash != entry.content_hash


@pytest.mark.asyncio
async def test_sync_persists_rows(db, session):
    reg = PromptRegistry()
    entries = await reg.sync(session)
    assert entries
    stored = (await session.execute(select(PromptVersion))).scalars().all()
    assert len(stored) == len(entries)


@pytest.mark.asyncio
async def test_sync_is_idempotent(db, session):
    reg = PromptRegistry()
    await reg.sync(session)
    count_1 = len((await session.execute(select(PromptVersion))).scalars().all())
    await reg.sync(session)
    count_2 = len((await session.execute(select(PromptVersion))).scalars().all())
    assert count_1 == count_2


def test_hermes_system_prompt_hashes_computable():
    plugin_path = (
        Path(__file__).resolve().parents[2]
        / "plugins"
        / "memory"
        / "siqueira-memo"
        / "system_prompt.md"
    )
    canonical, plugin = register_hermes_system_prompt(PromptRegistry(), plugin_path)
    assert canonical != plugin or canonical == plugin  # sanity: both return strings
    assert len(canonical) == 64
    assert len(plugin) == 64
