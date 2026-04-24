"""Hindsight import adapter tests. Plan §6.2 / §8.1."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from siqueira_memo.models import Fact, MemoryEvent, Message
from siqueira_memo.models.constants import (
    EVENT_TYPE_HINDSIGHT_IMPORTED,
    MESSAGE_SOURCE_HINDSIGHT_IMPORT,
    STATUS_CANDIDATE,
    TRUST_SECONDARY,
)
from siqueira_memo.services.hindsight_adapter import (
    HindsightAdapter,
    HindsightRecord,
    iter_hindsight_export,
)


@pytest.mark.asyncio
async def test_import_marks_records_as_secondary(db, session):
    adapter = HindsightAdapter(profile_id="p1")
    records = [
        HindsightRecord(
            kind="message", content="Hello from Hindsight", metadata={"role": "user"}
        ),
        HindsightRecord(
            kind="fact",
            content="Shannon auth is api key",
            metadata={"subject": "shannon", "predicate": "primary_auth", "object": "api_key"},
        ),
    ]
    summary = await adapter.import_records(session, records)
    assert summary.imported_events == 2
    assert summary.imported_messages == 1
    assert summary.imported_fact_candidates == 1

    events = (
        await session.execute(
            select(MemoryEvent).where(MemoryEvent.event_type == EVENT_TYPE_HINDSIGHT_IMPORTED)
        )
    ).scalars().all()
    assert len(events) == 2

    msg = (await session.execute(select(Message))).scalars().one()
    assert msg.source == MESSAGE_SOURCE_HINDSIGHT_IMPORT
    assert msg.extra_metadata["trust_level"] == TRUST_SECONDARY

    fact = (await session.execute(select(Fact))).scalars().one()
    assert fact.status == STATUS_CANDIDATE
    assert fact.extra_metadata["requires_verification"] is True


@pytest.mark.asyncio
async def test_import_dedupes_messages(db, session):
    adapter = HindsightAdapter(profile_id="p1")
    record = HindsightRecord(kind="message", content="same message", metadata={"role": "user"})
    first = await adapter.import_records(session, [record])
    second = await adapter.import_records(session, [record])
    assert first.imported_messages == 1
    assert second.imported_messages == 0
    assert second.skipped >= 1


def test_iter_hindsight_export(tmp_path):
    path = tmp_path / "export.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"kind": "memory", "content": "x", "id": "1"}),
                json.dumps({"kind": "message", "content": "y", "created_at": "2024-01-02T00:00:00Z"}),
                "",  # blank
                "not-json",
            ]
        ),
        encoding="utf-8",
    )
    records = list(iter_hindsight_export(path))
    assert len(records) == 2
    assert records[0].kind == "memory"
    assert records[1].created_at.year == 2024
