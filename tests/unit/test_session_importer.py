"""Hermes session importer tests. Plan §33.7."""

from __future__ import annotations

import json
import sqlite3

import pytest
from sqlalchemy import select

from siqueira_memo.models import Message
from siqueira_memo.models.constants import MESSAGE_SOURCE_HERMES_SESSION_IMPORT
from siqueira_memo.services.ingest_service import IngestService
from siqueira_memo.services.session_importer import (
    SessionImporter,
    iter_jsonl,
    iter_sqlite,
)


@pytest.mark.asyncio
async def test_importer_from_jsonl(tmp_path, db, session, queue):
    path = tmp_path / "sessions.jsonl"
    rows = [
        {"session_id": "s1", "role": "user", "content": "hello", "created_at": "2024-01-01T00:00:00Z"},
        {"session_id": "s1", "role": "assistant", "content": "hi back"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    ingest = IngestService(queue=queue, profile_id="p1")
    importer = SessionImporter(ingest=ingest)
    result = await importer.import_iter(session, iter_jsonl(path))
    assert result["imported"] == 2
    assert result["duplicates"] == 0

    messages = (await session.execute(select(Message))).scalars().all()
    assert [m.source for m in messages] == [MESSAGE_SOURCE_HERMES_SESSION_IMPORT] * 2


def test_iter_jsonl_uses_filename_session_id_when_rows_omit_it(tmp_path):
    path = tmp_path / "20260426_072220_89fd3a.jsonl"
    rows = [
        {"role": "user", "content": "continue adapter work"},
        {"role": "tool", "content": "curl returned health OK"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    imported = list(iter_jsonl(path))

    assert [m.session_id for m in imported] == ["20260426_072220_89fd3a"] * 2
    assert [m.role for m in imported] == ["user", "tool"]


@pytest.mark.asyncio
async def test_importer_from_sqlite(tmp_path, db, session, queue):
    db_path = tmp_path / "hermes.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT, "
        "created_at REAL, metadata TEXT)"
    )
    conn.execute(
        "INSERT INTO messages VALUES ('s1', 'user', 'imported hi', 1700000000.0, '{\"a\":1}')"
    )
    conn.commit()
    conn.close()

    ingest = IngestService(queue=queue, profile_id="p1")
    importer = SessionImporter(ingest=ingest)
    result = await importer.import_iter(session, iter_sqlite(db_path))
    assert result["imported"] == 1
    messages = (await session.execute(select(Message))).scalars().all()
    assert messages[0].content_raw == "imported hi"
    assert messages[0].extra_metadata.get("a") == 1
