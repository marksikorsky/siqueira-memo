"""Hermes session/FTS importer. Plan §33.7.

Reads one of:

* a JSONL export with one message per line;
* a Hermes SQLite SessionDB with ``messages`` table (``session_id``, ``role``,
  ``content``, ``created_at``).

In both paths each message is converted to a ``MessageIngestIn`` with
``source = hermes_session_import`` and fed through the ingest service, which
takes care of dedupe, redaction, and downstream chunking jobs. Messages that
already exist (matched by ``profile_id``/``session_id``/``role``/
``content_hash``/``source``) are treated as duplicates and skipped.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.logging import get_logger
from siqueira_memo.models.constants import (
    ALL_ROLES,
    MESSAGE_SOURCE_HERMES_SESSION_IMPORT,
    ROLE_USER,
)
from siqueira_memo.schemas.ingest import MessageIngestIn
from siqueira_memo.services.ingest_service import IngestService

log = get_logger(__name__)


@dataclass
class ImportedMessage:
    session_id: str
    role: str
    content: str
    created_at: datetime | None = None
    platform: str = "hermes"
    metadata: dict[str, Any] | None = None


def iter_jsonl(path: Path) -> Iterator[ImportedMessage]:
    default_session_id = _default_session_id_from_path(path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                doc = json.loads(stripped)
            except json.JSONDecodeError:
                log.warning("hermes_session.bad_jsonl_line")
                continue
            yield _row_to_message(doc, default_session_id=default_session_id)


def iter_sqlite(path: Path) -> Iterator[ImportedMessage]:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        created_expr = "created_at" if "created_at" in columns else "timestamp AS created_at"
        metadata_expr = "metadata" if "metadata" in columns else "NULL AS metadata"
        platform_expr = "platform" if "platform" in columns else "'hermes' AS platform"
        query = (
            f"SELECT session_id, role, content, {created_expr}, {metadata_expr}, {platform_expr} "
            "FROM messages ORDER BY "
            + ("created_at ASC" if "created_at" in columns else "timestamp ASC")
        )
        for row in conn.execute(query):
            doc = dict(row)
            yield _row_to_message(doc)
    finally:
        conn.close()


def _default_session_id_from_path(path: Path) -> str:
    stem = path.stem
    return stem.removeprefix("session_") or "imported"


def _row_to_message(doc: dict[str, Any], *, default_session_id: str = "imported") -> ImportedMessage:
    created = doc.get("created_at")
    if isinstance(created, (int, float)):
        created_dt: datetime | None = datetime.fromtimestamp(created, tz=UTC)
    elif isinstance(created, str):
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            created_dt = None
    else:
        created_dt = None
    metadata = doc.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {"raw": metadata}
    return ImportedMessage(
        session_id=str(doc.get("session_id") or default_session_id),
        role=str(doc.get("role", ROLE_USER)),
        content=str(doc.get("content", "")),
        created_at=created_dt,
        platform=str(doc.get("platform", "hermes")),
        metadata=metadata or {},
    )


@dataclass
class SessionImporter:
    ingest: IngestService

    async def import_iter(
        self, session: AsyncSession, rows: Iterable[ImportedMessage]
    ) -> dict[str, int]:
        imported = 0
        duplicates = 0
        errors = 0
        skipped = 0
        for row in rows:
            if not row.content.strip():
                skipped += 1
                continue
            if row.role not in ALL_ROLES:
                skipped += 1
                continue
            payload = MessageIngestIn(
                session_id=row.session_id,
                platform=row.platform or "hermes",
                role=row.role,
                content=row.content,
                created_at=row.created_at,
                source=MESSAGE_SOURCE_HERMES_SESSION_IMPORT,
                metadata=dict(row.metadata or {}),
            )
            try:
                result = await self.ingest.ingest_message(session, payload)
                if result.duplicate:
                    duplicates += 1
                else:
                    imported += 1
            except Exception:  # pragma: no cover
                log.exception("hermes_session.import_failed")
                errors += 1
        return {
            "imported": imported,
            "duplicates": duplicates,
            "skipped": skipped,
            "errors": errors,
        }
