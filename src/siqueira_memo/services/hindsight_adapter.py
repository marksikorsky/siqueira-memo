"""Hindsight import adapter — offline/backfill only. Plan §6.2 / §8 / §33.7.

This module MUST NEVER be imported into the live MemoryProvider runtime as a
fallback source. Hindsight integration is strictly offline import.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.logging import get_logger
from siqueira_memo.models import Fact, MemoryEvent, Message
from siqueira_memo.models.constants import (
    EVENT_TYPE_HINDSIGHT_IMPORTED,
    MESSAGE_SOURCE_HINDSIGHT_IMPORT,
    ROLE_USER,
    STATUS_CANDIDATE,
    TRUST_SECONDARY,
)
from siqueira_memo.services.redaction_service import RedactionService
from siqueira_memo.utils.canonical import content_hash, fact_canonical_key

log = get_logger(__name__)


@dataclass
class HindsightRecord:
    kind: str  # "message" | "fact" | "memory"
    content: str
    metadata: dict[str, Any]
    created_at: datetime | None = None


@dataclass
class ImportSummary:
    imported_events: int = 0
    imported_messages: int = 0
    imported_fact_candidates: int = 0
    skipped: int = 0


def iter_hindsight_export(path: Path) -> Iterator[HindsightRecord]:
    """Iterate over a JSONL Hindsight export produced by ``hindsight export``.

    The adapter is tolerant of shape changes: it only requires a ``kind`` and
    ``content`` field and treats the rest as metadata.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                log.warning("hindsight.export.bad_jsonl_line")
                continue
            kind = doc.get("kind") or doc.get("type") or "memory"
            content = doc.get("content") or doc.get("text") or ""
            created = doc.get("created_at")
            created_dt = None
            if isinstance(created, (int, float)):
                created_dt = datetime.fromtimestamp(created, tz=UTC)
            elif isinstance(created, str):
                try:
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except ValueError:
                    created_dt = None
            metadata = {k: v for k, v in doc.items() if k not in {"kind", "type", "content", "text"}}
            yield HindsightRecord(
                kind=kind,
                content=str(content),
                metadata=metadata,
                created_at=created_dt,
            )


@dataclass
class HindsightAdapter:
    profile_id: str
    redaction: RedactionService | None = None

    def __post_init__(self) -> None:
        self.redaction = self.redaction or RedactionService()

    async def import_records(
        self,
        session: AsyncSession,
        records: Iterable[HindsightRecord],
        *,
        session_id: str = "hindsight-import",
    ) -> ImportSummary:
        summary = ImportSummary()
        assert self.redaction is not None
        for record in records:
            if not record.content.strip():
                summary.skipped += 1
                continue
            redaction = self.redaction.redact(record.content)
            event_id = uuid.uuid4()
            session.add(
                MemoryEvent(
                    id=event_id,
                    event_type=EVENT_TYPE_HINDSIGHT_IMPORTED,
                    source="hindsight_import",
                    actor="import_job",
                    profile_id=self.profile_id,
                    session_id=session_id,
                    payload={
                        "event_type": EVENT_TYPE_HINDSIGHT_IMPORTED,
                        "source_id": record.metadata.get("id", str(event_id)),
                        "trust_level": TRUST_SECONDARY,
                        "requires_verification": True,
                    },
                )
            )
            summary.imported_events += 1

            if record.kind in {"message", "chat", "user_message", "assistant_message"}:
                c_hash = content_hash(record.content)
                existing = (
                    await session.execute(
                        select(Message).where(
                            Message.profile_id == self.profile_id,
                            Message.session_id == session_id,
                            Message.content_hash == c_hash,
                            Message.source == MESSAGE_SOURCE_HINDSIGHT_IMPORT,
                        )
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    summary.skipped += 1
                    continue
                role = record.metadata.get("role", ROLE_USER)
                session.add(
                    Message(
                        id=uuid.uuid4(),
                        event_id=event_id,
                        profile_id=self.profile_id,
                        session_id=session_id,
                        platform="hindsight",
                        role=role,
                        content_raw=record.content,
                        content_redacted=redaction.redacted,
                        content_hash=c_hash,
                        source=MESSAGE_SOURCE_HINDSIGHT_IMPORT,
                        extra_metadata={
                            **record.metadata,
                            "trust_level": TRUST_SECONDARY,
                            "requires_verification": True,
                        },
                        created_at=record.created_at or datetime.now(UTC),
                    )
                )
                summary.imported_messages += 1

            elif record.kind in {"fact", "memory", "note"}:
                # Create as candidate fact — not active.
                subject = record.metadata.get("subject") or "hindsight"
                predicate = record.metadata.get("predicate") or "note"
                obj = record.metadata.get("object") or record.content[:240]
                key = fact_canonical_key(
                    subject, predicate, obj, profile_id=self.profile_id
                )
                session.add(
                    Fact(
                        id=uuid.uuid4(),
                        profile_id=self.profile_id,
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        statement=redaction.redacted,
                        canonical_key=key,
                        project=record.metadata.get("project"),
                        topic=record.metadata.get("topic"),
                        confidence=float(record.metadata.get("confidence", 0.4)),
                        status=STATUS_CANDIDATE,
                        source_event_ids=[event_id],
                        source_message_ids=[],
                        extractor_name="hindsight_adapter",
                        extractor_version="1",
                        prompt_version="0",
                        model_provider="import",
                        model_name="hindsight",
                        source_scope="import",
                        schema_version="v1",
                        extra_metadata={
                            **record.metadata,
                            "trust_level": TRUST_SECONDARY,
                            "requires_verification": True,
                        },
                    )
                )
                summary.imported_fact_candidates += 1
            else:
                summary.skipped += 1
        await session.flush()
        return summary
