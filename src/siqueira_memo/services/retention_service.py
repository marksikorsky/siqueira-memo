"""Retention policy enforcer + audit helpers. Plan §9 / §25.

The service never deletes raw messages unless the policy explicitly allows it.
Retrieval logs and worker logs roll off after configurable windows; audit
events stay forever (plan §9.2).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.config import Settings, get_settings
from siqueira_memo.logging import get_logger
from siqueira_memo.models import MemoryEvent, RetrievalLog

log = get_logger(__name__)


@dataclass
class RetentionReport:
    retrieval_logs_deleted: int = 0
    worker_logs_deleted: int = 0
    cutoffs: dict[str, datetime] = field(default_factory=dict)


@dataclass
class RetentionService:
    profile_id: str | None = None
    settings: Settings | None = None

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()

    async def enforce(self, session: AsyncSession) -> RetentionReport:
        assert self.settings is not None
        report = RetentionReport()
        now = datetime.now(UTC)

        retrieval_cutoff = now - timedelta(days=self.settings.retention_retrieval_logs_days)
        report.cutoffs["retrieval_logs"] = retrieval_cutoff
        stmt = delete(RetrievalLog).where(RetrievalLog.created_at < retrieval_cutoff)
        if self.profile_id is not None:
            stmt = stmt.where(RetrievalLog.profile_id == self.profile_id)
        result = await session.execute(stmt)
        report.retrieval_logs_deleted = getattr(result, "rowcount", 0) or 0

        log.info(
            "retention.enforced",
            extra={
                "profile_id": self.profile_id,
                "retrieval_deleted": report.retrieval_logs_deleted,
                "cutoff_retrieval": retrieval_cutoff.isoformat(),
            },
        )
        return report


@dataclass
class AuditEntry:
    id: uuid.UUID
    event_type: str
    target_type: str | None
    target_id: str | None
    actor: str
    created_at: datetime
    reason: str | None
    mode: str | None


class AuditLog:
    """Convenience reader over ``memory_events`` for deletion/correction audit.

    The raw events remain the source of truth — this class only filters and
    shapes them for human-visible audit endpoints. It never returns deleted
    content; plan §9.2 requires that audit retains metadata only.
    """

    def __init__(self, profile_id: str) -> None:
        self.profile_id = profile_id

    async def fetch_deletion_events(
        self, session: AsyncSession, *, since: datetime | None = None, limit: int = 100
    ) -> list[AuditEntry]:
        stmt = (
            select(MemoryEvent)
            .where(MemoryEvent.profile_id == self.profile_id)
            .where(
                MemoryEvent.event_type.in_(
                    [
                        "memory_deleted",
                        "fact_invalidated",
                        "decision_superseded",
                        "user_correction_received",
                    ]
                )
            )
            .order_by(MemoryEvent.created_at.desc())
            .limit(limit)
        )
        if since is not None:
            stmt = stmt.where(MemoryEvent.created_at >= since)
        rows = (await session.execute(stmt)).scalars().all()
        entries: list[AuditEntry] = []
        for row in rows:
            payload = row.payload or {}
            entries.append(
                AuditEntry(
                    id=row.id,
                    event_type=row.event_type,
                    target_type=payload.get("target_type")
                    or _target_type_from_payload(payload),
                    target_id=payload.get("target_id")
                    or payload.get("fact_id")
                    or payload.get("decision_id"),
                    actor=row.actor,
                    created_at=row.created_at,
                    reason=payload.get("reason"),
                    mode=payload.get("mode"),
                )
            )
        return entries


def _target_type_from_payload(payload: dict[str, Any]) -> str | None:
    if "fact_id" in payload:
        return "fact"
    if "decision_id" in payload:
        return "decision"
    return None
