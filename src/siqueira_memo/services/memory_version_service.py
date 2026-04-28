"""Memory versioning, diffing, and rollback helpers."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, cast

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.models import (
    Decision,
    DecisionSource,
    Fact,
    FactSource,
    MemoryEvent,
    MemoryVersion,
    Message,
)
from siqueira_memo.models.constants import STATUS_ACTIVE
from siqueira_memo.services.secret_policy import (
    is_secret_metadata,
    masked_preview,
    sanitize_metadata,
)
from siqueira_memo.utils.canonical import decision_canonical_key, fact_canonical_key

VersionedTarget = Literal["fact", "decision"]
MAX_VERSION_REASON_LENGTH = 1024


def _bounded_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    text = str(reason)
    return text if len(text) <= MAX_VERSION_REASON_LENGTH else text[:MAX_VERSION_REASON_LENGTH]


@dataclass(frozen=True)
class VersionDiff:
    target_type: VersionedTarget
    target_id: uuid.UUID
    from_version: int
    to_version: int
    changes: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class RollbackResult:
    target_type: VersionedTarget
    target_id: uuid.UUID
    rolled_back: bool
    rollback_to_version: int
    new_version: int
    event_id: uuid.UUID


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _uuid(value: uuid.UUID | None) -> str | None:
    return str(value) if value is not None else None


def _uuid_list(values: list[uuid.UUID] | None) -> list[str]:
    return [str(value) for value in (values or [])]


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _parse_uuid(value: Any) -> uuid.UUID | None:
    if value in (None, ""):
        return None
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _parse_uuid_list(values: Any) -> list[uuid.UUID]:
    if not values:
        return []
    return [value if isinstance(value, uuid.UUID) else uuid.UUID(str(value)) for value in values]


def _loaded_attr(row: object, name: str) -> Any:
    """Read already-loaded ORM state without triggering async lazy IO."""
    return getattr(row, "__dict__", {}).get(name)


def snapshot_memory(row: Fact | Decision | None) -> dict[str, Any] | None:
    """Return a JSON-safe snapshot for version storage."""
    if row is None:
        return None
    if isinstance(row, Fact):
        return {
            "id": str(row.id),
            "target_type": "fact",
            "profile_id": row.profile_id,
            "subject": row.subject,
            "predicate": row.predicate,
            "object": row.object,
            "statement": row.statement,
            "canonical_key": row.canonical_key,
            "project": row.project,
            "topic": row.topic,
            "confidence": row.confidence,
            "status": row.status,
            "valid_from": _dt(row.valid_from),
            "valid_to": _dt(row.valid_to),
            "source_event_ids": _uuid_list(row.source_event_ids),
            "source_message_ids": _uuid_list(row.source_message_ids),
            "superseded_by": _uuid(row.superseded_by),
            "extractor_name": row.extractor_name,
            "extractor_version": row.extractor_version,
            "prompt_version": row.prompt_version,
            "model_provider": row.model_provider,
            "model_name": row.model_name,
            "source_scope": row.source_scope,
            "schema_version": row.schema_version,
            "extra_metadata": row.extra_metadata or {},
            "created_at": _dt(_loaded_attr(row, "created_at")),
            "updated_at": _dt(_loaded_attr(row, "updated_at")),
        }
    return {
        "id": str(row.id),
        "target_type": "decision",
        "profile_id": row.profile_id,
        "project": row.project,
        "topic": row.topic,
        "decision": row.decision,
        "context": row.context,
        "options_considered": row.options_considered or [],
        "rationale": row.rationale,
        "tradeoffs": row.tradeoffs or {},
        "canonical_key": row.canonical_key,
        "status": row.status,
        "reversible": row.reversible,
        "superseded_by": _uuid(row.superseded_by),
        "decided_at": _dt(row.decided_at),
        "source_event_ids": _uuid_list(row.source_event_ids),
        "source_message_ids": _uuid_list(row.source_message_ids),
        "extractor_name": row.extractor_name,
        "extractor_version": row.extractor_version,
        "prompt_version": row.prompt_version,
        "model_provider": row.model_provider,
        "model_name": row.model_name,
        "source_scope": row.source_scope,
        "schema_version": row.schema_version,
        "extra_metadata": row.extra_metadata or {},
        "created_at": _dt(_loaded_attr(row, "created_at")),
        "updated_at": _dt(_loaded_attr(row, "updated_at")),
    }


_TEXT_SNAPSHOT_FIELDS = {
    "statement",
    "object",
    "decision",
    "context",
    "rationale",
    "subject",
    "predicate",
}


def sanitize_snapshot_for_public(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a snapshot safe for ordinary admin/API display."""
    if snapshot is None:
        return None
    safe = dict(snapshot)
    metadata = dict(safe.get("extra_metadata") or {})
    safe["extra_metadata"] = sanitize_metadata(metadata)
    secret = is_secret_metadata(metadata)
    for key in _TEXT_SNAPSHOT_FIELDS:
        value = safe.get(key)
        if isinstance(value, str):
            safe[key] = masked_preview(value, metadata)
    if secret:
        safe["secret"] = True
        safe["masked"] = True
    return safe


def snapshot_for_hard_delete(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep hard-delete version tombstones without preserving recoverable content."""
    if snapshot is None:
        return None
    return {
        "id": snapshot.get("id"),
        "target_type": snapshot.get("target_type"),
        "profile_id": snapshot.get("profile_id"),
        "status": "hard_deleted",
        "hard_deleted": True,
        "masked": True,
    }


def _diff_snapshots(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    before = before or {}
    after = after or {}
    keys = sorted(set(before) | set(after))
    return {key: {"from": before.get(key), "to": after.get(key)} for key in keys if before.get(key) != after.get(key)}


def _target_model(target_type: str) -> type[Fact] | type[Decision]:
    if target_type == "fact":
        return Fact
    if target_type == "decision":
        return Decision
    raise ValueError(f"unsupported versioned target_type: {target_type}")


class MemoryVersionService:
    """Append-only version store with explicit diff and rollback operations."""

    async def record(
        self,
        session: AsyncSession,
        *,
        target_type: VersionedTarget,
        target_id: uuid.UUID,
        profile_id: str,
        operation: str,
        before_snapshot: dict[str, Any] | None,
        after_snapshot: dict[str, Any] | None,
        event_id: uuid.UUID | None = None,
        actor: str = "system",
        reason: str | None = None,
        rollback_to_version: int | None = None,
    ) -> MemoryVersion:
        latest = await session.execute(
            select(func.max(MemoryVersion.version)).where(
                MemoryVersion.profile_id == profile_id,
                MemoryVersion.target_type == target_type,
                MemoryVersion.target_id == target_id,
            )
        )
        next_version = int(latest.scalar_one() or 0) + 1
        before_snapshot = await self._with_source_links(session, target_type, before_snapshot)
        after_snapshot = await self._with_source_links(session, target_type, after_snapshot)
        row = MemoryVersion(
            profile_id=profile_id,
            target_type=target_type,
            target_id=target_id,
            version=next_version,
            operation=operation,
            actor=actor,
            reason=_bounded_reason(reason),
            event_id=event_id,
            rollback_to_version=rollback_to_version,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
        )
        session.add(row)
        await session.flush()
        return row

    async def diff(
        self,
        session: AsyncSession,
        target_type: VersionedTarget,
        target_id: uuid.UUID,
        from_version: int,
        to_version: int,
        *,
        profile_id: str,
    ) -> VersionDiff:
        from_row = await self._get_version(
            session, target_type, target_id, from_version, profile_id=profile_id
        )
        to_row = await self._get_version(
            session, target_type, target_id, to_version, profile_id=profile_id
        )
        return VersionDiff(
            target_type=target_type,
            target_id=target_id,
            from_version=from_version,
            to_version=to_version,
            changes=_diff_snapshots(
                sanitize_snapshot_for_public(from_row.after_snapshot),
                sanitize_snapshot_for_public(to_row.after_snapshot),
            ),
        )

    async def rollback(
        self,
        session: AsyncSession,
        *,
        target_type: VersionedTarget,
        target_id: uuid.UUID,
        to_version: int,
        profile_id: str,
        actor: str,
        reason: str | None = None,
    ) -> RollbackResult:
        await self._assert_not_hard_deleted(session, target_type, target_id, profile_id=profile_id)
        version_row = await self._get_version(
            session, target_type, target_id, to_version, profile_id=profile_id
        )
        target_snapshot = version_row.after_snapshot
        if not target_snapshot:
            raise ValueError("cannot rollback to an empty/deleted snapshot")
        await self._assert_no_active_canonical_conflict(
            session, target_type, target_id, target_snapshot, profile_id=profile_id
        )

        if target_type == "fact":
            current_fact = (
                await session.execute(
                    select(Fact).where(Fact.id == target_id, Fact.profile_id == profile_id)
                )
            ).scalar_one_or_none()
            before = snapshot_memory(current_fact)
        else:
            current_decision = (
                await session.execute(
                    select(Decision).where(
                        Decision.id == target_id, Decision.profile_id == profile_id
                    )
                )
            ).scalar_one_or_none()
            before = snapshot_memory(current_decision)
        restored = await self._restore_snapshot(session, target_type, target_snapshot)
        await session.flush()
        after = snapshot_memory(restored)

        event_id = uuid.uuid4()
        session.add(
            MemoryEvent(
                id=event_id,
                event_type="memory_rolled_back",
                source="version_api",
                actor=actor,
                profile_id=profile_id,
                payload={
                    "event_type": "memory_rolled_back",
                    "target_type": target_type,
                    "target_id": str(target_id),
                    "to_version": to_version,
                    "reason": reason,
                },
            )
        )
        await session.flush()
        version = await self.record(
            session,
            target_type=target_type,
            target_id=target_id,
            profile_id=profile_id,
            operation="rollback",
            before_snapshot=before,
            after_snapshot=after,
            event_id=event_id,
            actor=actor,
            reason=reason,
            rollback_to_version=to_version,
        )
        return RollbackResult(
            target_type=target_type,
            target_id=target_id,
            rolled_back=True,
            rollback_to_version=to_version,
            new_version=version.version,
            event_id=event_id,
        )

    async def _assert_not_hard_deleted(
        self,
        session: AsyncSession,
        target_type: VersionedTarget,
        target_id: uuid.UUID,
        *,
        profile_id: str,
    ) -> None:
        latest = (
            await session.execute(
                select(MemoryVersion)
                .where(
                    MemoryVersion.profile_id == profile_id,
                    MemoryVersion.target_type == target_type,
                    MemoryVersion.target_id == target_id,
                )
                .order_by(MemoryVersion.version.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if latest is not None and latest.operation == "hard_delete":
            raise ValueError(f"cannot rollback hard-deleted {target_type} {target_id}")

    async def scrub_target_history_for_hard_delete(
        self,
        session: AsyncSession,
        *,
        target_type: VersionedTarget,
        target_id: uuid.UUID,
        profile_id: str,
    ) -> None:
        versions = (
            await session.execute(
                select(MemoryVersion).where(
                    MemoryVersion.profile_id == profile_id,
                    MemoryVersion.target_type == target_type,
                    MemoryVersion.target_id == target_id,
                )
            )
        ).scalars().all()
        for version in versions:
            version.before_snapshot = snapshot_for_hard_delete(version.before_snapshot)
            version.after_snapshot = snapshot_for_hard_delete(version.after_snapshot)

    async def _with_source_links(
        self,
        session: AsyncSession,
        target_type: VersionedTarget,
        snapshot: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if snapshot is None or not snapshot.get("id"):
            return snapshot
        enriched = dict(snapshot)
        target_id = uuid.UUID(str(snapshot["id"]))
        if target_type == "fact":
            links = (
                await session.execute(
                    select(FactSource.event_id, FactSource.message_id).where(
                        FactSource.fact_id == target_id
                    )
                )
            ).all()
        else:
            links = (
                await session.execute(
                    select(DecisionSource.event_id, DecisionSource.message_id).where(
                        DecisionSource.decision_id == target_id
                    )
                )
            ).all()
        if links:
            enriched["source_links"] = [
                {"event_id": str(event_id), "message_id": _uuid(message_id)}
                for event_id, message_id in links
            ]
        return enriched

    async def _get_version(
        self,
        session: AsyncSession,
        target_type: VersionedTarget,
        target_id: uuid.UUID,
        version: int,
        *,
        profile_id: str,
    ) -> MemoryVersion:
        row = (
            await session.execute(
                select(MemoryVersion).where(
                    MemoryVersion.profile_id == profile_id,
                    MemoryVersion.target_type == target_type,
                    MemoryVersion.target_id == target_id,
                    MemoryVersion.version == version,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise ValueError(f"version {version} not found for {target_type} {target_id}")
        return row

    async def _assert_no_active_canonical_conflict(
        self,
        session: AsyncSession,
        target_type: VersionedTarget,
        target_id: uuid.UUID,
        snapshot: dict[str, Any],
        *,
        profile_id: str,
    ) -> None:
        if str(snapshot.get("status") or STATUS_ACTIVE) != STATUS_ACTIVE:
            return
        conflict_id: uuid.UUID | None
        if target_type == "fact":
            canonical_key = fact_canonical_key(
                str(snapshot["subject"]),
                str(snapshot["predicate"]),
                str(snapshot["object"]),
                project=cast(str | None, snapshot.get("project")),
                profile_id=profile_id,
            )
            fact_conflict = (
                await session.execute(
                    select(Fact).where(
                        Fact.profile_id == profile_id,
                        Fact.canonical_key == canonical_key,
                        Fact.status == STATUS_ACTIVE,
                        Fact.id != target_id,
                    )
                )
            ).scalar_one_or_none()
            conflict_id = fact_conflict.id if fact_conflict is not None else None
        else:
            canonical_key = decision_canonical_key(
                cast(str | None, snapshot.get("project")),
                str(snapshot["topic"]),
                str(snapshot["decision"]),
                profile_id=profile_id,
            )
            decision_conflict = (
                await session.execute(
                    select(Decision).where(
                        Decision.profile_id == profile_id,
                        Decision.canonical_key == canonical_key,
                        Decision.status == STATUS_ACTIVE,
                        Decision.id != target_id,
                    )
                )
            ).scalar_one_or_none()
            conflict_id = decision_conflict.id if decision_conflict is not None else None
        if conflict_id is not None:
            raise ValueError(
                f"rollback would conflict with active {target_type} {conflict_id}"
            )

    async def _restore_snapshot(
        self, session: AsyncSession, target_type: VersionedTarget, snapshot: dict[str, Any]
    ) -> Fact | Decision:
        if target_type == "fact":
            row = (
                await session.execute(select(Fact).where(Fact.id == uuid.UUID(snapshot["id"])))
            ).scalar_one_or_none()
            if row is None:
                row = Fact(id=uuid.UUID(snapshot["id"]))
                session.add(row)
            row.profile_id = str(snapshot["profile_id"])
            row.subject = str(snapshot["subject"])
            row.predicate = str(snapshot["predicate"])
            row.object = str(snapshot["object"])
            row.statement = str(snapshot["statement"])
            row.project = cast(str | None, snapshot.get("project"))
            row.topic = cast(str | None, snapshot.get("topic"))
            row.confidence = float(snapshot.get("confidence") or 0.0)
            row.status = str(snapshot.get("status") or STATUS_ACTIVE)
            row.valid_from = _parse_dt(snapshot.get("valid_from"))
            row.valid_to = _parse_dt(snapshot.get("valid_to"))
            row.source_event_ids = _parse_uuid_list(snapshot.get("source_event_ids"))
            row.source_message_ids = _parse_uuid_list(snapshot.get("source_message_ids"))
            row.superseded_by = _parse_uuid(snapshot.get("superseded_by"))
            row.extractor_name = str(snapshot.get("extractor_name") or "manual")
            row.extractor_version = str(snapshot.get("extractor_version") or "1")
            row.prompt_version = str(snapshot.get("prompt_version") or "v1")
            row.model_provider = str(snapshot.get("model_provider") or "manual")
            row.model_name = str(snapshot.get("model_name") or "manual")
            row.source_scope = str(snapshot.get("source_scope") or "message")
            row.schema_version = str(snapshot.get("schema_version") or "v1")
            row.extra_metadata = dict(snapshot.get("extra_metadata") or {})
            row.canonical_key = fact_canonical_key(
                row.subject, row.predicate, row.object, project=row.project, profile_id=row.profile_id
            )
            await self._restore_fact_sources(
                session,
                row.id,
                row.source_event_ids,
                cast(list[dict[str, Any]] | None, snapshot.get("source_links")),
                profile_id=row.profile_id,
            )
            return row

        decision_row = (
            await session.execute(select(Decision).where(Decision.id == uuid.UUID(snapshot["id"])))
        ).scalar_one_or_none()
        if decision_row is None:
            decision_row = Decision(id=uuid.UUID(snapshot["id"]))
            session.add(decision_row)
        decision_row.profile_id = str(snapshot["profile_id"])
        decision_row.project = cast(str | None, snapshot.get("project"))
        decision_row.topic = str(snapshot["topic"])
        decision_row.decision = str(snapshot["decision"])
        decision_row.context = str(snapshot.get("context") or "")
        decision_row.options_considered = list(snapshot.get("options_considered") or [])
        decision_row.rationale = str(snapshot.get("rationale") or "")
        decision_row.tradeoffs = dict(snapshot.get("tradeoffs") or {})
        decision_row.status = str(snapshot.get("status") or STATUS_ACTIVE)
        decision_row.reversible = bool(snapshot.get("reversible", True))
        decision_row.superseded_by = _parse_uuid(snapshot.get("superseded_by"))
        decision_row.decided_at = _parse_dt(snapshot.get("decided_at")) or datetime.now().astimezone()
        decision_row.source_event_ids = _parse_uuid_list(snapshot.get("source_event_ids"))
        decision_row.source_message_ids = _parse_uuid_list(snapshot.get("source_message_ids"))
        decision_row.extractor_name = str(snapshot.get("extractor_name") or "manual")
        decision_row.extractor_version = str(snapshot.get("extractor_version") or "1")
        decision_row.prompt_version = str(snapshot.get("prompt_version") or "v1")
        decision_row.model_provider = str(snapshot.get("model_provider") or "manual")
        decision_row.model_name = str(snapshot.get("model_name") or "manual")
        decision_row.source_scope = str(snapshot.get("source_scope") or "window")
        decision_row.schema_version = str(snapshot.get("schema_version") or "v1")
        decision_row.extra_metadata = dict(snapshot.get("extra_metadata") or {})
        decision_row.canonical_key = decision_canonical_key(
            decision_row.project,
            decision_row.topic,
            decision_row.decision,
            profile_id=decision_row.profile_id,
        )
        await self._restore_decision_sources(
            session,
            decision_row.id,
            decision_row.source_event_ids,
            cast(list[dict[str, Any]] | None, snapshot.get("source_links")),
            profile_id=decision_row.profile_id,
        )
        return decision_row

    async def _existing_event_ids(
        self, session: AsyncSession, event_ids: list[uuid.UUID], *, profile_id: str | None = None
    ) -> list[uuid.UUID]:
        if not event_ids:
            return []
        stmt = select(MemoryEvent.id).where(MemoryEvent.id.in_(event_ids))
        if profile_id is not None:
            stmt = stmt.where(MemoryEvent.profile_id == profile_id)
        rows = (await session.execute(stmt)).scalars().all()
        return list(rows)

    async def _valid_source_links(
        self, session: AsyncSession, links: list[dict[str, Any]], *, profile_id: str
    ) -> list[tuple[uuid.UUID, uuid.UUID | None]]:
        parsed_links: list[tuple[uuid.UUID, uuid.UUID | None]] = []
        event_ids: list[uuid.UUID] = []
        message_ids: list[uuid.UUID] = []
        for link in links:
            event_id = _parse_uuid(link.get("event_id"))
            if event_id is None:
                continue
            message_id = _parse_uuid(link.get("message_id"))
            parsed_links.append((event_id, message_id))
            event_ids.append(event_id)
            if message_id is not None:
                message_ids.append(message_id)
        if not parsed_links:
            return []
        valid_events = set(await self._existing_event_ids(session, event_ids, profile_id=profile_id))
        if message_ids:
            valid_messages = set(
                (
                    await session.execute(
                        select(Message.id).where(Message.id.in_(message_ids), Message.profile_id == profile_id)
                    )
                ).scalars().all()
            )
        else:
            valid_messages = set()
        return [
            (event_id, message_id if message_id in valid_messages else None)
            for event_id, message_id in parsed_links
            if event_id in valid_events
        ]

    async def _restore_fact_sources(
        self,
        session: AsyncSession,
        fact_id: uuid.UUID,
        event_ids: list[uuid.UUID],
        source_links: list[dict[str, Any]] | None = None,
        *,
        profile_id: str,
    ) -> None:
        await session.execute(delete(FactSource).where(FactSource.fact_id == fact_id))
        if source_links:
            for event_id, message_id in await self._valid_source_links(session, source_links, profile_id=profile_id):
                session.add(FactSource(fact_id=fact_id, event_id=event_id, message_id=message_id))
            return
        for event_id in await self._existing_event_ids(session, event_ids, profile_id=profile_id):
            session.add(FactSource(fact_id=fact_id, event_id=event_id, message_id=None))

    async def _restore_decision_sources(
        self,
        session: AsyncSession,
        decision_id: uuid.UUID,
        event_ids: list[uuid.UUID],
        source_links: list[dict[str, Any]] | None = None,
        *,
        profile_id: str,
    ) -> None:
        await session.execute(
            delete(DecisionSource).where(DecisionSource.decision_id == decision_id)
        )
        if source_links:
            for event_id, message_id in await self._valid_source_links(session, source_links, profile_id=profile_id):
                session.add(
                    DecisionSource(decision_id=decision_id, event_id=event_id, message_id=message_id)
                )
            return
        for event_id in await self._existing_event_ids(session, event_ids, profile_id=profile_id):
            session.add(
                DecisionSource(decision_id=decision_id, event_id=event_id, message_id=None)
            )
