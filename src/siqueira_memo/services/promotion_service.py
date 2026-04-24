"""Candidate → active promotion. Plan §31.7 / §33.12.

The service implements the documented lifecycle for facts and decisions:

```
candidate -> deduped          (equivalent active exists)
candidate -> promoted_active   (no collision, confidence above threshold)
candidate -> needs_review      (conflicting active exists)
candidate -> (unchanged)       (below promotion threshold)
```

Postgres acquires a transaction-scoped advisory lock keyed on the canonical
key (§33.12) so concurrent workers serialise promotion of equivalent rows.
The SQLite test path takes the equivalent Python-level lock for determinism.
"""

from __future__ import annotations

import asyncio
import enum
import uuid
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.logging import get_logger
from siqueira_memo.models import Decision, Fact, MemoryConflict
from siqueira_memo.models.constants import (
    CONFLICT_STATUS_NEEDS_REVIEW,
    STATUS_ACTIVE,
    STATUS_CANDIDATE,
    STATUS_DEDUPED,
    STATUS_NEEDS_REVIEW,
)
from siqueira_memo.utils.canonical import advisory_lock_key, normalize_text

log = get_logger(__name__)


MemoryLike = Fact | Decision


class PromotionOutcome(enum.StrEnum):
    PROMOTED = "promoted_active"
    DEDUPED = "deduped"
    NEEDS_REVIEW = "needs_review"
    BELOW_THRESHOLD = "below_threshold"
    ALREADY_ACTIVE = "already_active"


_locks: dict[int, asyncio.Lock] = {}


def _python_lock(key: int) -> asyncio.Lock:
    """Per-key asyncio lock used when Postgres advisory locks are unavailable."""
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    return lock


@dataclass
class PromotionService:
    profile_id: str
    promotion_threshold: float = 0.7

    async def promote(
        self, session: AsyncSession, candidate: MemoryLike
    ) -> PromotionOutcome:
        if candidate.status == STATUS_ACTIVE:
            return PromotionOutcome.ALREADY_ACTIVE
        if candidate.status != STATUS_CANDIDATE:
            # Already deduped/superseded/invalidated/etc.
            return PromotionOutcome.ALREADY_ACTIVE

        lock_key = advisory_lock_key(f"{self.profile_id}:{candidate.canonical_key}")
        dialect = session.bind.dialect.name if session.bind is not None else "sqlite"

        if dialect == "postgresql":
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:k)"), {"k": lock_key}
            )
            return await self._promote_locked(session, candidate)
        async with _python_lock(lock_key):
            return await self._promote_locked(session, candidate)

    async def _promote_locked(
        self, session: AsyncSession, candidate: MemoryLike
    ) -> PromotionOutcome:
        model = type(candidate)
        existing_row = (
            await session.execute(
                select(model)
                .where(model.profile_id == self.profile_id)
                .where(model.canonical_key == candidate.canonical_key)
                .where(model.status == STATUS_ACTIVE)
                .where(model.id != candidate.id)
            )
        ).scalar_one_or_none()
        existing: MemoryLike | None = existing_row  # type: ignore[assignment]

        if existing is not None:
            # Equivalent active exists — merge sources and deduplicate.
            merged_strs = sorted(
                {
                    str(x)
                    for x in (existing.source_event_ids or [])
                    + (candidate.source_event_ids or [])
                }
            )
            # Convert back to UUIDs for the Postgres UUID[] type. SQLite stores
            # them as JSON strings anyway.
            existing.source_event_ids = [uuid.UUID(x) for x in merged_strs]
            candidate.status = STATUS_DEDUPED
            log.info(
                "promotion.deduped",
                extra={
                    "profile_id": self.profile_id,
                    "canonical_key": candidate.canonical_key,
                    "kept_id": str(existing.id),
                    "dropped_id": str(candidate.id),
                },
            )
            await session.flush()
            return PromotionOutcome.DEDUPED

        conflicting = await self._find_conflicting_active(session, candidate)
        if conflicting is not None:
            candidate.status = STATUS_NEEDS_REVIEW
            session.add(
                MemoryConflict(
                    profile_id=self.profile_id,
                    conflict_type="fact_fact" if isinstance(candidate, Fact) else "decision_decision",
                    left_type="fact" if isinstance(candidate, Fact) else "decision",
                    left_id=conflicting.id,
                    right_type="fact" if isinstance(candidate, Fact) else "decision",
                    right_id=candidate.id,
                    severity="high",
                    status=CONFLICT_STATUS_NEEDS_REVIEW,
                    resolution_hint="candidate contradicts active; reviewer must resolve",
                    confidence=_candidate_confidence(candidate),
                    extra_metadata={"detector": "promotion_conflict"},
                )
            )
            await session.flush()
            log.info(
                "promotion.needs_review",
                extra={
                    "profile_id": self.profile_id,
                    "candidate_id": str(candidate.id),
                    "active_id": str(conflicting.id),
                },
            )
            return PromotionOutcome.NEEDS_REVIEW

        confidence = _candidate_confidence(candidate)
        if confidence < self.promotion_threshold:
            log.info(
                "promotion.below_threshold",
                extra={
                    "profile_id": self.profile_id,
                    "candidate_id": str(candidate.id),
                    "confidence": confidence,
                    "threshold": self.promotion_threshold,
                },
            )
            return PromotionOutcome.BELOW_THRESHOLD

        candidate.status = STATUS_ACTIVE
        await session.flush()
        log.info(
            "promotion.promoted",
            extra={
                "profile_id": self.profile_id,
                "candidate_id": str(candidate.id),
                "canonical_key": candidate.canonical_key,
            },
        )
        return PromotionOutcome.PROMOTED

    async def _find_conflicting_active(
        self, session: AsyncSession, candidate: MemoryLike
    ) -> MemoryLike | None:
        if isinstance(candidate, Fact):
            fact_rows = (
                (
                    await session.execute(
                        select(Fact)
                        .where(Fact.profile_id == self.profile_id)
                        .where(Fact.status == STATUS_ACTIVE)
                        .where(Fact.subject == candidate.subject)
                        .where(Fact.predicate == candidate.predicate)
                    )
                )
                .scalars()
                .all()
            )
            for fact_row in fact_rows:
                if fact_row.id == candidate.id:
                    continue
                if normalize_text(fact_row.object) != normalize_text(candidate.object):
                    return fact_row
            return None
        # Decision: same normalised topic+project, different decision wording.
        decision_rows = (
            (
                await session.execute(
                    select(Decision)
                    .where(Decision.profile_id == self.profile_id)
                    .where(Decision.status == STATUS_ACTIVE)
                    .where(Decision.topic == candidate.topic)
                )
            )
            .scalars()
            .all()
        )
        for decision_row in decision_rows:
            if decision_row.id == candidate.id:
                continue
            if normalize_text(decision_row.project or "") != normalize_text(
                candidate.project or ""
            ):
                continue
            if normalize_text(decision_row.decision) != normalize_text(candidate.decision):
                return decision_row
        return None


def _candidate_confidence(candidate: MemoryLike) -> float:
    if isinstance(candidate, Fact):
        return float(candidate.confidence or 0.0)
    meta = candidate.extra_metadata or {}
    try:
        return float(meta.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
