"""Explainable trust/source reputation scoring. Roadmap Phase 8."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.models import Decision, Fact, MemoryConflict, MemoryEvent
from siqueira_memo.models.constants import (
    CONFLICT_STATUS_OPEN,
    STATUS_ACTIVE,
    STATUS_DELETED,
    STATUS_INVALIDATED,
    STATUS_SUPERSEDED,
)

MemoryTarget = Fact | Decision
TargetType = Literal["fact", "decision"]
FeedbackValue = Literal["useful", "wrong", "stale", "duplicate"]


@dataclass(frozen=True)
class TrustAssessment:
    """Computed trust score plus an audit-friendly explanation."""

    target_type: TargetType
    target_id: uuid.UUID
    trust_score: float
    trust_label: str
    explanation: str
    factors: dict[str, float]


class TrustService:
    """Compute and record trust signals for facts/decisions.

    The first Phase 8 slice keeps trust derived/explainable and stores admin
    feedback in ``extra_metadata`` to avoid a migration until the UI contract
    settles. Feedback still writes an append-only ``MemoryEvent`` audit row.
    """

    def __init__(self, *, profile_id: str, actor: str = "trust_service") -> None:
        self.profile_id = profile_id
        self.actor = actor

    async def score_memory(
        self, session: AsyncSession, target_type: TargetType, target: MemoryTarget
    ) -> TrustAssessment:
        target = await self._load_target(session, target_type, target.id)
        conflict_count = await self._open_conflict_count(session, target_type, target.id)
        return self.estimate_memory(target_type, target, open_conflict_count=conflict_count)

    async def score_memories(
        self, session: AsyncSession, target_type: TargetType, targets: list[MemoryTarget]
    ) -> dict[uuid.UUID, TrustAssessment]:
        """Bulk score already-loaded memories without N+1 target reloads/conflict queries."""
        if not targets:
            return {}
        target_ids = [target.id for target in targets]
        conflict_counts = await self._open_conflict_counts(session, target_type, target_ids)
        return {
            target.id: self.estimate_memory(
                target_type,
                target,
                open_conflict_count=conflict_counts.get(target.id, 0),
            )
            for target in targets
            if target.profile_id == self.profile_id
        }

    @classmethod
    def estimate_memory(
        cls,
        target_type: TargetType,
        target: MemoryTarget,
        *,
        open_conflict_count: int = 0,
    ) -> TrustAssessment:
        metadata = dict(target.extra_metadata or {})
        confidence = cls._confidence(target)
        factors: dict[str, float] = {
            "base": 0.32,
            "confidence": 0.22 * confidence,
            "source_backed": 0.14 if target.source_event_ids else 0.0,
            "user_confirmed": 0.12 if _is_user_confirmed(metadata) else 0.0,
            "recency": 0.05 * _recency_signal(target),
            "feedback": _feedback_signal(metadata),
            "status_penalty": _status_penalty(target.status),
            "open_conflict_penalty": -0.12 * min(3, open_conflict_count),
            "summary_or_import_penalty": _summary_or_import_penalty(target, metadata),
        }
        score = max(0.0, min(1.0, sum(factors.values())))
        label = _label(score)
        explanation = _explanation(label, factors, open_conflict_count=open_conflict_count)
        return TrustAssessment(
            target_type=target_type,
            target_id=target.id,
            trust_score=round(score, 4),
            trust_label=label,
            explanation=explanation,
            factors={key: round(value, 4) for key, value in factors.items()},
        )

    async def record_feedback(
        self,
        session: AsyncSession,
        *,
        target_type: TargetType,
        target_id: uuid.UUID,
        feedback: FeedbackValue,
        reason: str | None = None,
    ) -> MemoryTarget:
        if feedback not in {"useful", "wrong", "stale", "duplicate"}:
            raise ValueError(f"unsupported trust feedback: {feedback}")
        target = await self._load_target(session, target_type, target_id)
        metadata = dict(target.extra_metadata or {})
        entries = list(metadata.get("trust_feedback") or [])
        entries.append(
            {
                "feedback": feedback,
                "reason": (reason or "")[:512],
                "actor": self.actor,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        metadata["trust_feedback"] = entries[-20:]
        target.extra_metadata = metadata
        session.add(
            MemoryEvent(
                id=uuid.uuid4(),
                event_type="trust_feedback_recorded",
                source="admin",
                actor=self.actor,
                profile_id=self.profile_id,
                payload={
                    "target_type": target_type,
                    "target_id": str(target_id),
                    "feedback": feedback,
                    "reason": (reason or "")[:512],
                },
            )
        )
        await session.flush()
        return target

    async def _load_target(
        self, session: AsyncSession, target_type: TargetType, target_id: uuid.UUID
    ) -> MemoryTarget:
        model = Fact if target_type == "fact" else Decision
        target = await session.get(model, target_id)
        if target is None:
            raise ValueError(f"{target_type} not found: {target_id}")
        target = cast(MemoryTarget, target)
        await session.refresh(target)
        if target.profile_id != self.profile_id:
            raise ValueError(f"{target_type} not found: {target_id}")
        return target

    async def _open_conflict_count(
        self, session: AsyncSession, target_type: TargetType, target_id: uuid.UUID
    ) -> int:
        return (await self._open_conflict_counts(session, target_type, [target_id])).get(target_id, 0)

    async def _open_conflict_counts(
        self, session: AsyncSession, target_type: TargetType, target_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, int]:
        if not target_ids:
            return {}
        left_rows = (
            await session.execute(
                select(MemoryConflict.left_id, func.count())
                .where(
                    MemoryConflict.profile_id == self.profile_id,
                    MemoryConflict.status == CONFLICT_STATUS_OPEN,
                    MemoryConflict.left_type == target_type,
                    MemoryConflict.left_id.in_(target_ids),
                )
                .group_by(MemoryConflict.left_id)
            )
        ).all()
        right_rows = (
            await session.execute(
                select(MemoryConflict.right_id, func.count())
                .where(
                    MemoryConflict.profile_id == self.profile_id,
                    MemoryConflict.status == CONFLICT_STATUS_OPEN,
                    MemoryConflict.right_type == target_type,
                    MemoryConflict.right_id.in_(target_ids),
                )
                .group_by(MemoryConflict.right_id)
            )
        ).all()
        counts: dict[uuid.UUID, int] = {}
        for target_id, count in [*left_rows, *right_rows]:
            counts[target_id] = counts.get(target_id, 0) + int(count)
        return counts

    @staticmethod
    def _confidence(target: MemoryTarget) -> float:
        if isinstance(target, Fact):
            return max(0.0, min(1.0, float(target.confidence or 0.0)))
        metadata = target.extra_metadata or {}
        return max(0.0, min(1.0, float(metadata.get("confidence") or 0.0)))


def _is_user_confirmed(metadata: dict[str, Any]) -> bool:
    return metadata.get("confirmed_by") == "user" or metadata.get("source_type") == "user"


def _recency_signal(target: MemoryTarget) -> float:
    anchor = getattr(target, "updated_at", None) or getattr(target, "created_at", None)
    if isinstance(target, Decision):
        anchor = target.decided_at or anchor
    if anchor is None:
        return 0.0
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=UTC)
    age_days = max(0, (datetime.now(UTC) - anchor).days)
    if age_days <= 30:
        return 1.0
    if age_days <= 180:
        return 0.6
    if age_days <= 365:
        return 0.3
    return 0.0


def _feedback_signal(metadata: dict[str, Any]) -> float:
    score = 0.0
    for entry in metadata.get("trust_feedback") or []:
        feedback = entry.get("feedback") if isinstance(entry, dict) else None
        if feedback == "useful":
            score += 0.08
        elif feedback == "wrong":
            score -= 0.35
        elif feedback == "stale":
            score -= 0.18
        elif feedback == "duplicate":
            score -= 0.12
    return max(-0.45, min(0.2, score))


def _status_penalty(status: str) -> float:
    if status == STATUS_ACTIVE:
        return 0.0
    if status == STATUS_SUPERSEDED:
        return -0.22
    if status == STATUS_INVALIDATED:
        return -0.35
    if status == STATUS_DELETED:
        return -0.5
    return -0.08


def _summary_or_import_penalty(target: MemoryTarget, metadata: dict[str, Any]) -> float:
    source_type = str(metadata.get("source_type") or "").lower()
    extractor = str(getattr(target, "extractor_name", "") or "").lower()
    if metadata.get("inferred") is True:
        return -0.08
    if "summary" in source_type or "summary" in extractor or "import" in extractor:
        return -0.06
    return 0.0


def _label(score: float) -> str:
    if score >= 0.78:
        return "high"
    if score >= 0.52:
        return "medium"
    if score >= 0.28:
        return "low"
    return "very_low"


def _explanation(label: str, factors: dict[str, float], *, open_conflict_count: int) -> str:
    positives = [name for name, value in factors.items() if value > 0.001 and name != "base"]
    penalties = [name for name, value in factors.items() if value < -0.001]
    parts = [f"trust={label}"]
    if positives:
        parts.append("positive: " + ", ".join(positives[:4]))
    if penalties:
        parts.append("penalty: " + ", ".join(penalties[:4]))
    if open_conflict_count:
        parts.append(f"open_conflicts={open_conflict_count}")
    return "; ".join(parts)
