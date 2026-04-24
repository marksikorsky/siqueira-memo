"""Conflict detection + resolution service. Plan §21.

Pipeline per plan §31.5:

1. Rule prefilter — group facts by (subject, predicate, project) and decisions
   by (project, topic). Fast, runs entirely in Python after a single load.
2. Polarity detector — rules that infer negation between decision pairs (``use``
   vs ``do not use``, ``primary`` vs ``secondary`` …).
3. Temporal filter — `tstzrange`-style overlap check for facts with explicit
   validity windows.
4. LLM verifier — not wired in this module; ambiguous pairs are left as
   ``needs_review`` and can be verified by a worker later. The hook is in place
   on ``_score_pair`` via ``ConflictService.llm_verifier``.
5. Persistence — ``MemoryConflict`` rows are created or updated idempotently.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.logging import get_logger
from siqueira_memo.models import Decision, Fact, MemoryConflict, MemoryEvent
from siqueira_memo.models.constants import (
    CONFLICT_STATUS_AUTO_RESOLVED,
    CONFLICT_STATUS_IGNORED,
    CONFLICT_STATUS_NEEDS_REVIEW,
    CONFLICT_STATUS_OPEN,
    EVENT_TYPE_DECISION_SUPERSEDED,
    EVENT_TYPE_FACT_INVALIDATED,
    STATUS_ACTIVE,
    STATUS_SUPERSEDED,
)
from siqueira_memo.utils.canonical import normalize_text

log = get_logger(__name__)


class _LLMVerifier(Protocol):
    def verify(self, left: str, right: str) -> dict[str, Any]:
        ...


@dataclass
class ConflictService:
    profile_id: str
    max_pairs: int = 20
    llm_verifier: _LLMVerifier | None = None

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------
    async def scan(self, session: AsyncSession) -> list[MemoryConflict]:
        """Run detection and persist new/updated conflicts. Returns the list."""
        pairs: list[_CandidatePair] = []

        pairs.extend(await self._decision_decision_pairs(session))
        pairs.extend(await self._fact_fact_pairs(session))
        pairs.extend(await self._decision_fact_pairs(session))

        # Candidate narrowing (plan §31.5).
        pairs = pairs[: self.max_pairs]

        persisted: list[MemoryConflict] = []
        for pair in pairs:
            existing = await self._existing_conflict(session, pair)
            if existing is not None:
                # Refresh severity/confidence in case evidence strengthened.
                if existing.severity != pair.severity:
                    existing.severity = pair.severity
                if existing.confidence < pair.confidence:
                    existing.confidence = pair.confidence
                persisted.append(existing)
                continue
            row = MemoryConflict(
                id=uuid.uuid4(),
                profile_id=self.profile_id,
                conflict_type=pair.conflict_type,
                left_type=pair.left_type,
                left_id=pair.left_id,
                right_type=pair.right_type,
                right_id=pair.right_id,
                severity=pair.severity,
                status=CONFLICT_STATUS_OPEN if not pair.requires_review else CONFLICT_STATUS_NEEDS_REVIEW,
                resolution_hint=pair.hint,
                confidence=pair.confidence,
                extra_metadata={"detector": pair.detector},
            )
            session.add(row)
            persisted.append(row)

        await session.flush()
        log.info(
            "conflict.scan",
            extra={
                "profile_id": self.profile_id,
                "candidates": len(pairs),
                "stored": len(persisted),
            },
        )
        return persisted

    async def _existing_conflict(
        self, session: AsyncSession, pair: _CandidatePair
    ) -> MemoryConflict | None:
        stmt = (
            select(MemoryConflict)
            .where(MemoryConflict.profile_id == self.profile_id)
            .where(MemoryConflict.conflict_type == pair.conflict_type)
            .where(
                or_(
                    and_(MemoryConflict.left_id == pair.left_id, MemoryConflict.right_id == pair.right_id),
                    and_(MemoryConflict.left_id == pair.right_id, MemoryConflict.right_id == pair.left_id),
                )
            )
        )
        return (await session.execute(stmt.limit(1))).scalar_one_or_none()

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    async def _decision_decision_pairs(
        self, session: AsyncSession
    ) -> list[_CandidatePair]:
        rows = (
            await session.execute(
                select(Decision).where(
                    Decision.profile_id == self.profile_id,
                    Decision.status == STATUS_ACTIVE,
                )
            )
        ).scalars().all()
        by_topic: dict[tuple[str, str], list[Decision]] = {}
        for row in rows:
            key = (normalize_text(row.project or "__global__"), normalize_text(row.topic))
            by_topic.setdefault(key, []).append(row)

        pairs: list[_CandidatePair] = []
        for group in by_topic.values():
            if len(group) < 2:
                continue
            # Newest first; pair every newer with older that contradicts it.
            ordered = sorted(group, key=lambda d: d.decided_at, reverse=True)
            for i, newer in enumerate(ordered):
                for older in ordered[i + 1 :]:
                    if not _polarity_conflict(newer.decision, older.decision):
                        continue
                    pairs.append(
                        _CandidatePair(
                            conflict_type="decision_decision",
                            left_type="decision",
                            left_id=older.id,
                            right_type="decision",
                            right_id=newer.id,
                            severity="high",
                            confidence=0.9,
                            hint="newer decision supersedes older",
                            detector="polarity_rule",
                        )
                    )
        return pairs

    async def _fact_fact_pairs(self, session: AsyncSession) -> list[_CandidatePair]:
        rows = (
            await session.execute(
                select(Fact).where(
                    Fact.profile_id == self.profile_id,
                    Fact.status == STATUS_ACTIVE,
                )
            )
        ).scalars().all()
        groups: dict[tuple[str, str, str], list[Fact]] = {}
        for row in rows:
            key = (
                normalize_text(row.subject),
                normalize_text(row.predicate),
                normalize_text(row.project or "__global__"),
            )
            groups.setdefault(key, []).append(row)

        pairs: list[_CandidatePair] = []
        for group in groups.values():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if normalize_text(a.object) == normalize_text(b.object):
                        continue
                    if not _temporal_overlap(a, b):
                        continue
                    older, newer = sorted([a, b], key=lambda f: f.valid_from or f.created_at)
                    pairs.append(
                        _CandidatePair(
                            conflict_type="fact_fact",
                            left_type="fact",
                            left_id=older.id,
                            right_type="fact",
                            right_id=newer.id,
                            severity="medium",
                            confidence=0.8,
                            hint="divergent facts with overlapping validity",
                            detector="same_subject_predicate",
                        )
                    )
        return pairs

    async def _decision_fact_pairs(
        self, session: AsyncSession
    ) -> list[_CandidatePair]:
        """Cheap rule: decisions using negation of a fact's phrase.

        We look for active decisions whose normalized phrase inverts an active
        fact's statement for the same project. This is intentionally narrow;
        ambiguous cases are ``needs_review``.
        """
        decisions = (
            await session.execute(
                select(Decision).where(
                    Decision.profile_id == self.profile_id,
                    Decision.status == STATUS_ACTIVE,
                )
            )
        ).scalars().all()
        facts = (
            await session.execute(
                select(Fact).where(
                    Fact.profile_id == self.profile_id,
                    Fact.status == STATUS_ACTIVE,
                )
            )
        ).scalars().all()
        pairs: list[_CandidatePair] = []
        for decision in decisions:
            for fact in facts:
                if (decision.project or "") != (fact.project or ""):
                    continue
                if _polarity_conflict(decision.decision, fact.statement):
                    pairs.append(
                        _CandidatePair(
                            conflict_type="decision_fact",
                            left_type="decision",
                            left_id=decision.id,
                            right_type="fact",
                            right_id=fact.id,
                            severity="medium",
                            confidence=0.7,
                            hint="active decision contradicts active fact",
                            detector="decision_fact_polarity",
                            requires_review=True,
                        )
                    )
        return pairs

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------
    async def resolve_by_supersession(
        self,
        session: AsyncSession,
        *,
        conflict_id: uuid.UUID,
        kept_id: uuid.UUID,
        dropped_id: uuid.UUID,
        actor: str = "auto",
    ) -> MemoryConflict:
        conflict = await self._load_conflict(session, conflict_id)
        if conflict is None:
            raise ValueError("conflict not found")
        await self._supersede(session, conflict, kept_id=kept_id, dropped_id=dropped_id, actor=actor)
        return conflict

    async def record_user_correction(
        self,
        session: AsyncSession,
        *,
        conflict_id: uuid.UUID,
        kept_id: uuid.UUID,
        dropped_id: uuid.UUID,
    ) -> MemoryConflict:
        conflict = await self._load_conflict(session, conflict_id)
        if conflict is None:
            raise ValueError("conflict not found")
        await self._supersede(
            session, conflict, kept_id=kept_id, dropped_id=dropped_id, actor="user_correction"
        )
        return conflict

    async def ignore(self, session: AsyncSession, conflict_id: uuid.UUID) -> MemoryConflict:
        conflict = await self._load_conflict(session, conflict_id)
        if conflict is None:
            raise ValueError("conflict not found")
        conflict.status = CONFLICT_STATUS_IGNORED
        conflict.resolution = "manually ignored"
        conflict.resolved_at = datetime.now(UTC)
        await session.flush()
        return conflict

    async def _supersede(
        self,
        session: AsyncSession,
        conflict: MemoryConflict,
        *,
        kept_id: uuid.UUID,
        dropped_id: uuid.UUID,
        actor: str,
    ) -> None:
        dropped_type = (
            conflict.left_type if conflict.left_id == dropped_id else conflict.right_type
        )
        if dropped_type == "decision":
            decision_row = (
                await session.execute(
                    select(Decision).where(
                        Decision.id == dropped_id, Decision.profile_id == self.profile_id
                    )
                )
            ).scalar_one()
            decision_row.status = STATUS_SUPERSEDED
            decision_row.superseded_by = kept_id
            session.add(
                MemoryEvent(
                    event_type=EVENT_TYPE_DECISION_SUPERSEDED,
                    source="conflict_resolver",
                    actor=actor,
                    profile_id=self.profile_id,
                    payload={
                        "event_type": EVENT_TYPE_DECISION_SUPERSEDED,
                        "decision_id": str(dropped_id),
                        "superseded_by": str(kept_id),
                        "reason": f"resolved by {actor}",
                    },
                )
            )
        else:
            fact_row = (
                await session.execute(
                    select(Fact).where(
                        Fact.id == dropped_id, Fact.profile_id == self.profile_id
                    )
                )
            ).scalar_one()
            fact_row.status = STATUS_SUPERSEDED
            fact_row.superseded_by = kept_id
            session.add(
                MemoryEvent(
                    event_type=EVENT_TYPE_FACT_INVALIDATED,
                    source="conflict_resolver",
                    actor=actor,
                    profile_id=self.profile_id,
                    payload={
                        "event_type": EVENT_TYPE_FACT_INVALIDATED,
                        "fact_id": str(dropped_id),
                        "reason": f"resolved by {actor}",
                        "superseded_by": str(kept_id),
                    },
                )
            )
        conflict.status = CONFLICT_STATUS_AUTO_RESOLVED
        conflict.resolved_by = actor
        conflict.resolution = f"kept={kept_id} dropped={dropped_id}"
        conflict.resolved_at = datetime.now(UTC)
        await session.flush()

    async def _load_conflict(
        self, session: AsyncSession, conflict_id: uuid.UUID
    ) -> MemoryConflict | None:
        return (
            await session.execute(
                select(MemoryConflict).where(
                    MemoryConflict.id == conflict_id,
                    MemoryConflict.profile_id == self.profile_id,
                )
            )
        ).scalar_one_or_none()


# ----------------------------------------------------------------------
# Dataclasses + helpers
# ----------------------------------------------------------------------
@dataclass
class _CandidatePair:
    conflict_type: str
    left_type: str
    left_id: uuid.UUID
    right_type: str
    right_id: uuid.UUID
    severity: str = "medium"
    confidence: float = 0.75
    hint: str = ""
    detector: str = "rule"
    requires_review: bool = False


_NEGATION_TOKENS = {
    ("use", "do not use"),
    ("use", "don't use"),
    ("use", "avoid"),
    ("use", "stop using"),
    ("adopt", "drop"),
    ("adopt", "abandon"),
    ("primary", "secondary"),
    ("enable", "disable"),
    ("enabled", "disabled"),
    ("keep", "remove"),
    ("accept", "reject"),
    ("approve", "reject"),
    ("include", "exclude"),
    ("install", "uninstall"),
}


def _polarity_conflict(a_text: str, b_text: str) -> bool:
    a = normalize_text(a_text)
    b = normalize_text(b_text)
    if not a or not b or a == b:
        return False
    for positive, negative in _NEGATION_TOKENS:
        if (positive in a and negative in b) or (negative in a and positive in b):
            return True
    # Direct Russian/English negation: "use X" vs "не использовать X".
    return bool(_is_negation(a, b) or _is_negation(b, a))


def _is_negation(pos: str, neg: str) -> bool:
    """Simple heuristic: ``neg`` contains a negation token and shares content with ``pos``."""
    import re

    if not re.search(r"\b(не|not|do not|don't|never|no longer|stop)\b", neg):
        return False
    stripped = re.sub(r"\b(не|not|do not|don't|never|no longer|stop)\b", "", neg).strip()
    if not stripped:
        return False
    # Overlap >= 2 tokens between positive and stripped-negative.
    pos_tokens = set(pos.split())
    stripped_tokens = set(stripped.split())
    common = {t for t in pos_tokens & stripped_tokens if len(t) > 2}
    return len(common) >= 2


def _temporal_overlap(a: Fact, b: Fact) -> bool:
    a_from = _coerce_utc(a.valid_from)
    a_to = _coerce_utc(a.valid_to)
    b_from = _coerce_utc(b.valid_from)
    b_to = _coerce_utc(b.valid_to)
    if a_from is None and b_from is None and a_to is None and b_to is None:
        return True
    a_start = a_from or datetime.min.replace(tzinfo=UTC)
    a_end = a_to or datetime.max.replace(tzinfo=UTC)
    b_start = b_from or datetime.min.replace(tzinfo=UTC)
    b_end = b_to or datetime.max.replace(tzinfo=UTC)
    return a_start <= b_end and b_start <= a_end


def _coerce_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value
