"""Hybrid recall: structured memory + chunk lexical + vector. Plan §8.

Design principles:

- Structured decisions and facts are retrieved first and given priority.
- Chunks are ranked by a combination of lexical overlap and (where possible)
  vector similarity. The SQLite test backend cannot run HNSW so it scores in
  Python; Postgres production uses the per-model embedding table.
- Every recall writes a ``retrieval_logs`` row. The caller is responsible for
  materialising the log into a ``memory_events`` row only if it causes a
  durable state change (plan §31.10).
- Conflict detection runs cheap rules over the returned structured memory and
  surfaces conflicts as ``ConflictEntry`` rows.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.logging import get_logger
from siqueira_memo.models import (
    Chunk,
    ChunkEmbeddingBGEM3,
    ChunkEmbeddingMock,
    ChunkEmbeddingOpenAITEL3,
    Decision,
    Fact,
    RetrievalLog,
    SessionSummary,
    TopicSummary,
)
from siqueira_memo.models.constants import (
    RECALL_MODE_BALANCED,
    RECALL_MODE_DEEP,
    RECALL_MODE_FORENSIC,
    STATUS_ACTIVE,
)
from siqueira_memo.schemas.common import SourceRef
from siqueira_memo.schemas.recall import (
    ConflictEntry,
    ContextPack,
    RecallChunk,
    RecallDecision,
    RecallFact,
    RecallRequest,
    RecallSummary,
)
from siqueira_memo.services.embedding_service import (
    EmbeddingProvider,
    MockEmbeddingProvider,
    cosine,
)
from siqueira_memo.utils.canonical import normalize_text

log = get_logger(__name__)


_EMBEDDING_TABLE_BY_NAME = {
    "chunk_embeddings_mock": ChunkEmbeddingMock,
    "chunk_embeddings_openai_text_embedding_3_large": ChunkEmbeddingOpenAITEL3,
    "chunk_embeddings_bge_m3": ChunkEmbeddingBGEM3,
}


@dataclass
class _Candidate:
    chunk: Chunk
    lexical: float = 0.0
    vector: float = 0.0
    recency: float = 0.0

    @property
    def score(self) -> float:
        return self.lexical * 0.5 + self.vector * 0.4 + self.recency * 0.1


@dataclass
class RetrievalResult:
    context_pack: ContextPack
    candidates_count: int = 0
    rejected_count: int = 0
    conflicts_count: int = 0
    latency_ms: int = 0


def _tokenize(query: str) -> list[str]:
    text = normalize_text(query)
    # Keep words of length >= 2 to cut single-letter noise.
    return [t for t in re.findall(r"[\w\-]+", text, flags=re.UNICODE) if len(t) >= 2]


def _recency_weight(created_at: datetime | None) -> float:
    if created_at is None:
        return 0.0
    age_days = max(
        0.0, (datetime.now(created_at.tzinfo) - created_at).total_seconds() / 86_400.0
    )
    # Smooth decay: ~1 at age 0, ~0.5 at 14 days, ~0.1 at 90 days.
    return 1.0 / (1.0 + age_days / 14.0)


def _limits_for_mode(mode: str, requested: int) -> dict[str, int]:
    # plan §33.5 token budgets shape the chunk/snippet caps.
    if mode == RECALL_MODE_FORENSIC:
        return {"decisions": 40, "facts": 40, "chunks": 40, "summaries": 20}
    if mode == RECALL_MODE_DEEP:
        return {"decisions": 25, "facts": 25, "chunks": 25, "summaries": 10}
    if mode == RECALL_MODE_BALANCED:
        return {"decisions": 15, "facts": 15, "chunks": 12, "summaries": 6}
    # fast mode
    return {"decisions": 8, "facts": 8, "chunks": 5, "summaries": 3}


class RetrievalService:
    def __init__(
        self,
        *,
        profile_id: str,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_table_name: str | None = None,
    ) -> None:
        self.profile_id = profile_id
        self.embedding_provider = embedding_provider or MockEmbeddingProvider()
        self.embedding_table_name = (
            embedding_table_name or self.embedding_provider.spec.table_name
        )

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------
    async def recall(
        self, session: AsyncSession, request: RecallRequest
    ) -> RetrievalResult:
        started = time.perf_counter()
        mode = request.mode
        limits = _limits_for_mode(mode, request.limit)
        want_types = set(request.types or [])

        decisions_out: list[RecallDecision] = []
        facts_out: list[RecallFact] = []
        chunks_out: list[RecallChunk] = []
        summaries_out: list[RecallSummary] = []
        candidates_count = 0
        rejected_count = 0

        tokens = _tokenize(request.query)

        # ---------- decisions ----------
        if "decisions" in want_types or not want_types:
            decision_rows = await self._query_decisions(
                session, request, tokens, limits["decisions"]
            )
            candidates_count += len(decision_rows)
            for decision in decision_rows:
                decisions_out.append(self._decision_to_schema(decision))

        # ---------- facts ----------
        if "facts" in want_types or not want_types:
            fact_rows = await self._query_facts(session, request, tokens, limits["facts"])
            candidates_count += len(fact_rows)
            for fact in fact_rows:
                facts_out.append(self._fact_to_schema(fact))

        # ---------- chunks (lexical + vector) ----------
        if "chunks" in want_types or not want_types:
            candidates = await self._query_chunks(
                session, request, tokens, limits["chunks"] * 3
            )
            candidates_count += len(candidates)
            scored = self._score_candidates(candidates, request.query)
            scored.sort(key=lambda c: c.score, reverse=True)
            kept = scored[: limits["chunks"]]
            rejected_count += max(0, len(scored) - len(kept))
            for candidate in kept:
                chunks_out.append(self._chunk_to_schema(candidate))

        # ---------- summaries ----------
        if "summaries" in want_types or not want_types:
            summaries_out.extend(
                await self._query_summaries(session, request, tokens, limits["summaries"])
            )
            candidates_count += len(summaries_out)

        # ---------- conflicts ----------
        conflicts = _detect_conflicts(decisions_out, facts_out) if request.include_conflicts else []

        # ---------- build answer context / confidence ----------
        answer_context = _build_answer_context(decisions_out, facts_out, summaries_out)
        confidence = _confidence_hint(decisions_out, facts_out, chunks_out, conflicts)
        warnings = _collect_warnings(decisions_out, facts_out, conflicts, mode, limits, chunks_out)

        token_estimate = _estimate_tokens(answer_context, chunks_out, decisions_out, facts_out)

        pack = ContextPack(
            answer_context=answer_context,
            decisions=decisions_out,
            facts=facts_out,
            chunks=chunks_out,
            summaries=summaries_out,
            source_snippets=_top_sources(chunks_out, limits),
            conflicts=conflicts,
            confidence=confidence,
            warnings=warnings,
            mode=mode,
            embedding_table=self.embedding_table_name,
            latency_ms=0,
            token_estimate=token_estimate,
        )

        latency_ms = int((time.perf_counter() - started) * 1000)
        pack.latency_ms = latency_ms

        # Persist a retrieval log for diagnostics (plan §10.3).
        session.add(
            RetrievalLog(
                id=uuid.uuid4(),
                profile_id=self.profile_id,
                session_id=request.session_id,
                query=request.query,
                mode=mode,
                types={"requested": list(want_types)},
                selected_source_ids=_source_ids(chunks_out, decisions_out, facts_out),
                rejected_count=rejected_count,
                candidates_count=candidates_count,
                conflicts_count=len(conflicts),
                latency_ms=latency_ms,
                embedding_table=self.embedding_table_name,
                extra_metadata={
                    "limits": limits,
                    "filters": {
                        "project": request.project,
                        "topic": request.topic,
                        "entities": list(request.entities),
                    },
                },
            )
        )
        await session.flush()

        return RetrievalResult(
            context_pack=pack,
            candidates_count=candidates_count,
            rejected_count=rejected_count,
            conflicts_count=len(conflicts),
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------
    async def _query_decisions(
        self,
        session: AsyncSession,
        request: RecallRequest,
        tokens: Sequence[str],
        limit: int,
    ) -> list[Decision]:
        stmt = select(Decision).where(Decision.profile_id == self.profile_id)
        if request.project is not None:
            stmt = stmt.where(Decision.project == request.project)
        if request.topic is not None:
            stmt = stmt.where(Decision.topic == request.topic)

        # active first, most recent first.
        stmt = stmt.order_by(
            Decision.status == STATUS_ACTIVE,  # TRUE sorts after FALSE in ASC; see below
            Decision.decided_at.desc(),
        ).limit(limit * 4)

        rows = list((await session.execute(stmt)).scalars().all())
        if tokens:
            # Rank by token overlap with decision/topic/rationale.
            def match(d: Decision) -> float:
                text = " ".join(
                    filter(None, [d.decision, d.topic, d.rationale, d.context, d.project or ""])
                )
                normalized = normalize_text(text)
                return sum(1.0 for t in tokens if t in normalized) / max(1, len(tokens))

            rows.sort(key=lambda d: (d.status == STATUS_ACTIVE, match(d)), reverse=True)
        else:
            rows.sort(key=lambda d: (d.status == STATUS_ACTIVE, d.decided_at), reverse=True)
        return rows[:limit]

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------
    async def _query_facts(
        self,
        session: AsyncSession,
        request: RecallRequest,
        tokens: Sequence[str],
        limit: int,
    ) -> list[Fact]:
        stmt = select(Fact).where(Fact.profile_id == self.profile_id)
        if request.project is not None:
            stmt = stmt.where(Fact.project == request.project)
        if request.topic is not None:
            stmt = stmt.where(Fact.topic == request.topic)
        rows = list((await session.execute(stmt.limit(limit * 4))).scalars().all())
        if tokens:
            def score(fact: Fact) -> float:
                text = " ".join(
                    filter(
                        None,
                        [fact.statement, fact.subject, fact.predicate, fact.object, fact.topic or ""],
                    )
                )
                normalized = normalize_text(text)
                return sum(1.0 for t in tokens if t in normalized) / max(1, len(tokens))

            rows.sort(
                key=lambda f: (f.status == STATUS_ACTIVE, score(f), f.confidence),
                reverse=True,
            )
        else:
            rows.sort(key=lambda f: (f.status == STATUS_ACTIVE, f.confidence), reverse=True)
        return rows[:limit]

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------
    async def _query_chunks(
        self,
        session: AsyncSession,
        request: RecallRequest,
        tokens: Sequence[str],
        limit: int,
    ) -> list[Chunk]:
        stmt = select(Chunk).where(Chunk.profile_id == self.profile_id)
        if request.project is not None:
            stmt = stmt.where(Chunk.project == request.project)
        if request.topic is not None:
            stmt = stmt.where(Chunk.topic == request.topic)
        rows = list((await session.execute(stmt.limit(limit * 4))).scalars().all())
        return rows

    def _score_candidates(self, candidates: Iterable[Chunk], query: str) -> list[_Candidate]:
        tokens = set(_tokenize(query))
        embedded_query = self.embedding_provider.embed(query) if tokens else None
        scored: list[_Candidate] = []
        for chunk in candidates:
            lexical = 0.0
            if tokens:
                chunk_tokens = set(_tokenize(chunk.chunk_text))
                if chunk_tokens:
                    overlap = tokens & chunk_tokens
                    lexical = len(overlap) / max(1, len(tokens))
            vec_score = 0.0
            emb = chunk.extra_metadata.get("embedding") if chunk.extra_metadata else None
            if embedded_query is not None and isinstance(emb, list):
                vec_score = max(0.0, cosine(embedded_query, emb))
            scored.append(
                _Candidate(
                    chunk=chunk,
                    lexical=lexical,
                    vector=vec_score,
                    recency=_recency_weight(chunk.created_at),
                )
            )
        return scored

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    async def _query_summaries(
        self,
        session: AsyncSession,
        request: RecallRequest,
        tokens: Sequence[str],
        limit: int,
    ) -> list[RecallSummary]:
        out: list[RecallSummary] = []
        if request.session_id:
            session_row = (
                await session.execute(
                    select(SessionSummary)
                    .where(SessionSummary.profile_id == self.profile_id)
                    .where(SessionSummary.session_id == request.session_id)
                    .order_by(SessionSummary.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if session_row is not None:
                out.append(
                    RecallSummary(
                        id=session_row.id,
                        scope="session",
                        summary_short=session_row.summary_short,
                        summary_long=session_row.summary_long,
                        created_at=session_row.created_at,
                    )
                )
        if request.topic:
            topic_rows = (
                (
                    await session.execute(
                        select(TopicSummary)
                        .where(TopicSummary.profile_id == self.profile_id)
                        .where(TopicSummary.topic == request.topic)
                        .order_by(TopicSummary.created_at.desc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            for topic_row in topic_rows:
                out.append(
                    RecallSummary(
                        id=topic_row.id,
                        scope="topic",
                        summary_short=topic_row.summary_short,
                        summary_long=topic_row.summary_long,
                        created_at=topic_row.created_at,
                    )
                )
        return out[:limit]

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def _decision_to_schema(self, decision: Decision) -> RecallDecision:
        return RecallDecision(
            id=decision.id,
            project=decision.project,
            topic=decision.topic,
            decision=decision.decision,
            rationale=decision.rationale,
            status=decision.status,
            reversible=decision.reversible,
            decided_at=decision.decided_at,
            confidence=float(decision.extra_metadata.get("confidence", 0.0))
            if decision.extra_metadata
            else 0.0,
            sources=[
                SourceRef(event_id=str(eid))
                for eid in (decision.source_event_ids or [])
            ],
        )

    def _fact_to_schema(self, fact: Fact) -> RecallFact:
        return RecallFact(
            id=fact.id,
            subject=fact.subject,
            predicate=fact.predicate,
            object=fact.object,
            statement=fact.statement,
            status=fact.status,
            confidence=float(fact.confidence or 0.0),
            project=fact.project,
            topic=fact.topic,
            valid_from=fact.valid_from,
            valid_to=fact.valid_to,
            sources=[
                SourceRef(event_id=str(eid))
                for eid in (fact.source_event_ids or [])
            ],
        )

    def _chunk_to_schema(self, candidate: _Candidate) -> RecallChunk:
        chunk = candidate.chunk
        return RecallChunk(
            id=chunk.id,
            source_type=chunk.source_type,
            source_id=chunk.source_id,
            chunk_text=chunk.chunk_text,
            score=round(candidate.score, 4),
            project=chunk.project,
            topic=chunk.topic,
            sensitivity=chunk.sensitivity,
            created_at=chunk.created_at,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _build_answer_context(
    decisions: Sequence[RecallDecision],
    facts: Sequence[RecallFact],
    summaries: Sequence[RecallSummary],
) -> str:
    parts: list[str] = []
    active_decisions = [d for d in decisions if d.status == STATUS_ACTIVE]
    if active_decisions:
        parts.append(
            "Active decisions:\n"
            + "\n".join(f"- {d.decision}" for d in active_decisions[:5])
        )
    active_facts = [f for f in facts if f.status == STATUS_ACTIVE]
    if active_facts:
        parts.append(
            "Known facts:\n" + "\n".join(f"- {f.statement}" for f in active_facts[:5])
        )
    if summaries:
        parts.append("Recent summary: " + summaries[0].summary_short)
    return "\n\n".join(parts)


def _confidence_hint(
    decisions: Sequence[RecallDecision],
    facts: Sequence[RecallFact],
    chunks: Sequence[RecallChunk],
    conflicts: Sequence[ConflictEntry],
) -> str:
    if conflicts:
        return "low"
    active = [d for d in decisions if d.status == STATUS_ACTIVE] + [
        f for f in facts if f.status == STATUS_ACTIVE
    ]
    if active and chunks:
        return "high"
    if active or chunks:
        return "medium"
    return "low"


def _collect_warnings(
    decisions: Sequence[RecallDecision],
    facts: Sequence[RecallFact],
    conflicts: Sequence[ConflictEntry],
    mode: str,
    limits: dict[str, int],
    chunks: Sequence[RecallChunk],
) -> list[str]:
    warnings: list[str] = []
    if not decisions and not facts and not chunks:
        warnings.append("no matching memory found")
    if conflicts:
        warnings.append(f"{len(conflicts)} conflict(s) detected; live data takes precedence")
    if mode in {RECALL_MODE_DEEP, RECALL_MODE_FORENSIC}:
        warnings.append(
            "deep/forensic mode: caller should not auto-inject this into prefetch"
        )
    return warnings


def _top_sources(
    chunks: Sequence[RecallChunk], limits: dict[str, int]
) -> list[SourceRef]:
    out: list[SourceRef] = []
    for c in chunks[:3]:
        out.append(
            SourceRef(
                event_id=str(c.source_id),
                snippet=c.chunk_text[:240],
                created_at=c.created_at,
            )
        )
    return out


def _source_ids(
    chunks: Sequence[RecallChunk],
    decisions: Sequence[RecallDecision],
    facts: Sequence[RecallFact],
) -> list[uuid.UUID]:
    ids: set[uuid.UUID] = set()
    for c in chunks:
        ids.add(c.source_id)
    for d in decisions:
        for s in d.sources:
            try:
                ids.add(uuid.UUID(s.event_id))
            except (ValueError, AttributeError):
                continue
    for f in facts:
        for s in f.sources:
            try:
                ids.add(uuid.UUID(s.event_id))
            except (ValueError, AttributeError):
                continue
    return sorted(ids, key=str)


def _estimate_tokens(
    answer_context: str,
    chunks: Sequence[RecallChunk],
    decisions: Sequence[RecallDecision],
    facts: Sequence[RecallFact],
) -> int:
    # Rough approximation. Production would pass through tiktoken.
    size = len(answer_context.split())
    size += sum(len(c.chunk_text.split()) for c in chunks)
    size += sum(len((d.decision or "").split()) + len((d.rationale or "").split()) for d in decisions)
    size += sum(len((f.statement or "").split()) for f in facts)
    return size


def _detect_conflicts(
    decisions: Sequence[RecallDecision], facts: Sequence[RecallFact]
) -> list[ConflictEntry]:
    """Cheap rule-based conflict detector. Plan §21."""
    conflicts: list[ConflictEntry] = []
    # Decision-decision: same normalised topic and status=active with opposite polarity wording.
    active = [d for d in decisions if d.status == STATUS_ACTIVE]
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            if normalize_text(a.topic) != normalize_text(b.topic):
                continue
            if _polarity_conflict(a.decision, b.decision):
                older, newer = sorted([a, b], key=lambda d: d.decided_at)
                conflicts.append(
                    ConflictEntry(
                        older={"id": str(older.id), "decision": older.decision},
                        newer={"id": str(newer.id), "decision": newer.decision},
                        resolution="newer decision supersedes older",
                        severity="high",
                    )
                )
    # Fact-fact: same subject+predicate, different object, both active.
    by_key: dict[tuple[str, str], list[RecallFact]] = {}
    for f in facts:
        if f.status != STATUS_ACTIVE:
            continue
        key = (normalize_text(f.subject), normalize_text(f.predicate))
        by_key.setdefault(key, []).append(f)
    for _key, group in by_key.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda f: (f.valid_to or datetime.max.replace(tzinfo=None)))
        objs = {normalize_text(f.object) for f in group}
        if len(objs) > 1:
            older_fact = min(group, key=lambda f: f.valid_from or datetime.min)
            newer_fact = max(group, key=lambda f: f.valid_from or datetime.min)
            conflicts.append(
                ConflictEntry(
                    older={"id": str(older_fact.id), "statement": older_fact.statement},
                    newer={"id": str(newer_fact.id), "statement": newer_fact.statement},
                    resolution="divergent facts for same subject/predicate",
                    severity="medium",
                )
            )
    return conflicts


_NEGATION_PAIRS = [
    ("use", "do not use"),
    ("use", "don't use"),
    ("use", "avoid"),
    ("primary", "secondary"),
    ("enable", "disable"),
    ("enabled", "disabled"),
    ("keep", "remove"),
    ("add", "remove"),
    ("accept", "reject"),
]


def _polarity_conflict(a_text: str, b_text: str) -> bool:
    a = normalize_text(a_text)
    b = normalize_text(b_text)
    if a == b:
        return False
    for positive, negative in _NEGATION_PAIRS:
        if positive in a and negative in b:
            return True
        if negative in a and positive in b:
            return True
    # Direct negation.
    if re.search(r"\b(не|not|do not|don't|never)\b", a) and not re.search(
        r"\b(не|not|do not|don't|never)\b", b
    ):
        common = a.replace("не ", "").replace("not ", "").replace("do not ", "").replace(
            "don't ", ""
        )
        if common.strip() and common.strip() in b:
            return True
    if re.search(r"\b(не|not|do not|don't|never)\b", b) and not re.search(
        r"\b(не|not|do not|don't|never)\b", a
    ):
        common = b.replace("не ", "").replace("not ", "").replace("do not ", "").replace(
            "don't ", ""
        )
        if common.strip() and common.strip() in a:
            return True
    return False
