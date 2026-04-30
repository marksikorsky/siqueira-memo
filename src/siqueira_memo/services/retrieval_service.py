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

import json
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.logging import get_logger
from siqueira_memo.models import (
    Chunk,
    ChunkEmbeddingBGEM3,
    ChunkEmbeddingMock,
    ChunkEmbeddingOpenAITEL3,
    Decision,
    Fact,
    MemoryEvent,
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
)
from siqueira_memo.services.redaction_service import redact
from siqueira_memo.services.relationship_service import RelationshipService
from siqueira_memo.services.retrieval_fusion import (
    ChunkScoringInput,
    ScoredChunk,
    entity_match_terms,
    filter_non_matching_chunks,
    has_temporal_intent,
    lexical_overlap_score,
    recency_weight,
    score_chunks,
    summarize_chunk_fusion,
    temporal_content_tokens,
    tokenize_query,
)
from siqueira_memo.services.secret_policy import (
    is_secret_metadata,
    masked_preview,
    recall_policy,
    secret_ref,
)
from siqueira_memo.services.trust_service import TrustService
from siqueira_memo.utils.canonical import normalize_text

log = get_logger(__name__)


_EMBEDDING_TABLE_BY_NAME = {
    "chunk_embeddings_mock": ChunkEmbeddingMock,
    "chunk_embeddings_openai_text_embedding_3_large": ChunkEmbeddingOpenAITEL3,
    "chunk_embeddings_bge_m3": ChunkEmbeddingBGEM3,
}


@dataclass
class RetrievalResult:
    context_pack: ContextPack
    candidates_count: int = 0
    rejected_count: int = 0
    conflicts_count: int = 0
    latency_ms: int = 0


def _tokenize(query: str) -> list[str]:
    return tokenize_query(query)


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
        trusted_internal: bool = False,
    ) -> None:
        self.profile_id = profile_id
        self.embedding_provider = embedding_provider or MockEmbeddingProvider()
        self.embedding_table_name = (
            embedding_table_name or self.embedding_provider.spec.table_name
        )
        self.trusted_internal = trusted_internal

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
                lane, explanation, score_breakdown = self._decision_retrieval_metadata(
                    decision, request=request, tokens=tokens
                )
                decisions_out.append(
                    self._decision_to_schema(
                        decision,
                        allow_secret_recall=self._raw_secret_recall(request),
                        retrieval_lane=lane,
                        retrieval_explanation=explanation,
                        score_breakdown=score_breakdown,
                    )
                )

        # ---------- facts ----------
        if "facts" in want_types or not want_types:
            fact_rows = await self._query_facts(session, request, tokens, limits["facts"])
            candidates_count += len(fact_rows)
            for fact in fact_rows:
                lane, explanation, score_breakdown = self._fact_retrieval_metadata(
                    fact, request=request, tokens=tokens
                )
                facts_out.append(
                    self._fact_to_schema(
                        fact,
                        allow_secret_recall=self._raw_secret_recall(request),
                        retrieval_lane=lane,
                        retrieval_explanation=explanation,
                        score_breakdown=score_breakdown,
                    )
                )

        fusion_metadata: dict[str, object] = {
            "lane_counts": {},
            "chunk_score_fields": [
                "lexical_score",
                "vector_score",
                "recency_score",
                "final_score",
            ],
        }

        # ---------- chunks (explicit lexical/vector/recency fusion) ----------
        if "chunks" in want_types or not want_types:
            candidates = await self._query_chunks(
                session, request, tokens, limits["chunks"] * 3
            )
            candidates_count += len(candidates)
            chunk_inputs = await self._chunk_scoring_inputs(session, candidates)
            scored = score_chunks(
                chunk_inputs,
                query=request.query,
                embedding_provider=self.embedding_provider,
                entities=request.entities,
            )
            matched = filter_non_matching_chunks(scored, query=request.query, entities=request.entities)
            matched.sort(key=lambda c: c.score.final_score, reverse=True)
            kept = matched[: limits["chunks"]]
            rejected_count += max(0, len(scored) - len(kept))
            fusion_metadata = summarize_chunk_fusion(kept)
            for candidate in kept:
                chunks_out.append(self._chunk_to_schema(candidate))

        # ---------- summaries ----------
        if "summaries" in want_types or not want_types:
            summaries_out.extend(
                await self._query_summaries(session, request, tokens, limits["summaries"])
            )
            candidates_count += len(summaries_out)

        # ---------- relationship graph expansion ----------
        if "decisions" in want_types or "facts" in want_types or not want_types:
            await self._expand_relationship_graph(
                session,
                request,
                decisions_out=decisions_out,
                facts_out=facts_out,
                limit=max(2, min(12, request.limit)),
            )

        # ---------- conflicts ----------
        conflicts = _detect_conflicts(decisions_out, facts_out) if request.include_conflicts else []

        # ---------- build answer context / confidence ----------
        answer_context = _build_answer_context(decisions_out, facts_out, summaries_out)
        confidence = _confidence_hint(decisions_out, facts_out, chunks_out, conflicts)
        warnings = _collect_warnings(decisions_out, facts_out, conflicts, mode, limits, chunks_out)
        if not request.allow_secret_recall and await self._has_secret_candidates(session, request):
            warnings.append("secret memories matched scope but were omitted; set allow_secret_recall=true and ask explicitly to inspect masked secret records")

        token_estimate = _estimate_tokens(answer_context, chunks_out, decisions_out, facts_out)

        source_snippets: list[SourceRef] = []
        if request.include_sources:
            source_snippets = _top_sources(chunks_out, limits)
            event_snippets = await self._query_source_events(
                session, request, tokens, limit=max(3, min(10, request.limit))
            )
            source_snippets = _merge_source_snippets(event_snippets, source_snippets, limit=max(3, min(10, request.limit)))

        pack = ContextPack(
            answer_context=answer_context,
            decisions=decisions_out,
            facts=facts_out,
            chunks=chunks_out,
            summaries=summaries_out,
            source_snippets=source_snippets,
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
                    "retrieval_fusion": {
                        **fusion_metadata,
                        "structured_lane_counts": _structured_lane_counts(decisions_out, facts_out),
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

        query_tokens = temporal_content_tokens(request.query) if has_temporal_intent(request.query) else list(tokens)
        if query_tokens and request.topic is None:
            columns = [
                Decision.decision,
                Decision.topic,
                Decision.rationale,
                Decision.context,
                Decision.project,
            ]
            per_token_filters = [
                or_(*(column.ilike(f"%{token}%") for column in columns))
                for token in query_tokens
            ]
            stmt = stmt.where(
                and_(*per_token_filters)
                if has_temporal_intent(request.query)
                else or_(*per_token_filters)
            )

        stmt = stmt.order_by(
            (Decision.status == STATUS_ACTIVE).desc(),
            Decision.decided_at.desc(),
            Decision.id.asc(),
        ).limit(limit * 8)

        rows = list((await session.execute(stmt)).scalars().all())
        rows = [row for row in rows if _secret_allowed(row.extra_metadata, request)]
        if tokens:
            if has_temporal_intent(request.query):
                rows.sort(
                    key=lambda d: (
                        d.status == STATUS_ACTIVE,
                        self._decision_score_breakdown(d, request=request, tokens=tokens)["final_score"],
                        d.decided_at,
                    ),
                    reverse=True,
                )
            else:
                rows.sort(
                    key=lambda d: (
                        d.status == STATUS_ACTIVE,
                        self._decision_score_breakdown(d, request=request, tokens=tokens)["final_score"],
                    ),
                    reverse=True,
                )
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
        query_tokens = temporal_content_tokens(request.query) if has_temporal_intent(request.query) else list(tokens)
        if query_tokens and request.topic is None:
            columns = [Fact.statement, Fact.subject, Fact.predicate, Fact.object, Fact.topic, Fact.project]
            per_token_filters = [
                or_(*(column.ilike(f"%{token}%") for column in columns))
                for token in query_tokens
            ]
            stmt = stmt.where(
                and_(*per_token_filters)
                if has_temporal_intent(request.query)
                else or_(*per_token_filters)
            )
        stmt = stmt.order_by(
            (Fact.status == STATUS_ACTIVE).desc(),
            func.coalesce(Fact.valid_from, Fact.valid_to, Fact.created_at, Fact.updated_at).desc(),
            Fact.id.asc(),
        )
        rows = list((await session.execute(stmt.limit(limit * 8))).scalars().all())
        rows = [row for row in rows if _secret_allowed(row.extra_metadata, request)]
        if tokens:
            if has_temporal_intent(request.query):
                rows.sort(
                    key=lambda f: (
                        f.status == STATUS_ACTIVE,
                        self._fact_score_breakdown(f, request=request, tokens=tokens)["final_score"],
                        f.valid_from or f.valid_to or f.created_at or f.updated_at,
                    ),
                    reverse=True,
                )
            else:
                rows.sort(
                    key=lambda f: (
                        f.status == STATUS_ACTIVE,
                        self._fact_score_breakdown(f, request=request, tokens=tokens)["final_score"],
                    ),
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
        if not self._raw_secret_recall(request):
            stmt = stmt.where(Chunk.sensitivity.notin_({"sensitive", "secret"}))

        if tokens:
            rows: list[Chunk] = []
            lexical_stmt = stmt.where(
                or_(*(Chunk.chunk_text.ilike(f"%{token}%") for token in tokens))
            ).order_by(Chunk.created_at.desc(), Chunk.id.asc())
            rows.extend(list((await session.execute(lexical_stmt.limit(limit * 8))).scalars().all()))
            rows.extend(await self._query_embedding_chunk_rows(session, request, limit=limit * 4))
            rows.extend(await self._query_entity_chunk_rows(session, request, limit=limit * 8))
            rows = _dedupe_chunks(rows)
            if request.entities:
                rows = [row for row in rows if _chunk_matches_requested_entities(row, request.entities)]
            return rows

        rows = list(
            (await session.execute(stmt.order_by(Chunk.created_at.desc(), Chunk.id.asc()).limit(limit * 4)))
            .scalars()
            .all()
        )
        rows.extend(await self._query_entity_chunk_rows(session, request, limit=limit * 8))
        rows = _dedupe_chunks(rows)
        if request.entities:
            rows = [row for row in rows if _chunk_matches_requested_entities(row, request.entities)]
        return rows

    async def _query_entity_chunk_rows(
        self, session: AsyncSession, request: RecallRequest, *, limit: int
    ) -> list[Chunk]:
        if not request.entities:
            return []
        stmt = select(Chunk).where(Chunk.profile_id == self.profile_id)
        if request.project is not None:
            stmt = stmt.where(Chunk.project == request.project)
        if request.topic is not None:
            stmt = stmt.where(Chunk.topic == request.topic)
        if not self._raw_secret_recall(request):
            stmt = stmt.where(Chunk.sensitivity.notin_({"sensitive", "secret"}))
        if not any(normalize_text(entity).strip() for entity in request.entities):
            return []
        # Correctness beats cleverness here: substring SQL prefilters can be
        # saturated by api-gateway/public-api style false positives and silently
        # miss an older exact entity=["api"] chunk. Until Phase 7 introduces a
        # proper entity index/search service, scan the already scoped chunk rows
        # and apply exact normalized matching before limiting.
        stmt = stmt.order_by(Chunk.created_at.desc(), Chunk.id.asc())
        rows = list((await session.execute(stmt)).scalars().all())
        exact_matches = [row for row in rows if _chunk_matches_requested_entities(row, request.entities)]
        return exact_matches[:limit]

    async def _query_embedding_chunk_rows(
        self, session: AsyncSession, request: RecallRequest, *, limit: int
    ) -> list[Chunk]:
        embedding_model: Any = _EMBEDDING_TABLE_BY_NAME.get(self.embedding_table_name)
        if embedding_model is None:
            return []
        stmt = (
            select(Chunk)
            .join(embedding_model, embedding_model.chunk_id == Chunk.id)
            .where(Chunk.profile_id == self.profile_id)
            .where(embedding_model.embedding.is_not(None))
            .order_by(Chunk.created_at.desc(), Chunk.id.asc())
            .limit(limit)
        )
        if request.project is not None:
            stmt = stmt.where(Chunk.project == request.project)
        if request.topic is not None:
            stmt = stmt.where(Chunk.topic == request.topic)
        if not self._raw_secret_recall(request):
            stmt = stmt.where(Chunk.sensitivity.notin_({"sensitive", "secret"}))
        return list((await session.execute(stmt)).scalars().all())

    async def _chunk_scoring_inputs(
        self, session: AsyncSession, chunks: Sequence[Chunk]
    ) -> list[ChunkScoringInput]:
        embedding_by_chunk_id = await self._load_chunk_embeddings(session, [chunk.id for chunk in chunks])
        return [
            ChunkScoringInput(
                id=chunk.id,
                source_type=chunk.source_type,
                source_id=chunk.source_id,
                chunk_text=chunk.chunk_text,
                token_count=chunk.token_count,
                tokenizer_name=chunk.tokenizer_name,
                project=chunk.project,
                topic=chunk.topic,
                entities=list(chunk.entities or []),
                sensitivity=chunk.sensitivity,
                created_at=chunk.created_at,
                extra_metadata=dict(chunk.extra_metadata or {}),
                embedding=embedding_by_chunk_id.get(chunk.id),
            )
            for chunk in chunks
        ]

    async def _load_chunk_embeddings(
        self, session: AsyncSession, chunk_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, list[float]]:
        if not chunk_ids:
            return {}
        embedding_model: Any = _EMBEDDING_TABLE_BY_NAME.get(self.embedding_table_name)
        if embedding_model is None:
            return {}
        rows = (
            await session.execute(
                select(embedding_model.chunk_id, embedding_model.embedding)
                .where(embedding_model.chunk_id.in_(chunk_ids))
                .where(embedding_model.embedding.is_not(None))
            )
        ).all()
        out: dict[uuid.UUID, list[float]] = {}
        for chunk_id, embedding in rows:
            if embedding is not None:
                out[chunk_id] = [float(v) for v in embedding]
        return out

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

    async def _expand_relationship_graph(
        self,
        session: AsyncSession,
        request: RecallRequest,
        *,
        decisions_out: list[RecallDecision],
        facts_out: list[RecallFact],
        limit: int,
    ) -> None:
        seeds: list[tuple[str, uuid.UUID]] = [("decision", d.id) for d in decisions_out]
        seeds.extend(("fact", f.id) for f in facts_out)
        seeds.extend(await self._query_relationship_seed_rows(session, request, limit=max(20, limit * 4)))
        seeds = _dedupe_seed_nodes(seeds)
        if not seeds:
            return
        related = await RelationshipService(profile_id=self.profile_id).expand_related(
            session, seeds, limit=limit
        )
        if not related:
            return
        decision_ids = {d.id for d in decisions_out}
        fact_ids = {f.id for f in facts_out}
        want_decisions = "decisions" in set(request.types or []) or not request.types
        want_facts = "facts" in set(request.types or []) or not request.types
        for rel in related:
            if rel.target_type == "decision" and want_decisions:
                existing = next((d for d in decisions_out if d.id == rel.target_id), None)
                if existing is not None:
                    if existing.retrieval_lane == "structured":
                        existing.retrieval_lane = "graph"
                        existing.retrieval_explanation = rel.explanation
                    elif existing.retrieval_explanation:
                        existing.retrieval_explanation = f"{existing.retrieval_explanation}; related: {rel.explanation}"
                    continue
                decision_row = (
                    await session.execute(
                        select(Decision).where(
                            Decision.id == rel.target_id,
                            Decision.profile_id == self.profile_id,
                        )
                    )
                ).scalar_one_or_none()
                if decision_row is None or not _secret_allowed(decision_row.extra_metadata, request):
                    continue
                decision_schema = self._decision_to_schema(
                    decision_row,
                    allow_secret_recall=self._raw_secret_recall(request),
                    retrieval_lane="graph",
                    retrieval_explanation=rel.explanation,
                )
                decisions_out.append(decision_schema)
                decision_ids.add(decision_schema.id)
            elif rel.target_type == "fact" and want_facts:
                existing_fact = next((f for f in facts_out if f.id == rel.target_id), None)
                if existing_fact is not None:
                    if existing_fact.retrieval_lane == "structured":
                        existing_fact.retrieval_lane = "graph"
                        existing_fact.retrieval_explanation = rel.explanation
                    elif existing_fact.retrieval_explanation:
                        existing_fact.retrieval_explanation = f"{existing_fact.retrieval_explanation}; related: {rel.explanation}"
                    continue
                fact_row = (
                    await session.execute(
                        select(Fact).where(
                            Fact.id == rel.target_id,
                            Fact.profile_id == self.profile_id,
                        )
                    )
                ).scalar_one_or_none()
                if fact_row is None or not _secret_allowed(fact_row.extra_metadata, request):
                    continue
                fact_schema = self._fact_to_schema(
                    fact_row,
                    allow_secret_recall=self._raw_secret_recall(request),
                    retrieval_lane="graph",
                    retrieval_explanation=rel.explanation,
                )
                facts_out.append(fact_schema)
                fact_ids.add(fact_schema.id)

    async def _query_relationship_seed_rows(
        self, session: AsyncSession, request: RecallRequest, *, limit: int
    ) -> list[tuple[str, uuid.UUID]]:
        tokens = _tokenize(request.query)
        if not tokens:
            return []
        seeds: list[tuple[str, uuid.UUID]] = []
        fact_stmt = select(Fact).where(Fact.profile_id == self.profile_id)
        decision_stmt = select(Decision).where(Decision.profile_id == self.profile_id)
        if request.project is not None:
            fact_stmt = fact_stmt.where(Fact.project == request.project)
            decision_stmt = decision_stmt.where(Decision.project == request.project)
        if request.topic is not None:
            fact_stmt = fact_stmt.where(Fact.topic == request.topic)
            decision_stmt = decision_stmt.where(Decision.topic == request.topic)
        fact_rows = list((await session.execute(fact_stmt.limit(limit))).scalars().all())
        for fact in fact_rows:
            text = normalize_text(" ".join([fact.statement, fact.subject, fact.predicate, fact.object, fact.topic or ""]))
            if any(token in text for token in tokens):
                seeds.append(("fact", fact.id))
        decision_rows = list((await session.execute(decision_stmt.limit(limit))).scalars().all())
        for decision in decision_rows:
            text = normalize_text(" ".join([decision.decision, decision.topic, decision.context or "", decision.rationale]))
            if any(token in text for token in tokens):
                seeds.append(("decision", decision.id))
        return seeds

    def _raw_secret_recall(self, request: RecallRequest) -> bool:
        return self.trusted_internal and request.allow_secret_recall

    async def _query_source_events(
        self, session: AsyncSession, request: RecallRequest, tokens: Sequence[str], *, limit: int
    ) -> list[SourceRef]:
        source_tokens = temporal_content_tokens(request.query) or list(tokens)
        if not source_tokens or request.project is not None or request.topic is not None or request.entities:
            return []
        trusted = self.trusted_internal and _trusted_source_recall(request)
        stmt = select(MemoryEvent).where(MemoryEvent.profile_id == self.profile_id)
        if request.session_id is not None:
            stmt = stmt.where(MemoryEvent.session_id == request.session_id)
        if request.since is not None:
            stmt = stmt.where(MemoryEvent.created_at >= request.since)
        if request.until is not None:
            stmt = stmt.where(MemoryEvent.created_at <= request.until)
        candidate_window = (
            1000
            if request.mode == RECALL_MODE_FORENSIC
            else 500
            if request.mode == RECALL_MODE_DEEP
            else 200
        )
        stmt = stmt.order_by(MemoryEvent.created_at.desc(), MemoryEvent.id.asc()).limit(candidate_window)
        rows = list((await session.execute(stmt)).scalars().all())
        scored = sorted(
            rows,
            key=lambda event: (
                lexical_overlap_score(source_tokens, _event_search_text(event, trusted=trusted)),
                event.created_at,
            ),
            reverse=True,
        )
        out: list[SourceRef] = []
        for event in scored:
            score = lexical_overlap_score(source_tokens, _event_search_text(event, trusted=trusted))
            if score <= 0:
                continue
            out.append(_event_to_source_ref(event, source_tokens, trusted=trusted))
            if len(out) >= limit:
                break
        return out

    async def _has_secret_candidates(self, session: AsyncSession, request: RecallRequest) -> bool:
        fact_stmt = select(Fact).where(Fact.profile_id == self.profile_id)
        decision_stmt = select(Decision).where(Decision.profile_id == self.profile_id)
        if request.project is not None:
            fact_stmt = fact_stmt.where(Fact.project == request.project)
            decision_stmt = decision_stmt.where(Decision.project == request.project)
        if request.topic is not None:
            fact_stmt = fact_stmt.where(Fact.topic == request.topic)
            decision_stmt = decision_stmt.where(Decision.topic == request.topic)
        facts = list((await session.execute(fact_stmt.limit(50))).scalars().all())
        if any(is_secret_metadata(fact.extra_metadata) for fact in facts):
            return True
        decisions = list((await session.execute(decision_stmt.limit(50))).scalars().all())
        return any(is_secret_metadata(decision.extra_metadata) for decision in decisions)

    # ------------------------------------------------------------------
    # Structured retrieval scoring
    # ------------------------------------------------------------------
    def _decision_score_breakdown(
        self, decision: Decision, *, request: RecallRequest, tokens: Sequence[str]
    ) -> dict[str, float]:
        text = " ".join(
            filter(None, [decision.decision, decision.topic, decision.rationale, decision.context, decision.project or ""])
        )
        score_tokens = temporal_content_tokens(request.query) if has_temporal_intent(request.query) else list(tokens)
        lexical_score = lexical_overlap_score(score_tokens, text)
        recency_score = recency_weight(decision.decided_at)
        confidence_score = float((decision.extra_metadata or {}).get("confidence") or 0.0)
        confidence_score = max(0.0, min(1.0, confidence_score))
        trust_score = TrustService.estimate_memory("decision", decision).trust_score
        if has_temporal_intent(request.query):
            final_score = lexical_score * 0.62 + recency_score * 0.18 + confidence_score * 0.08 + trust_score * 0.12
        else:
            final_score = lexical_score * 0.58 + confidence_score * 0.14 + recency_score * 0.08 + trust_score * 0.20
        return {
            "lexical_score": round(lexical_score, 4),
            "recency_score": round(recency_score, 4),
            "confidence_score": round(confidence_score, 4),
            "trust_score": round(trust_score, 4),
            "final_score": round(final_score, 4),
        }

    def _fact_score_breakdown(
        self, fact: Fact, *, request: RecallRequest, tokens: Sequence[str]
    ) -> dict[str, float]:
        text = " ".join(
            filter(None, [fact.statement, fact.subject, fact.predicate, fact.object, fact.topic or ""])
        )
        score_tokens = temporal_content_tokens(request.query) if has_temporal_intent(request.query) else list(tokens)
        lexical_score = lexical_overlap_score(score_tokens, text)
        temporal_anchor = fact.valid_from or fact.valid_to or fact.created_at or fact.updated_at
        recency_score = recency_weight(temporal_anchor)
        confidence_score = max(0.0, min(1.0, float(fact.confidence or 0.0)))
        trust_score = TrustService.estimate_memory("fact", fact).trust_score
        if has_temporal_intent(request.query):
            final_score = lexical_score * 0.62 + recency_score * 0.18 + confidence_score * 0.08 + trust_score * 0.12
        else:
            final_score = lexical_score * 0.50 + confidence_score * 0.16 + recency_score * 0.08 + trust_score * 0.26
        return {
            "lexical_score": round(lexical_score, 4),
            "recency_score": round(recency_score, 4),
            "confidence_score": round(confidence_score, 4),
            "trust_score": round(trust_score, 4),
            "final_score": round(final_score, 4),
        }

    def _decision_retrieval_metadata(
        self, decision: Decision, *, request: RecallRequest, tokens: Sequence[str]
    ) -> tuple[str, str | None, dict[str, float]]:
        score_breakdown = self._decision_score_breakdown(decision, request=request, tokens=tokens)
        if has_temporal_intent(request.query):
            return (
                "temporal",
                "latest/current query ranked this decision by lexical match and recency",
                score_breakdown,
            )
        if score_breakdown["lexical_score"] > 0:
            return "structured", "structured decision matched query terms", score_breakdown
        return "structured", None, score_breakdown

    def _fact_retrieval_metadata(
        self, fact: Fact, *, request: RecallRequest, tokens: Sequence[str]
    ) -> tuple[str, str | None, dict[str, float]]:
        score_breakdown = self._fact_score_breakdown(fact, request=request, tokens=tokens)
        if has_temporal_intent(request.query):
            return (
                "temporal",
                "latest/current query ranked this fact by lexical match and valid_from recency",
                score_breakdown,
            )
        if score_breakdown["lexical_score"] > 0:
            return "structured", "structured fact matched query terms", score_breakdown
        return "structured", None, score_breakdown

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def _decision_to_schema(
        self,
        decision: Decision,
        *,
        allow_secret_recall: bool,
        retrieval_lane: str = "structured",
        retrieval_explanation: str | None = None,
        score_breakdown: dict[str, float] | None = None,
    ) -> RecallDecision:
        metadata = decision.extra_metadata or {}
        trust = TrustService.estimate_memory("decision", decision)
        secret = is_secret_metadata(metadata)
        should_mask = secret and not allow_secret_recall
        decision_text = masked_preview(decision.decision, metadata) if should_mask else decision.decision
        rationale = masked_preview(decision.rationale, metadata) if should_mask else decision.rationale
        return RecallDecision(
            id=decision.id,
            project=decision.project,
            topic=decision.topic,
            decision=decision_text,
            rationale=rationale,
            status=decision.status,
            reversible=decision.reversible,
            decided_at=decision.decided_at,
            confidence=float(decision.extra_metadata.get("confidence", 0.0))
            if decision.extra_metadata
            else 0.0,
            sensitivity="secret" if secret else str(metadata.get("sensitivity") or "internal"),
            masked_preview=masked_preview(decision.decision, metadata) if secret else None,
            secret_ref=secret_ref(metadata),
            secret_masked=should_mask,
            retrieval_lane=retrieval_lane,
            retrieval_explanation=retrieval_explanation,
            score_breakdown=score_breakdown or {},
            trust_score=trust.trust_score,
            trust_label=trust.trust_label,
            trust_explanation=trust.explanation,
            sources=[
                SourceRef(event_id=str(eid))
                for eid in (decision.source_event_ids or [])
            ],
        )

    def _fact_to_schema(
        self,
        fact: Fact,
        *,
        allow_secret_recall: bool,
        retrieval_lane: str = "structured",
        retrieval_explanation: str | None = None,
        score_breakdown: dict[str, float] | None = None,
    ) -> RecallFact:
        metadata = fact.extra_metadata or {}
        trust = TrustService.estimate_memory("fact", fact)
        secret = is_secret_metadata(metadata)
        should_mask = secret and not allow_secret_recall
        preview = masked_preview(fact.statement, metadata) if secret else None
        return RecallFact(
            id=fact.id,
            subject=masked_preview(fact.subject, metadata) if should_mask else fact.subject,
            predicate=fact.predicate,
            object=masked_preview(fact.object, metadata) if should_mask else fact.object,
            statement=preview if should_mask else fact.statement,
            status=fact.status,
            confidence=float(fact.confidence or 0.0),
            project=fact.project,
            topic=fact.topic,
            valid_from=fact.valid_from,
            valid_to=fact.valid_to,
            sensitivity="secret" if secret else str(metadata.get("sensitivity") or "internal"),
            masked_preview=preview,
            secret_ref=secret_ref(metadata),
            secret_masked=should_mask,
            retrieval_lane=retrieval_lane,
            retrieval_explanation=retrieval_explanation,
            score_breakdown=score_breakdown or {},
            trust_score=trust.trust_score,
            trust_label=trust.trust_label,
            trust_explanation=trust.explanation,
            sources=[
                SourceRef(event_id=str(eid)) for eid in (fact.source_event_ids or [])
            ],

        )

    def _chunk_to_schema(self, candidate: ScoredChunk) -> RecallChunk:
        chunk = candidate.chunk
        return RecallChunk(
            id=chunk.id,
            source_type=chunk.source_type,
            source_id=chunk.source_id,
            chunk_text=chunk.chunk_text,
            score=round(candidate.score.final_score, 4),
            project=chunk.project,
            topic=chunk.topic,
            sensitivity=chunk.sensitivity,
            created_at=chunk.created_at,
            retrieval_lane=candidate.score.source_lane,
            retrieval_explanation=candidate.score.explanation,
            score_breakdown=candidate.score.as_breakdown(),
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
_SAFE_LABEL_RE = re.compile(r"^[a-zA-Z0-9_.:-]{1,128}$")
_HEX_64_RE = re.compile(r"^[a-fA-F0-9]{64}$")
_SOURCE_TIMELINE_EVENT_TYPES = frozenset(
    {
        "decision_recorded",
        "decision_superseded",
        "fact_extracted",
        "fact_invalidated",
        "memory_deleted",
        "memory_rolled_back",
        "summary_created",
        "user_correction_received",
    }
)
_SAFE_EVENT_PAYLOAD_KEYS = frozenset(
    {
        "canonical_key",
        "confidence",
        "decision_id",
        "event_type",
        "fact_id",
        "kind",
        "operation",
        "relationship_type",
        "replacement_id",
        "scope",
        "source_id",
        "source_type",
        "status",
        "summary_id",
        "target_id",
        "target_type",
        "version_id",
    }
)


def _trusted_source_recall(request: RecallRequest) -> bool:
    return request.allow_secret_recall and request.mode == RECALL_MODE_FORENSIC


def _event_search_text(event: MemoryEvent, *, trusted: bool = False) -> str:
    if trusted:
        return _trusted_event_text(event)
    if event.event_type not in _SOURCE_TIMELINE_EVENT_TYPES:
        return ""
    payload_text = " ".join(str(value) for value in _safe_event_payload(event).values())
    return " ".join(filter(None, [event.event_type, payload_text]))


def _chunk_matches_requested_entities(chunk: Chunk, requested_entities: Sequence[str]) -> bool:
    return bool(entity_match_terms(list(chunk.entities or []), requested_entities))


def _event_to_source_ref(event: MemoryEvent, tokens: Sequence[str], *, trusted: bool = False) -> SourceRef:
    if trusted:
        text = _trusted_event_text(event)
        return SourceRef(
            event_id=str(event.id),
            snippet=_query_centered_excerpt(text, tokens, max_len=900),
            created_at=event.created_at,
            retrieval_lane="source_trusted",
            retrieval_explanation="trusted forensic recall matched raw source-event metadata/payload",
        )
    fields: dict[str, str] = {"event_type": event.event_type}
    fields.update({key: str(value) for key, value in _safe_event_payload(event).items()})
    text = "; ".join(f"{key}={redact(value).redacted}" for key, value in fields.items() if value)
    return SourceRef(
        event_id=str(event.id),
        snippet=_query_centered_excerpt(text, tokens),
        created_at=event.created_at,
        retrieval_lane="source",
        retrieval_explanation="matched safe source-event metadata/timeline by query terms",
    )


def _trusted_event_text(event: MemoryEvent) -> str:
    fields = {
        "event_type": event.event_type,
        "source": event.source,
        "actor": event.actor,
        "session_id": event.session_id or "",
        "payload": json.dumps(event.payload or {}, ensure_ascii=False, sort_keys=True, default=str),
    }
    return "; ".join(
        f"{key}={_compact_whitespace(str(value))}" for key, value in fields.items() if value
    )


def _safe_event_payload(event: MemoryEvent) -> dict[str, str]:
    if event.event_type not in _SOURCE_TIMELINE_EVENT_TYPES:
        return {}
    safe: dict[str, str] = {}
    for key, value in (event.payload or {}).items():
        if key not in _SAFE_EVENT_PAYLOAD_KEYS:
            continue
        if key == "event_type" and str(value) != event.event_type:
            continue
        safe_value = _safe_event_payload_value(key, value)
        if safe_value:
            safe[key] = safe_value
    return safe


def _safe_event_payload_value(key: str, value: object) -> str | None:
    if value is None:
        return None
    if key in {"confidence"}:
        return str(value) if isinstance(value, (int, float)) else None
    if key in {
        "decision_id",
        "fact_id",
        "replacement_id",
        "summary_id",
        "target_id",
        "version_id",
    }:
        return str(value) if _is_uuid_string(value) else None
    if key in {"canonical_key", "source_id"}:
        text = str(value)
        return text if _HEX_64_RE.fullmatch(text) or _is_uuid_string(text) else None
    if key in {
        "event_type",
        "kind",
        "operation",
        "relationship_type",
        "scope",
        "source_type",
        "status",
        "target_type",
    }:
        text = str(value)
        return text if _SAFE_LABEL_RE.fullmatch(text) else None
    return None


def _is_uuid_string(value: object) -> bool:
    try:
        uuid.UUID(str(value))
    except (TypeError, ValueError):
        return False
    return True


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _query_centered_excerpt(text: str, tokens: Sequence[str], *, max_len: int = 360) -> str:
    if len(text) <= max_len:
        return text
    lowered = text.lower()
    positions = [lowered.find(token.lower()) for token in tokens if lowered.find(token.lower()) >= 0]
    if not positions:
        return text[:max_len]
    center = min(positions)
    start = max(0, center - max_len // 3)
    end = min(len(text), start + max_len)
    if end - start < max_len:
        start = max(0, end - max_len)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def _merge_source_snippets(
    first: Sequence[SourceRef], second: Sequence[SourceRef], *, limit: int
) -> list[SourceRef]:
    seen: set[str] = set()
    out: list[SourceRef] = []
    for source in [*first, *second]:
        key = source.event_id
        if key in seen:
            continue
        seen.add(key)
        out.append(source)
        if len(out) >= limit:
            break
    return out


def _structured_lane_counts(
    decisions: Sequence[RecallDecision], facts: Sequence[RecallFact]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.retrieval_lane] = counts.get(decision.retrieval_lane, 0) + 1
    for fact in facts:
        counts[fact.retrieval_lane] = counts.get(fact.retrieval_lane, 0) + 1
    return dict(sorted(counts.items()))


def _dedupe_chunks(chunks: Sequence[Chunk]) -> list[Chunk]:
    seen: set[uuid.UUID] = set()
    out: list[Chunk] = []
    for chunk in chunks:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        out.append(chunk)
    return out


def _dedupe_seed_nodes(seeds: Sequence[tuple[str, uuid.UUID]]) -> list[tuple[str, uuid.UUID]]:
    seen: set[tuple[str, uuid.UUID]] = set()
    out: list[tuple[str, uuid.UUID]] = []
    for seed in seeds:
        if seed in seen:
            continue
        seen.add(seed)
        out.append(seed)
    return out


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
        if c.sensitivity in {"sensitive", "secret"}:
            continue
        out.append(
            SourceRef(
                event_id=str(c.source_id),
                snippet=c.chunk_text[:240],
                created_at=c.created_at,
            )
        )
    return out


def _secret_allowed(metadata: dict[str, Any] | None, request: RecallRequest) -> bool:
    if not is_secret_metadata(metadata):
        return True
    return request.allow_secret_recall and recall_policy(metadata) != "never_prefetch"


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
