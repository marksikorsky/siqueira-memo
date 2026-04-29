"""Explicit retrieval fusion scoring for recall lanes.

This module keeps lane math out of ``RetrievalService`` so recall can explain
why a candidate was selected instead of returning an opaque score.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from siqueira_memo.services.embedding_service import EmbeddingProvider, cosine
from siqueira_memo.utils.canonical import normalize_text


class ChunkLike(Protocol):
    id: uuid.UUID
    source_type: str
    source_id: uuid.UUID
    chunk_text: str
    token_count: int
    tokenizer_name: str
    project: str | None
    topic: str | None
    entities: list[str]
    sensitivity: str
    created_at: datetime | None
    extra_metadata: dict[str, object]
    embedding: list[float] | None


@dataclass
class ChunkScoringInput:
    id: uuid.UUID
    source_type: str
    source_id: uuid.UUID
    chunk_text: str
    token_count: int
    tokenizer_name: str
    project: str | None
    topic: str | None
    entities: list[str] = field(default_factory=list)
    sensitivity: str = "normal"
    created_at: datetime | None = None
    extra_metadata: dict[str, object] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class RetrievalScore:
    lexical_score: float = 0.0
    vector_score: float = 0.0
    entity_score: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0
    source_lane: str = "recency"
    explanation: str = "ranked by recency"
    matched_terms: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)

    def as_breakdown(self) -> dict[str, float]:
        return {
            "lexical_score": round(self.lexical_score, 4),
            "vector_score": round(self.vector_score, 4),
            "entity_score": round(self.entity_score, 4),
            "recency_score": round(self.recency_score, 4),
            "final_score": round(self.final_score, 4),
        }


@dataclass
class ScoredChunk:
    chunk: ChunkLike
    score: RetrievalScore


def tokenize_query(query: str) -> list[str]:
    text = normalize_text(query)
    return [t for t in re.findall(r"[\w\-]+", text, flags=re.UNICODE) if len(t) >= 2]


_TEMPORAL_QUERY_TERMS = {
    "latest",
    "current",
    "recent",
    "newest",
    "актуальный",
    "актуальная",
    "актуальное",
    "актуальные",
    "текущий",
    "текущая",
    "текущее",
    "текущие",
    "последний",
    "последняя",
    "последнее",
    "последние",
    "свежий",
    "свежая",
    "свежее",
    "свежие",
}


def has_temporal_intent(query: str) -> bool:
    tokens = set(tokenize_query(query))
    return bool(tokens & _TEMPORAL_QUERY_TERMS)


_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "our",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "we",
    "а",
    "в",
    "где",
    "для",
    "и",
    "или",
    "как",
    "какая",
    "какие",
    "какой",
    "какое",
    "когда",
    "на",
    "по",
    "про",
    "у",
    "что",
}


def temporal_content_tokens(query: str) -> list[str]:
    return [
        token
        for token in tokenize_query(query)
        if token not in _TEMPORAL_QUERY_TERMS and token not in _QUERY_STOPWORDS
    ]


def lexical_overlap_score(tokens: Sequence[str], text: str) -> float:
    if not tokens:
        return 0.0
    normalized = normalize_text(text)
    return sum(1.0 for token in tokens if token in normalized) / max(1, len(tokens))


def entity_match_terms(candidate_entities: Sequence[str], requested_entities: Sequence[str]) -> list[str]:
    """Return requested entity names that exactly match candidate entity metadata.

    Entity retrieval is intentionally exact after normalization. Substring matching
    would turn explicit entity filters into a pollution source (e.g. ``api``
    matching every API-ish note).
    """

    candidate_norms = {normalize_text(entity).strip() for entity in candidate_entities if entity}
    matches: list[str] = []
    for entity in requested_entities:
        normalized = normalize_text(entity).strip()
        if normalized and normalized in candidate_norms:
            matches.append(entity)
    return matches


def recency_weight(created_at: datetime | None) -> float:
    if created_at is None:
        return 0.0
    age_days = max(
        0.0, (datetime.now(created_at.tzinfo) - created_at).total_seconds() / 86_400.0
    )
    return 1.0 / (1.0 + age_days / 14.0)


def score_chunks(
    candidates: Iterable[Any],
    *,
    query: str,
    embedding_provider: EmbeddingProvider,
    entities: Sequence[str] | None = None,
) -> list[ScoredChunk]:
    tokens = set(tokenize_query(query))
    requested_entities = [entity for entity in entities or [] if normalize_text(entity).strip()]
    embedded_query = embedding_provider.embed(query) if tokens else None
    scored: list[ScoredChunk] = []

    for chunk in candidates:
        chunk_tokens = set(tokenize_query(chunk.chunk_text))
        matched_terms = sorted(tokens & chunk_tokens)
        lexical_score = len(matched_terms) / max(1, len(tokens)) if tokens else 0.0

        vector_score = 0.0
        emb = chunk.embedding
        if emb is None:
            raw_emb = chunk.extra_metadata.get("embedding") if chunk.extra_metadata else None
            emb = raw_emb if isinstance(raw_emb, list) else None
        if embedded_query is not None and isinstance(emb, list):
            vector_values = [float(v) for v in emb if isinstance(v, (int, float))]
            vector_score = max(0.0, cosine(embedded_query, vector_values))

        matched_entities = entity_match_terms(getattr(chunk, "entities", []), requested_entities)
        entity_score = 1.0 if matched_entities else 0.0
        recency_score = recency_weight(chunk.created_at)
        if requested_entities:
            final_score = lexical_score * 0.35 + vector_score * 0.25 + entity_score * 0.35 + recency_score * 0.05
        else:
            final_score = lexical_score * 0.55 + vector_score * 0.35 + recency_score * 0.10

        if entity_score > 0.0 and entity_score >= max(lexical_score, vector_score):
            lane = "entity"
        elif lexical_score >= max(vector_score, 0.01):
            lane = "lexical"
        elif vector_score > 0.0:
            lane = "semantic"
        else:
            lane = "recency"

        explanation_parts: list[str] = []
        if matched_entities:
            explanation_parts.append("entity match: " + ", ".join(matched_entities[:8]))
        if matched_terms:
            explanation_parts.append("lexical match: " + ", ".join(matched_terms[:8]))
        if vector_score > 0.0:
            explanation_parts.append(f"semantic/vector similarity {vector_score:.2f}")
        if recency_score > 0.0:
            explanation_parts.append(f"recency boost {recency_score:.2f}")

        scored.append(
            ScoredChunk(
                chunk=chunk,
                score=RetrievalScore(
                    lexical_score=lexical_score,
                    vector_score=vector_score,
                    entity_score=entity_score,
                    recency_score=recency_score,
                    final_score=final_score,
                    source_lane=lane,
                    explanation="; ".join(explanation_parts) or "ranked by recency",
                    matched_terms=matched_terms,
                    matched_entities=matched_entities,
                ),
            )
        )

    return scored


def filter_non_matching_chunks(
    scored: Sequence[ScoredChunk], *, query: str, entities: Sequence[str] | None = None
) -> list[ScoredChunk]:
    """Drop recency-only noise for non-empty queries.

    Recency is a boost, not a retrieval lane by itself when the user asked a
    specific query. Without this guard, any recent chunk can leak into recall.
    Entity matches are allowed when the caller supplied explicit entity scope.
    """

    if not tokenize_query(query) and not entities:
        return list(scored)
    return [
        item
        for item in scored
        if item.score.lexical_score > 0.0 or item.score.vector_score > 0.0 or item.score.entity_score > 0.0
    ]


def summarize_chunk_fusion(scored: Sequence[ScoredChunk]) -> dict[str, object]:
    lane_counts = Counter(item.score.source_lane for item in scored)
    return {
        "lane_counts": dict(sorted(lane_counts.items())),
        "chunk_score_fields": [
            "lexical_score",
            "vector_score",
            "entity_score",
            "recency_score",
            "final_score",
        ],
    }
