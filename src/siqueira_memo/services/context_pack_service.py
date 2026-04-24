"""Context-pack shaping. Plan §5.8 / §33.5.

Given a ``ContextPack`` returned by the retrieval layer, shape it to satisfy
Hermes' prefetch budget so the provider never dumps uncontrolled context into
the prompt cache.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from siqueira_memo.config import Settings, get_settings
from siqueira_memo.models.constants import (
    RECALL_MODE_BALANCED,
    RECALL_MODE_DEEP,
    RECALL_MODE_FAST,
    RECALL_MODE_FORENSIC,
)
from siqueira_memo.schemas.recall import (
    ContextPack,
    RecallChunk,
    RecallDecision,
    RecallFact,
    RecallSummary,
)


@dataclass
class ContextPackShaper:
    settings: Settings | None = None

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()

    def shape_for_prefetch(self, pack: ContextPack, mode: str) -> ContextPack:
        assert self.settings is not None
        budget = self._budget_for_mode(mode)
        if budget is None:
            # Deep/forensic modes are never auto-injected into prefetch.
            return ContextPack(
                answer_context=(
                    "Deep/forensic context available — call `siqueira_memory_recall` "
                    "explicitly to retrieve the full pack."
                ),
                confidence=pack.confidence,
                warnings=[*pack.warnings, "deep/forensic mode not auto-injected"],
                mode=mode,
                latency_ms=pack.latency_ms,
                embedding_table=pack.embedding_table,
                token_estimate=0,
            )
        return self._trim_to_budget(pack, budget)

    def _budget_for_mode(self, mode: str) -> int | None:
        assert self.settings is not None
        if mode == RECALL_MODE_FAST:
            return self.settings.prefetch_fast_budget_tokens
        if mode == RECALL_MODE_BALANCED:
            return self.settings.prefetch_balanced_budget_tokens
        if mode in {RECALL_MODE_DEEP, RECALL_MODE_FORENSIC}:
            return None
        return self.settings.prefetch_balanced_budget_tokens

    def _trim_to_budget(self, pack: ContextPack, budget_tokens: int) -> ContextPack:
        assert self.settings is not None
        out = copy.deepcopy(pack)
        max_snippets = self.settings.prefetch_max_source_snippets
        out.source_snippets = list(out.source_snippets)[:max_snippets]

        # Priority order: decisions, facts, summaries, chunks.
        total = _rough_tokens(out.answer_context)
        decisions_kept: list[RecallDecision] = []
        for decision in out.decisions:
            total += _rough_tokens(decision.decision) + _rough_tokens(decision.rationale)
            if total > budget_tokens:
                break
            decisions_kept.append(decision)
        out.decisions = decisions_kept

        facts_kept: list[RecallFact] = []
        for fact in out.facts:
            total += _rough_tokens(fact.statement)
            if total > budget_tokens:
                break
            facts_kept.append(fact)
        out.facts = facts_kept

        summaries_kept: list[RecallSummary] = []
        for summary in out.summaries:
            total += _rough_tokens(summary.summary_short)
            if total > budget_tokens:
                break
            summaries_kept.append(summary)
        out.summaries = summaries_kept

        chunks_kept: list[RecallChunk] = []
        for chunk in out.chunks:
            total += _rough_tokens(chunk.chunk_text)
            if total > budget_tokens:
                break
            chunks_kept.append(chunk)
        out.chunks = chunks_kept

        out.token_estimate = total
        if total >= budget_tokens:
            out.warnings = [*out.warnings, "prefetch budget reached; call recall for more"]
        return out


def _rough_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))
