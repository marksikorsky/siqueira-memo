"""Context tree / budgeted context-pack service. Roadmap Phase 6.

This first slice is intentionally non-persistent: it builds a tree view from an
already retrieved ``ContextPack`` and produces a scoped/budgeted preview. The DB
model/UI can come later without blocking the core context-selection semantics.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from siqueira_memo.config import Settings, get_settings
from siqueira_memo.models.constants import RECALL_MODE_BALANCED, RECALL_MODE_FAST, STATUS_ACTIVE
from siqueira_memo.schemas.common import SourceRef
from siqueira_memo.schemas.recall import (
    ContextPack,
    RecallChunk,
    RecallDecision,
    RecallFact,
    RecallSummary,
)
from siqueira_memo.services.redaction_service import redact
from siqueira_memo.utils.canonical import normalize_text

ContextNodeKind = Literal["global", "project"]


@dataclass
class ContextTreeNode:
    path: str
    kind: ContextNodeKind
    decisions_count: int = 0
    facts_count: int = 0
    chunks_count: int = 0
    summaries_count: int = 0
    secret_count: int = 0
    token_estimate: int = 0

    @property
    def total_count(self) -> int:
        return self.decisions_count + self.facts_count + self.chunks_count + self.summaries_count


@dataclass
class ContextTree:
    nodes: list[ContextTreeNode] = field(default_factory=list)


@dataclass
class ContextTreePreview:
    pack: ContextPack
    tree: ContextTree
    selected_paths: list[str]
    budget_tokens: int


@dataclass
class ContextTreeService:
    settings: Settings | None = None

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()

    def build_tree(self, pack: ContextPack, *, exclude_secrets: bool = True) -> ContextTree:
        tree = ContextTree()
        nodes_by_path: dict[str, ContextTreeNode] = {}

        def node_for(path: str, kind: ContextNodeKind) -> ContextTreeNode:
            node = nodes_by_path.get(path)
            if node is None:
                node = ContextTreeNode(path=path, kind=kind)
                nodes_by_path[path] = node
                tree.nodes.append(node)
            return node

        for decision in pack.decisions:
            path, kind = _decision_path(decision)
            node = node_for(path, kind)
            if _decision_is_secret(decision):
                node.secret_count += 1
                if exclude_secrets:
                    continue
            node.decisions_count += 1
            node.token_estimate += _rough_tokens(decision.decision) + _rough_tokens(decision.rationale)

        for fact in pack.facts:
            path, kind = _fact_path(fact)
            node = node_for(path, kind)
            if _fact_is_secret(fact):
                node.secret_count += 1
                if exclude_secrets:
                    continue
            node.facts_count += 1
            node.token_estimate += _rough_tokens(fact.statement)

        for summary in pack.summaries:
            path = "global/summaries/recent"
            node = node_for(path, "global")
            node.summaries_count += 1
            node.token_estimate += _rough_tokens(summary.summary_short)

        for chunk in pack.chunks:
            path, kind = _chunk_path(chunk)
            node = node_for(path, kind)
            if _chunk_is_secret(chunk):
                node.secret_count += 1
                if exclude_secrets:
                    continue
            node.chunks_count += 1
            node.token_estimate += _rough_tokens(chunk.chunk_text)

        if exclude_secrets:
            tree.nodes = [node for node in tree.nodes if node.total_count > 0]
        return tree

    def preview_context_pack(
        self,
        pack: ContextPack,
        *,
        mode: str,
        project: str | None = None,
        topic: str | None = None,
        exclude_secrets: bool = True,
    ) -> ContextTreePreview:
        assert self.settings is not None
        scoped = self._scope_pack(pack, project=project, topic=topic)
        conflict_count = len(scoped.conflicts)
        if conflict_count:
            scoped.conflicts = []
            scoped.warnings = [
                *scoped.warnings,
                f"{conflict_count} conflict item(s) excluded from context tree preview; call explicit recall for conflicts",
            ]
        if exclude_secrets:
            secret_count = _drop_secret_records(scoped)
            if secret_count:
                scoped.warnings = [*scoped.warnings, f"{secret_count} secret item(s) excluded from context tree preview"]

        scoped.answer_context = _answer_context(scoped)
        scoped.source_snippets = list(scoped.source_snippets)[: self.settings.prefetch_max_source_snippets]
        tree = self.build_tree(scoped, exclude_secrets=exclude_secrets)
        selected_paths = [node.path for node in tree.nodes if node.total_count > 0]
        budget = self._budget_for_mode(mode)
        trimmed = self._trim_to_budget(scoped, budget)
        trimmed.mode = mode
        return ContextTreePreview(pack=trimmed, tree=tree, selected_paths=selected_paths, budget_tokens=budget)

    def _scope_pack(self, pack: ContextPack, *, project: str | None, topic: str | None) -> ContextPack:
        out = copy.deepcopy(pack)
        out.decisions = [d for d in out.decisions if _in_scope(d.project, d.topic, project=project, topic=topic)]
        out.facts = [f for f in out.facts if _in_scope(f.project, f.topic, project=project, topic=topic)]
        out.chunks = [c for c in out.chunks if _in_scope(c.project, c.topic, project=project, topic=topic)]
        # Summaries are scope-specific elsewhere; keep them only when no explicit project/topic is requested.
        if project is not None or topic is not None:
            for decision in out.decisions:
                decision.sources = []
            for fact in out.facts:
                fact.sources = []
            out.summaries = []
            # SourceRef has no project/topic metadata today. Keeping unscoped snippets in a scoped
            # prefetch would leak unrelated provenance from another project.
            out.source_snippets = []
        return out

    def _budget_for_mode(self, mode: str) -> int:
        assert self.settings is not None
        if mode == RECALL_MODE_FAST:
            return self.settings.prefetch_fast_budget_tokens
        if mode == RECALL_MODE_BALANCED:
            return self.settings.prefetch_balanced_budget_tokens
        # Explicit deep/forensic preview can be bigger but still bounded.
        return self.settings.prefetch_balanced_budget_tokens * 2

    def _trim_to_budget(self, pack: ContextPack, budget_tokens: int) -> ContextPack:
        out = copy.deepcopy(pack)
        original_sources = list(out.source_snippets)

        decisions_kept: list[RecallDecision] = []
        facts_kept: list[RecallFact] = []
        summaries_kept: list[RecallSummary] = []
        chunks_kept: list[RecallChunk] = []
        sources_kept: list[SourceRef] = []
        exhausted = False

        for decision in out.decisions:
            decision_candidate = [*decisions_kept, decision]
            if _pack_token_estimate(decision_candidate, facts_kept, summaries_kept, chunks_kept, sources_kept) > budget_tokens:
                exhausted = True
                break
            decisions_kept = decision_candidate

        if not exhausted:
            for fact in out.facts:
                fact_candidate = [*facts_kept, fact]
                if _pack_token_estimate(decisions_kept, fact_candidate, summaries_kept, chunks_kept, sources_kept) > budget_tokens:
                    exhausted = True
                    break
                facts_kept = fact_candidate

        if not exhausted:
            for summary in out.summaries:
                summary_candidate = [*summaries_kept, summary]
                if _pack_token_estimate(decisions_kept, facts_kept, summary_candidate, chunks_kept, sources_kept) > budget_tokens:
                    exhausted = True
                    break
                summaries_kept = summary_candidate

        if not exhausted:
            for chunk in out.chunks:
                chunk_candidate = [*chunks_kept, chunk]
                if _pack_token_estimate(decisions_kept, facts_kept, summaries_kept, chunk_candidate, sources_kept) > budget_tokens:
                    exhausted = True
                    break
                chunks_kept = chunk_candidate

        if not exhausted:
            for source in original_sources:
                source_candidate = [*sources_kept, source]
                if _pack_token_estimate(decisions_kept, facts_kept, summaries_kept, chunks_kept, source_candidate) > budget_tokens:
                    break
                sources_kept = source_candidate

        out.decisions = decisions_kept
        out.facts = facts_kept
        out.summaries = summaries_kept
        out.chunks = chunks_kept
        out.source_snippets = sources_kept
        out.answer_context = _answer_context(out)
        out.token_estimate = _pack_token_estimate(
            out.decisions, out.facts, out.summaries, out.chunks, out.source_snippets
        )
        dropped = (
            len(pack.decisions) - len(out.decisions)
            + len(pack.facts) - len(out.facts)
            + len(pack.summaries) - len(out.summaries)
            + len(pack.chunks) - len(out.chunks)
            + len(original_sources) - len(out.source_snippets)
        )
        if dropped:
            out.warnings = [*out.warnings, f"context tree budget trimmed {dropped} lower-priority item(s)"]
        return out


def _in_scope(item_project: str | None, item_topic: str | None, *, project: str | None, topic: str | None) -> bool:
    if item_project is None:
        return True
    if project is not None and item_project != project:
        return False
    return not (topic is not None and item_topic != topic)


def _decision_path(decision: RecallDecision) -> tuple[str, ContextNodeKind]:
    if decision.project:
        return f"projects/{_slug(decision.project)}/{_slug(decision.topic or 'general')}", "project"
    return f"global/agent/{_slug(decision.topic or 'behavior')}", "global"


def _fact_path(fact: RecallFact) -> tuple[str, ContextNodeKind]:
    if fact.project:
        return f"projects/{_slug(fact.project)}/{_slug(fact.topic or 'general')}", "project"
    return f"global/user/{_slug(fact.topic or 'preferences')}", "global"


def _chunk_path(chunk: RecallChunk) -> tuple[str, ContextNodeKind]:
    if chunk.project:
        return f"projects/{_slug(chunk.project)}/{_slug(chunk.topic or 'chunks')}", "project"
    return f"global/sources/{_slug(chunk.topic or 'chunks')}", "global"


def _slug(value: str) -> str:
    normalized = normalize_text(value).strip().lower()
    normalized = re.sub(r"[^a-z0-9а-яё._:-]+", "-", normalized, flags=re.IGNORECASE).strip("-")
    return normalized or "general"


def _drop_secret_records(pack: ContextPack) -> int:
    original = len(pack.decisions) + len(pack.facts) + len(pack.summaries) + len(pack.chunks) + len(pack.source_snippets)
    pack.decisions = [d for d in pack.decisions if not _decision_is_secret(d)]
    pack.facts = [f for f in pack.facts if not _fact_is_secret(f)]
    pack.summaries = [s for s in pack.summaries if not _summary_is_secret(s)]
    pack.chunks = [c for c in pack.chunks if not _chunk_is_secret(c)]
    pack.source_snippets = [s for s in pack.source_snippets if not _source_ref_is_secret(s)]
    new = len(pack.decisions) + len(pack.facts) + len(pack.summaries) + len(pack.chunks) + len(pack.source_snippets)
    return max(0, original - new)


def _decision_is_secret(decision: RecallDecision) -> bool:
    return (
        _excluded_sensitivity(decision.sensitivity)
        or decision.secret_masked
        or _text_has_secret(decision.decision, decision.rationale)
        or _record_has_secret(decision)
    )


def _fact_is_secret(fact: RecallFact) -> bool:
    return (
        _excluded_sensitivity(fact.sensitivity)
        or fact.secret_masked
        or _text_has_secret(fact.statement, fact.object, fact.masked_preview)
        or _record_has_secret(fact)
    )


def _summary_is_secret(summary: RecallSummary) -> bool:
    return _text_has_secret(summary.summary_short) or _record_has_secret(summary)


def _chunk_is_secret(chunk: RecallChunk) -> bool:
    return _excluded_sensitivity(chunk.sensitivity) or _text_has_secret(chunk.chunk_text) or _record_has_secret(chunk)


def _source_ref_is_secret(source: SourceRef) -> bool:
    return _text_has_secret(source.snippet) or _record_has_secret(source)


def _excluded_sensitivity(sensitivity: str) -> bool:
    return sensitivity in {"secret", "sensitive"}


def _text_has_secret(*values: str | None) -> bool:
    return any(redact(value or "").matches > 0 for value in values)


def _record_has_secret(record: Any) -> bool:
    dump = getattr(record, "model_dump", None)
    if callable(dump):
        return _value_has_secret(dump(mode="json"))
    return _text_has_secret(str(record))


def _value_has_secret(value: Any) -> bool:
    if isinstance(value, str):
        return _text_has_secret(value)
    if isinstance(value, dict):
        return any(_value_has_secret(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return any(_value_has_secret(item) for item in value)
    return False


def _answer_context(pack: ContextPack) -> str:
    parts: list[str] = []
    active_decisions = [d for d in pack.decisions if d.status == STATUS_ACTIVE]
    if active_decisions:
        parts.append("Active decisions:\n" + "\n".join(f"- {redact(d.decision).redacted}" for d in active_decisions[:5]))
    active_facts = [f for f in pack.facts if f.status == STATUS_ACTIVE]
    if active_facts:
        parts.append("Known facts:\n" + "\n".join(f"- {redact(f.statement).redacted}" for f in active_facts[:5]))
    if pack.summaries:
        parts.append("Recent summary: " + redact(pack.summaries[0].summary_short).redacted)
    return "\n\n".join(parts)


def _pack_token_estimate(
    decisions: Sequence[RecallDecision],
    facts: Sequence[RecallFact],
    summaries: Sequence[RecallSummary],
    chunks: Sequence[RecallChunk],
    source_snippets: Sequence[SourceRef],
) -> int:
    pack = ContextPack(
        decisions=list(decisions),
        facts=list(facts),
        summaries=list(summaries),
        chunks=list(chunks),
        source_snippets=list(source_snippets),
    )
    pack.answer_context = _answer_context(pack)
    total = _rough_tokens(pack.answer_context)
    total += sum(_rough_tokens(decision.decision) + _rough_tokens(decision.rationale) for decision in decisions)
    total += sum(_rough_tokens(fact.statement) for fact in facts)
    total += sum(_rough_tokens(summary.summary_short) for summary in summaries)
    total += sum(_rough_tokens(chunk.chunk_text) for chunk in chunks)
    total += sum(_rough_tokens(getattr(source, "snippet", None) or "") for source in source_snippets)
    return total


def _rough_tokens(text: str) -> int:
    return max(1, len(text.split())) if text else 0
