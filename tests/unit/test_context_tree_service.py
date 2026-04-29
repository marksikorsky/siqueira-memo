"""Context tree / budgeted context pack tests. Roadmap Phase 6."""

from __future__ import annotations

import uuid

from siqueira_memo.config import settings_for_tests
from siqueira_memo.models.constants import RECALL_MODE_BALANCED, STATUS_ACTIVE
from siqueira_memo.schemas.common import SourceRef
from siqueira_memo.schemas.recall import (
    ConflictEntry,
    ContextPack,
    RecallChunk,
    RecallDecision,
    RecallFact,
    RecallSummary,
)
from siqueira_memo.services.context_tree_service import ContextTreeService


def _fact(
    statement: str,
    *,
    project: str | None = None,
    topic: str | None = None,
    sensitivity: str = "internal",
    secret_masked: bool = False,
) -> RecallFact:
    return RecallFact(
        id=uuid.uuid4(),
        subject=statement.split()[0],
        predicate="states",
        object=statement,
        statement=statement,
        status=STATUS_ACTIVE,
        confidence=0.9,
        project=project,
        topic=topic,
        sensitivity=sensitivity,
        secret_masked=secret_masked,
    )


def _decision(
    text: str,
    *,
    project: str | None = None,
    topic: str = "architecture",
    sensitivity: str = "internal",
) -> RecallDecision:
    return RecallDecision(
        id=uuid.uuid4(),
        project=project,
        topic=topic,
        decision=text,
        rationale="needed for tests",
        status=STATUS_ACTIVE,
        reversible=True,
        decided_at="2026-01-01T00:00:00Z",
        sensitivity=sensitivity,
    )


def _chunk(text: str, *, project: str | None = None, topic: str | None = None) -> RecallChunk:
    return RecallChunk(
        id=uuid.uuid4(),
        source_type="message",
        source_id=uuid.uuid4(),
        chunk_text=text,
        score=0.5,
        project=project,
        topic=topic,
    )


def test_context_tree_preview_keeps_global_memory_and_selected_project_only():
    pack = ContextPack(
        facts=[
            _fact("User prefers direct Russian answers", project=None, topic="preferences"),
            _fact("Siqueira Memo uses trusted internal recall", project="siqueira-memo", topic="memory policy"),
            _fact("Clawik admin is light-first", project="clawik", topic="ui"),
        ]
    )

    preview = ContextTreeService(settings_for_tests()).preview_context_pack(
        pack,
        mode=RECALL_MODE_BALANCED,
        project="siqueira-memo",
    )

    statements = [fact.statement for fact in preview.pack.facts]
    assert "User prefers direct Russian answers" in statements
    assert "Siqueira Memo uses trusted internal recall" in statements
    assert "Clawik admin is light-first" not in statements
    assert preview.selected_paths == ["global/user/preferences", "projects/siqueira-memo/memory-policy"]


def test_context_tree_default_preview_excludes_secret_nodes_and_items():
    raw_secret = "sk-proj-" + "x" * 40
    pack = ContextPack(
        facts=[
            _fact(f"OpenAI admin key is {raw_secret}", project="siqueira-memo", topic="secrets", sensitivity="secret", secret_masked=False),
            _fact("Siqueira Memo deployment is healthy", project="siqueira-memo", topic="infra"),
        ]
    )

    preview = ContextTreeService(settings_for_tests()).preview_context_pack(
        pack,
        mode=RECALL_MODE_BALANCED,
        project="siqueira-memo",
    )

    assert all(f.sensitivity != "secret" for f in preview.pack.facts)
    assert raw_secret not in preview.pack.answer_context
    assert "projects/siqueira-memo/secrets" not in preview.selected_paths
    assert any("secret" in warning.lower() for warning in preview.pack.warnings)


def test_context_tree_budget_trims_lower_priority_chunks_first():
    settings = settings_for_tests(prefetch_balanced_budget_tokens=40)
    pack = ContextPack(
        decisions=[_decision("Use budgeted context tree packs", project="siqueira-memo")],
        facts=[_fact("Context tree keeps high priority facts", project="siqueira-memo", topic="retrieval")],
        chunks=[
            _chunk("low priority chunk " + "documentation " * 30, project="siqueira-memo", topic="retrieval"),
            _chunk("another low priority chunk " + "documentation " * 30, project="siqueira-memo", topic="retrieval"),
        ],
    )

    preview = ContextTreeService(settings).preview_context_pack(
        pack,
        mode=RECALL_MODE_BALANCED,
        project="siqueira-memo",
    )

    assert preview.pack.decisions
    assert preview.pack.facts
    assert preview.pack.chunks == []
    assert preview.pack.token_estimate <= settings.prefetch_balanced_budget_tokens
    assert any("budget" in warning.lower() for warning in preview.pack.warnings)


def test_context_tree_summary_reports_counts_per_node():
    pack = ContextPack(
        decisions=[_decision("Use Siqueira roadmap", project="siqueira-memo", topic="roadmap")],
        facts=[
            _fact("User prefers concise answers", topic="preferences"),
            _fact("Siqueira API is deployed", project="siqueira-memo", topic="infra"),
        ],
        chunks=[_chunk("Siqueira retrieval note", project="siqueira-memo", topic="retrieval")],
    )

    tree = ContextTreeService(settings_for_tests()).build_tree(pack)

    by_path = {node.path: node for node in tree.nodes}
    assert by_path["global/user/preferences"].facts_count == 1
    assert by_path["projects/siqueira-memo/roadmap"].decisions_count == 1
    assert by_path["projects/siqueira-memo/infra"].facts_count == 1
    assert by_path["projects/siqueira-memo/retrieval"].chunks_count == 1


def test_context_tree_prefetch_caps_source_snippets():
    settings = settings_for_tests(prefetch_max_source_snippets=3)
    pack = ContextPack(
        facts=[_fact("Siqueira Memo source snippet cap is enforced", project="siqueira-memo")],
        source_snippets=[SourceRef(event_id=str(uuid.uuid4()), snippet=f"source snippet {idx}") for idx in range(10)],
    )

    preview = ContextTreeService(settings).preview_context_pack(
        pack,
        mode=RECALL_MODE_BALANCED,
    )

    assert len(preview.pack.source_snippets) == 3


def test_context_tree_budget_does_not_keep_lower_priority_when_decision_overflows():
    settings = settings_for_tests(prefetch_balanced_budget_tokens=10)
    pack = ContextPack(
        decisions=[_decision("oversized decision " + "word " * 20, project="siqueira-memo")],
        facts=[_fact("small lower priority fact", project="siqueira-memo")],
    )

    preview = ContextTreeService(settings).preview_context_pack(
        pack,
        mode=RECALL_MODE_BALANCED,
        project="siqueira-memo",
    )

    assert preview.pack.decisions == []
    assert preview.pack.facts == []
    assert any("budget" in warning.lower() for warning in preview.pack.warnings)


def test_context_tree_token_estimate_counts_answer_context_and_source_snippets():
    settings = settings_for_tests(prefetch_balanced_budget_tokens=80, prefetch_max_source_snippets=2)
    pack = ContextPack(
        decisions=[_decision("Count answer context tokens", project="siqueira-memo")],
        facts=[_fact("Token estimate includes generated answer context", project="siqueira-memo")],
        source_snippets=[
            SourceRef(event_id=str(uuid.uuid4()), snippet="source snippet counted"),
            SourceRef(event_id=str(uuid.uuid4()), snippet="another source snippet counted"),
        ],
    )

    preview = ContextTreeService(settings).preview_context_pack(
        pack,
        mode=RECALL_MODE_BALANCED,
    )

    answer_tokens = len(preview.pack.answer_context.split())
    source_tokens = sum(len((snippet.snippet or "").split()) for snippet in preview.pack.source_snippets)
    assert preview.pack.token_estimate >= answer_tokens + source_tokens
    assert preview.pack.token_estimate <= settings.prefetch_balanced_budget_tokens


def test_context_tree_drops_secret_like_unmarked_structured_records_from_prefetch_pack():
    raw_secret = "sk-proj-" + "z" * 40
    pack = ContextPack(
        facts=[
            _fact(f"OpenAI unmarked key is {raw_secret}", project="siqueira-memo", topic="secrets"),
            _fact("Siqueira public fact remains", project="siqueira-memo", topic="infra"),
        ]
    )

    preview = ContextTreeService(settings_for_tests()).preview_context_pack(
        pack,
        mode=RECALL_MODE_BALANCED,
        project="siqueira-memo",
    )

    body = preview.pack.model_dump_json()
    assert raw_secret not in body
    assert [fact.statement for fact in preview.pack.facts] == ["Siqueira public fact remains"]


def test_context_tree_drops_unscoped_source_snippets_when_project_scope_selected():
    pack = ContextPack(
        facts=[_fact("Siqueira scoped fact", project="siqueira-memo", topic="infra")],
        source_snippets=[
            SourceRef(event_id=str(uuid.uuid4()), snippet="Clawik unrelated source snippet"),
            SourceRef(event_id=str(uuid.uuid4()), snippet="Siqueira source snippet without scope metadata"),
        ],
    )

    preview = ContextTreeService(settings_for_tests()).preview_context_pack(
        pack,
        mode=RECALL_MODE_BALANCED,
        project="siqueira-memo",
    )

    assert preview.pack.source_snippets == []


def test_context_tree_excludes_sensitive_facts_and_decisions_from_prefetch_pack():
    pack = ContextPack(
        decisions=[
            _decision("Sensitive deployment detail", project="siqueira-memo", sensitivity="sensitive"),
            _decision("Public architecture detail", project="siqueira-memo"),
        ],
        facts=[
            _fact("Sensitive billing detail", project="siqueira-memo", sensitivity="sensitive"),
            _fact("Public infra detail", project="siqueira-memo"),
        ],
    )

    preview = ContextTreeService(settings_for_tests()).preview_context_pack(
        pack,
        mode=RECALL_MODE_BALANCED,
        project="siqueira-memo",
    )

    body = preview.pack.model_dump_json()
    assert "Sensitive deployment detail" not in body
    assert "Sensitive billing detail" not in body
    assert [decision.decision for decision in preview.pack.decisions] == ["Public architecture detail"]
    assert [fact.statement for fact in preview.pack.facts] == ["Public infra detail"]


def test_context_tree_drops_conflicts_from_prefetch_pack_to_avoid_scope_secret_budget_bypass():
    raw_secret = "sk-proj-" + "c" * 40
    pack = ContextPack(
        facts=[_fact("Scoped public fact", project="siqueira-memo")],
        conflicts=[
            ConflictEntry(
                older={"project": "clawik", "statement": f"Clawik hidden value {raw_secret}"},
                newer={"project": "siqueira-memo", "statement": "Scoped public fact"},
                resolution="unresolved",
                severity="high",
            )
        ],
    )

    preview = ContextTreeService(settings_for_tests()).preview_context_pack(
        pack,
        mode=RECALL_MODE_BALANCED,
        project="siqueira-memo",
    )

    body = preview.pack.model_dump_json()
    assert preview.pack.conflicts == []
    assert raw_secret not in body
    assert "Clawik hidden value" not in body
    assert any("conflict" in warning.lower() for warning in preview.pack.warnings)


def test_context_tree_drops_secret_like_summary_long_from_prefetch_pack():
    raw_secret = "sk-proj-" + "l" * 40
    pack = ContextPack(
        summaries=[
            RecallSummary(
                id=uuid.uuid4(),
                scope="project",
                summary_short="Safe summary short text",
                summary_long=f"Unsafe long summary {raw_secret}",
            )
        ]
    )

    preview = ContextTreeService(settings_for_tests()).preview_context_pack(
        pack,
        mode=RECALL_MODE_BALANCED,
    )

    body = preview.pack.model_dump_json()
    assert preview.pack.summaries == []
    assert raw_secret not in body
    assert "Unsafe long summary" not in body


def test_context_tree_drops_secret_like_values_from_any_serialized_fact_field():
    raw_secret = "sk-proj-" + "s" * 40
    fact = _fact("Safe statement", project="siqueira-memo")
    fact.subject = raw_secret
    pack = ContextPack(facts=[fact])

    preview = ContextTreeService(settings_for_tests()).preview_context_pack(
        pack,
        mode=RECALL_MODE_BALANCED,
        project="siqueira-memo",
    )

    body = preview.pack.model_dump_json()
    assert preview.pack.facts == []
    assert raw_secret not in body


def test_context_tree_strips_nested_sources_when_project_scope_selected():
    decision = _decision("Scoped decision", project="siqueira-memo")
    decision.sources = [SourceRef(event_id=str(uuid.uuid4()), snippet="Clawik nested decision snippet")]
    fact = _fact("Scoped fact", project="siqueira-memo")
    fact.sources = [SourceRef(event_id=str(uuid.uuid4()), snippet="Clawik nested fact snippet")]
    pack = ContextPack(decisions=[decision], facts=[fact])

    preview = ContextTreeService(settings_for_tests()).preview_context_pack(
        pack,
        mode=RECALL_MODE_BALANCED,
        project="siqueira-memo",
    )

    body = preview.pack.model_dump_json()
    assert preview.pack.decisions[0].sources == []
    assert preview.pack.facts[0].sources == []
    assert "Clawik nested decision snippet" not in body
    assert "Clawik nested fact snippet" not in body
