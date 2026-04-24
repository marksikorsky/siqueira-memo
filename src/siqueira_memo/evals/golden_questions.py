"""Golden eval question corpus. Plan §11.1."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class SeedFact:
    subject: str
    predicate: str
    object: str
    statement: str
    project: str | None = None
    topic: str | None = None
    confidence: float = 0.95


@dataclass(frozen=True)
class SeedDecision:
    project: str | None
    topic: str
    decision: str
    rationale: str
    decided_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    expected_contains: tuple[str, ...]
    expected_sources_required: bool = True
    project: str | None = None
    topic: str | None = None


GOLDEN_SEED_DECISIONS: tuple[SeedDecision, ...] = (
    SeedDecision(
        project="siqueira-memo",
        topic="memory integration",
        decision="Use Hermes MemoryProvider plugin as the primary memory integration.",
        rationale="MemoryProvider is native to Hermes; MCP is an adapter only.",
    ),
    SeedDecision(
        project="siqueira-memo",
        topic="mcp role",
        decision="Do not use MCP as the primary integration.",
        rationale="MCP may be added as an adapter later; it is not primary.",
    ),
    SeedDecision(
        project="siqueira-memo",
        topic="hindsight role",
        decision="Hindsight is offline/import-only; it is not a live fallback provider.",
        rationale="Hermes allows one external live memory provider; Siqueira is active.",
    ),
    SeedDecision(
        project="siqueira-memo",
        topic="raw archive",
        decision="Store raw messages as the source of truth; derived memory is rebuildable.",
        rationale="Plan §1.1.",
    ),
)


GOLDEN_SEED_FACTS: tuple[SeedFact, ...] = (
    SeedFact(
        subject="siqueira-memo",
        predicate="primary_integration",
        object="MemoryProvider plugin",
        statement="Siqueira Memo integrates primarily through a Hermes MemoryProvider plugin.",
        project="siqueira-memo",
        topic="memory integration",
    ),
    SeedFact(
        subject="siqueira-memo",
        predicate="uses_database",
        object="PostgreSQL + pgvector",
        statement="Siqueira Memo stores memory in PostgreSQL with pgvector-backed embeddings.",
        project="siqueira-memo",
        topic="storage",
    ),
    SeedFact(
        subject="compact_memory",
        predicate="role",
        object="bootloader",
        statement="Compact memory remains a bootloader/hard-preferences layer only.",
        project="siqueira-memo",
        topic="memory architecture",
    ),
)


GOLDEN_QUESTIONS: tuple[GoldenQuestion, ...] = (
    GoldenQuestion(
        id="memory_primary",
        question="Что мы решили про primary integration для памяти?",
        expected_contains=("MemoryProvider plugin", "primary"),
        project="siqueira-memo",
        topic="memory integration",
    ),
    GoldenQuestion(
        id="mcp_role",
        question="Какую роль играет MCP в памяти?",
        expected_contains=("не", "primary"),
        project="siqueira-memo",
        topic="mcp role",
    ),
    GoldenQuestion(
        id="hindsight_role",
        question="Hindsight живой провайдер или нет?",
        expected_contains=("offline", "import"),
        project="siqueira-memo",
        topic="hindsight role",
    ),
    GoldenQuestion(
        id="storage_choice",
        question="В какой базе хранится память?",
        expected_contains=("PostgreSQL", "pgvector"),
        project="siqueira-memo",
        topic="storage",
    ),
    GoldenQuestion(
        id="compact_memory_role",
        question="Какая роль у compact memory?",
        expected_contains=("bootloader",),
        project="siqueira-memo",
        topic="memory architecture",
    ),
)
