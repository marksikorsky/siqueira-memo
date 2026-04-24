"""Deterministic eval runner. Plan §11 / §26.

The runner seeds a fresh SQLite in-memory DB, promotes the golden decisions +
facts via the normal extraction service, and then issues golden recall
questions. A JSON report is produced that unit tests and the CLI both
consume.

Because the runner avoids external services (mock embedding, mock reranker,
manual extraction) it is deterministic and safe to run in CI.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.config import Settings, settings_for_tests
from siqueira_memo.db import (
    create_all_for_tests,
    dispose_engines,
    drop_all_for_tests,
    get_session_factory,
)
from siqueira_memo.evals.golden_questions import (
    GOLDEN_QUESTIONS,
    GOLDEN_SEED_DECISIONS,
    GOLDEN_SEED_FACTS,
    GoldenQuestion,
)
from siqueira_memo.models import EvalRun
from siqueira_memo.schemas.memory import RememberRequest
from siqueira_memo.schemas.recall import RecallRequest
from siqueira_memo.services.embedding_service import MockEmbeddingProvider
from siqueira_memo.services.extraction_service import ExtractionService
from siqueira_memo.services.retrieval_service import RetrievalService


@dataclass
class EvalResult:
    id: str
    question: str
    passed: bool
    score: float
    missing_terms: list[str]
    candidates_returned: int
    latency_ms: int


@dataclass
class EvalReport:
    profile_id: str
    suite: str
    pass_rate: float
    total: int
    passed: int
    failed: int
    started_at: str
    finished_at: str
    results: list[EvalResult]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [asdict(r) for r in self.results]
        return payload


async def _seed(session: AsyncSession, profile_id: str) -> None:
    svc = ExtractionService(profile_id=profile_id)
    for decision in GOLDEN_SEED_DECISIONS:
        await svc.remember(
            session,
            RememberRequest(
                kind="decision",
                statement=decision.decision,
                topic=decision.topic,
                project=decision.project,
                rationale=decision.rationale,
                confidence=0.95,
            ),
        )
    for fact in GOLDEN_SEED_FACTS:
        await svc.remember(
            session,
            RememberRequest(
                kind="fact",
                subject=fact.subject,
                predicate=fact.predicate,
                object=fact.object,
                statement=fact.statement,
                project=fact.project,
                topic=fact.topic,
                confidence=fact.confidence,
            ),
        )


def _missing_terms(question: GoldenQuestion, rendered: str) -> list[str]:
    return [t for t in question.expected_contains if t.lower() not in rendered.lower()]


async def _evaluate(
    session: AsyncSession, profile_id: str, question: GoldenQuestion
) -> EvalResult:
    start = time.perf_counter()
    svc = RetrievalService(profile_id=profile_id, embedding_provider=MockEmbeddingProvider())
    recall = await svc.recall(
        session,
        RecallRequest(
            query=question.question,
            project=question.project,
            topic=question.topic,
            mode="balanced",
        ),
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    pack = recall.context_pack
    rendered = _render(pack)
    missing = _missing_terms(question, rendered)
    passed = not missing
    score = 1.0 - (len(missing) / max(1, len(question.expected_contains)))
    session.add(
        EvalRun(
            id=uuid.uuid4(),
            profile_id=profile_id,
            suite="golden",
            question=question.question,
            passed=passed,
            score=score,
            expected_contains=list(question.expected_contains),
            missing_terms=missing,
            candidates_returned=len(pack.chunks) + len(pack.decisions) + len(pack.facts),
            latency_ms=latency_ms,
            extra_metadata={"id": question.id},
        )
    )
    return EvalResult(
        id=question.id,
        question=question.question,
        passed=passed,
        score=score,
        missing_terms=missing,
        candidates_returned=len(pack.chunks) + len(pack.decisions) + len(pack.facts),
        latency_ms=latency_ms,
    )


def _render(pack: Any) -> str:
    parts: list[str] = [pack.answer_context]
    for d in pack.decisions:
        parts.append(d.decision)
        parts.append(d.rationale)
    for f in pack.facts:
        parts.append(f.statement)
    for c in pack.chunks:
        parts.append(c.chunk_text)
    return "\n".join(parts)


async def run_golden(
    questions: Iterable[GoldenQuestion] = GOLDEN_QUESTIONS,
    *,
    settings: Settings | None = None,
    profile_id: str = "golden-eval",
) -> EvalReport:
    settings = settings or settings_for_tests()
    await create_all_for_tests(settings)
    factory = get_session_factory(settings)
    started = datetime.now(UTC)
    try:
        async with factory() as seed_session:
            await _seed(seed_session, profile_id)
            await seed_session.commit()
        results: list[EvalResult] = []
        async with factory() as session:
            for question in questions:
                results.append(await _evaluate(session, profile_id, question))
            await session.commit()
    finally:
        await drop_all_for_tests(settings)
        await dispose_engines()
    finished = datetime.now(UTC)
    passed = sum(1 for r in results if r.passed)
    return EvalReport(
        profile_id=profile_id,
        suite="golden",
        pass_rate=passed / max(1, len(results)),
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        results=results,
    )


def _main() -> int:  # pragma: no cover
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(prog="siqueira-evals")
    parser.add_argument("--suite", default="golden")
    parser.add_argument("--output", default="-")
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    args = parser.parse_args()

    report = asyncio.run(run_golden())
    body = json.dumps(report.to_dict(), indent=2, default=str)
    if args.output == "-":
        print(body)
    else:
        from pathlib import Path

        Path(args.output).write_text(body, encoding="utf-8")
    return 0 if report.pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
