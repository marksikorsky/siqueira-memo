"""Golden eval pass-rate checks. Plan §11 / §26.2."""

from __future__ import annotations

import json

import pytest

from siqueira_memo.evals.runner import EvalReport, run_golden


@pytest.mark.asyncio
async def test_golden_suite_passes():
    report = await run_golden()
    assert isinstance(report, EvalReport)
    assert report.total == 5
    assert report.pass_rate >= 0.8, f"failing_ids={[r.id for r in report.results if not r.passed]}"
    # Serialisable for the CLI.
    blob = json.dumps(report.to_dict())
    assert "siqueira-memo" not in blob or "MemoryProvider" in blob


@pytest.mark.asyncio
async def test_golden_report_contains_expected_ids():
    report = await run_golden()
    ids = {r.id for r in report.results}
    assert ids == {
        "memory_primary",
        "mcp_role",
        "hindsight_role",
        "storage_choice",
        "compact_memory_role",
    }
