"""Markdown export. Plan §9.2 Task 9.3."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from siqueira_memo.models import Decision, Fact, MemoryConflict, MemoryEvent, SessionSummary
from siqueira_memo.models.constants import (
    CONFLICT_STATUS_OPEN,
    STATUS_ACTIVE,
)
from siqueira_memo.services.markdown_export import ExportFilter, export_markdown


async def _mk_event(session, profile: str) -> uuid.UUID:
    event = MemoryEvent(
        id=uuid.uuid4(),
        event_type="decision_recorded",
        source="test",
        actor="test",
        profile_id=profile,
        payload={"event_type": "decision_recorded"},
    )
    session.add(event)
    await session.flush()
    return event.id


@pytest.mark.asyncio
async def test_export_markdown_includes_active_decisions_and_facts(db, session):
    profile = "p1"
    event_id = await _mk_event(session, profile)
    session.add_all(
        [
            Decision(
                profile_id=profile,
                project="siqueira-memo",
                topic="memory integration",
                decision="Use Hermes MemoryProvider plugin as primary integration",
                context="ctx",
                rationale="MemoryProvider is native",
                canonical_key="k1",
                status=STATUS_ACTIVE,
                decided_at=datetime.now(UTC),
                source_event_ids=[event_id],
            ),
            Fact(
                profile_id=profile,
                subject="siqueira-memo",
                predicate="primary_integration",
                object="plugin",
                statement="Siqueira Memo uses the Hermes MemoryProvider plugin.",
                canonical_key="f1",
                project="siqueira-memo",
                topic="memory integration",
                status=STATUS_ACTIVE,
                confidence=0.9,
                source_event_ids=[event_id],
            ),
            SessionSummary(
                profile_id=profile,
                session_id="s1",
                summary_short="Discussed memory architecture.",
                summary_long="Long summary body here.",
                source_event_ids=[event_id],
                model="mock",
                version=1,
            ),
            MemoryConflict(
                profile_id=profile,
                conflict_type="decision_decision",
                left_type="decision",
                left_id=uuid.uuid4(),
                right_type="decision",
                right_id=uuid.uuid4(),
                severity="high",
                status=CONFLICT_STATUS_OPEN,
                resolution_hint="newer supersedes older",
                confidence=0.9,
            ),
        ]
    )
    await session.flush()

    markdown = await export_markdown(
        session, ExportFilter(profile_id=profile, project="siqueira-memo")
    )
    assert "# Siqueira Memo — siqueira-memo" in markdown
    assert "MemoryProvider plugin" in markdown
    assert "## Active facts" in markdown
    assert "Discussed memory architecture" in markdown
    assert "## Open conflicts" in markdown
    # Must not leak internal schema secrets (here just a generic check).
    assert "content_raw" not in markdown


@pytest.mark.asyncio
async def test_export_markdown_is_empty_friendly(db, session):
    out = await export_markdown(session, ExportFilter(profile_id="nobody"))
    assert out.startswith("# Siqueira Memo")
    assert "Active decisions" not in out
    assert out.strip().endswith("no raw secrets.*")
