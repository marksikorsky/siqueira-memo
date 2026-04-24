"""Retention enforcement + audit log tests. Plan §9 / §25."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from siqueira_memo.config import settings_for_tests
from siqueira_memo.models import RetrievalLog
from siqueira_memo.schemas.memory import CorrectRequest, ForgetRequest, RememberRequest
from siqueira_memo.services.deletion_service import DeletionService
from siqueira_memo.services.extraction_service import ExtractionService
from siqueira_memo.services.retention_service import AuditLog, RetentionService


@pytest.mark.asyncio
async def test_retention_deletes_old_retrieval_logs(db, session):
    settings = settings_for_tests(retention_retrieval_logs_days=30)
    now = datetime.now(UTC)
    old = RetrievalLog(
        id=uuid.uuid4(),
        profile_id="p1",
        query="old",
        created_at=now - timedelta(days=60),
    )
    recent = RetrievalLog(
        id=uuid.uuid4(),
        profile_id="p1",
        query="recent",
        created_at=now - timedelta(days=5),
    )
    session.add_all([old, recent])
    await session.flush()

    svc = RetentionService(profile_id="p1", settings=settings)
    report = await svc.enforce(session)
    assert report.retrieval_logs_deleted == 1

    remaining = (await session.execute(select(RetrievalLog))).scalars().all()
    assert {r.query for r in remaining} == {"recent"}


@pytest.mark.asyncio
async def test_retention_isolates_profiles(db, session):
    settings = settings_for_tests(retention_retrieval_logs_days=10)
    session.add_all(
        [
            RetrievalLog(
                id=uuid.uuid4(),
                profile_id="p1",
                query="old p1",
                created_at=datetime.now(UTC) - timedelta(days=20),
            ),
            RetrievalLog(
                id=uuid.uuid4(),
                profile_id="p2",
                query="old p2",
                created_at=datetime.now(UTC) - timedelta(days=20),
            ),
        ]
    )
    await session.flush()
    await RetentionService(profile_id="p1", settings=settings).enforce(session)
    remaining = {r.query for r in (await session.execute(select(RetrievalLog))).scalars().all()}
    assert "old p1" not in remaining
    assert "old p2" in remaining


@pytest.mark.asyncio
async def test_audit_log_lists_delete_and_correction(db, session):
    profile = "p1"
    ex = ExtractionService(profile_id=profile)
    remember = await ex.remember(
        session,
        RememberRequest(
            kind="fact", subject="s", predicate="p", object="o", statement="s p o"
        ),
    )
    await ex.apply_correction(
        session,
        CorrectRequest(
            target_type="fact",
            target_id=remember.id,
            correction_text="actually wrong",
        ),
    )
    # Now also hard-forget a fresh fact to get a memory_deleted event.
    other = await ex.remember(
        session,
        RememberRequest(
            kind="fact", subject="x", predicate="y", object="z", statement="x y z"
        ),
    )
    await DeletionService(profile_id=profile).forget(
        session,
        ForgetRequest(target_type="fact", target_id=other.id, mode="hard", reason="cleanup"),
    )

    entries = await AuditLog(profile_id=profile).fetch_deletion_events(session)
    kinds = [e.event_type for e in entries]
    assert "memory_deleted" in kinds
    assert "user_correction_received" in kinds
    assert "fact_invalidated" in kinds
    for entry in entries:
        # Never include raw deleted content — only metadata.
        assert entry.target_id is None or isinstance(entry.target_id, str)


def test_audit_log_never_returns_deleted_text(db):
    """Guardrail: the audit dataclass has no ``content`` field."""
    from siqueira_memo.services.retention_service import AuditEntry

    fields = AuditEntry.__dataclass_fields__
    assert "content" not in fields
    assert "content_raw" not in fields
