"""Remember / correct lifecycle. Plan §18.2.6 / §31.7 / §9.3."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from siqueira_memo.models import Decision, Fact, FactSource, MemoryEvent, MemoryVersion, Message
from siqueira_memo.models.constants import STATUS_ACTIVE, STATUS_INVALIDATED, STATUS_SUPERSEDED
from siqueira_memo.schemas.memory import CorrectRequest, ForgetRequest, RememberRequest
from siqueira_memo.services.deletion_service import DeletionService
from siqueira_memo.services.extraction_service import ExtractionService
from siqueira_memo.services.memory_version_service import MemoryVersionService, snapshot_memory


@pytest.mark.asyncio
async def test_remember_fact_is_idempotent(db, session):
    svc = ExtractionService(profile_id="p1")
    req = RememberRequest(
        kind="fact",
        subject="siqueira-memo",
        predicate="primary_integration",
        object="MemoryProvider plugin",
        statement="Siqueira Memo integrates via MemoryProvider plugin.",
        project="siqueira-memo",
        confidence=0.9,
    )
    first = await svc.remember(session, req)
    second = await svc.remember(session, req)
    assert first.id == second.id
    facts = (await session.execute(select(Fact))).scalars().all()
    active = [f for f in facts if f.status == STATUS_ACTIVE]
    assert len(active) == 1
    assert first.event_id in active[0].source_event_ids
    assert second.event_id in active[0].source_event_ids


@pytest.mark.asyncio
async def test_remember_decision_creates_event(db, session):
    svc = ExtractionService(profile_id="p1")
    req = RememberRequest(
        kind="decision",
        statement="Use Hermes MemoryProvider plugin as primary integration",
        topic="memory integration",
        project="siqueira-memo",
    )
    result = await svc.remember(session, req)
    ev = (
        await session.execute(select(MemoryEvent).where(MemoryEvent.id == result.event_id))
    ).scalar_one()
    assert ev.event_type == "decision_recorded"
    assert ev.payload["decision_id"] == str(result.id)
    decision = (
        await session.execute(select(Decision).where(Decision.id == result.id))
    ).scalar_one()
    assert result.event_id in decision.source_event_ids


@pytest.mark.asyncio
async def test_correction_supersedes_existing_fact(db, session):
    svc = ExtractionService(profile_id="p1")
    original = await svc.remember(
        session,
        RememberRequest(
            kind="fact",
            subject="shannon",
            predicate="primary_auth",
            object="api_key",
            statement="Shannon primary auth is API key.",
        ),
    )

    replacement = RememberRequest(
        kind="fact",
        subject="shannon",
        predicate="primary_auth",
        object="claude_oauth",
        statement="Shannon primary auth is Claude OAuth token.",
    )
    result = await svc.apply_correction(
        session,
        CorrectRequest(
            target_type="fact",
            target_id=original.id,
            correction_text="actually it's the OAuth token",
            replacement=replacement,
        ),
    )

    row = (await session.execute(select(Fact).where(Fact.id == original.id))).scalar_one()
    assert row.status == STATUS_SUPERSEDED
    assert result.replacement_id is not None
    assert row.superseded_by == result.replacement_id

    invalidated_events = (
        await session.execute(
            select(MemoryEvent).where(MemoryEvent.event_type == "fact_invalidated")
        )
    ).scalars().all()
    assert invalidated_events


@pytest.mark.asyncio
async def test_remember_fact_records_initial_memory_version(db, session):
    svc = ExtractionService(profile_id="p1")
    result = await svc.remember(
        session,
        RememberRequest(
            kind="fact",
            subject="siqueira-memo",
            predicate="versioning",
            object="enabled",
            statement="Siqueira Memo has memory versioning enabled.",
            project="siqueira-memo",
            confidence=0.88,
        ),
    )

    versions = (
        await session.execute(
            select(MemoryVersion).where(
                MemoryVersion.target_type == "fact",
                MemoryVersion.target_id == result.id,
            )
        )
    ).scalars().all()
    assert len(versions) == 1
    version = versions[0]
    assert version.version == 1
    assert version.operation == "create"
    assert version.event_id == result.event_id
    assert version.before_snapshot is None
    assert version.after_snapshot["statement"] == "Siqueira Memo has memory versioning enabled."
    assert version.after_snapshot["status"] == STATUS_ACTIVE
    assert version.after_snapshot["confidence"] == 0.88


@pytest.mark.asyncio
async def test_correction_records_supersede_version_and_diff(db, session):
    svc = ExtractionService(profile_id="p1")
    original = await svc.remember(
        session,
        RememberRequest(
            kind="fact",
            subject="shannon",
            predicate="primary_auth",
            object="api_key",
            statement="Shannon primary auth is API key.",
        ),
    )
    result = await svc.apply_correction(
        session,
        CorrectRequest(
            target_type="fact",
            target_id=original.id,
            correction_text="actually it is Claude OAuth",
            replacement=RememberRequest(
                kind="fact",
                subject="shannon",
                predicate="primary_auth",
                object="claude_oauth",
                statement="Shannon primary auth is Claude OAuth token.",
            ),
        ),
    )

    original_versions = (
        await session.execute(
            select(MemoryVersion)
            .where(MemoryVersion.target_id == original.id)
            .order_by(MemoryVersion.version.asc())
        )
    ).scalars().all()
    assert [v.operation for v in original_versions] == ["create", "supersede"]
    supersede = original_versions[1]
    assert supersede.before_snapshot["status"] == STATUS_ACTIVE
    assert supersede.after_snapshot["status"] == STATUS_SUPERSEDED
    assert supersede.after_snapshot["superseded_by"] == str(result.replacement_id)

    replacement_versions = (
        await session.execute(
            select(MemoryVersion).where(MemoryVersion.target_id == result.replacement_id)
        )
    ).scalars().all()
    assert len(replacement_versions) == 1
    assert replacement_versions[0].operation == "create"

    diff = await MemoryVersionService().diff(session, "fact", original.id, 1, 2, profile_id="p1")
    assert diff.target_id == original.id
    assert diff.from_version == 1
    assert diff.to_version == 2
    assert diff.changes["status"] == {"from": STATUS_ACTIVE, "to": STATUS_SUPERSEDED}
    assert diff.changes["superseded_by"] == {"from": None, "to": str(result.replacement_id)}


@pytest.mark.asyncio
async def test_rollback_restores_fact_snapshot_and_records_version(db, session):
    svc = ExtractionService(profile_id="p1")
    original = await svc.remember(
        session,
        RememberRequest(
            kind="fact",
            subject="shannon",
            predicate="primary_auth",
            object="api_key",
            statement="Shannon primary auth is API key.",
        ),
    )
    await svc.apply_correction(
        session,
        CorrectRequest(
            target_type="fact",
            target_id=original.id,
            correction_text="actually it is Claude OAuth",
            replacement=RememberRequest(
                kind="fact",
                subject="shannon",
                predicate="primary_auth",
                object="claude_oauth",
                statement="Shannon primary auth is Claude OAuth token.",
            ),
        ),
    )

    rollback = await MemoryVersionService().rollback(
        session,
        target_type="fact",
        target_id=original.id,
        to_version=1,
        profile_id="p1",
        actor="test",
        reason="operator rollback",
    )

    row = (await session.execute(select(Fact).where(Fact.id == original.id))).scalar_one()
    assert row.status == STATUS_ACTIVE
    assert row.superseded_by is None
    assert row.statement == "Shannon primary auth is API key."
    assert rollback.new_version == 3
    versions = (
        await session.execute(
            select(MemoryVersion)
            .where(MemoryVersion.target_id == original.id)
            .order_by(MemoryVersion.version.asc())
        )
    ).scalars().all()
    assert [v.operation for v in versions] == ["create", "supersede", "rollback"]
    assert versions[-1].rollback_to_version == 1


@pytest.mark.asyncio
async def test_sensitive_value_version_diff_returns_masked_values(db, session):
    svc = ExtractionService(profile_id="p1")
    first = await svc.remember(
        session,
        RememberRequest(
            kind="fact",
            subject="service",
            predicate="api_key",
            object="sk-test-secret-value-1234",
            statement="Service API key is sk-test-secret-value-1234.",
            metadata={
                "sensitivity": "secret",
                "secret_value": "sk-test-secret-value-1234",
            },
        ),
    )
    await svc.remember(
        session,
        RememberRequest(
            kind="fact",
            subject="service",
            predicate="api_key",
            object="sk-test-secret-value-1234",
            statement="Updated service API key is sk-test-secret-value-1234.",
            confidence=0.95,
            metadata={
                "sensitivity": "secret",
                "secret_value": "sk-test-secret-value-1234",
            },
        ),
    )

    diff = await MemoryVersionService().diff(session, "fact", first.id, 1, 2, profile_id="p1")

    rendered = str(diff.changes)
    assert "sk-test-secret-value-1234" not in rendered
    assert diff.changes["confidence"] == {"from": 0.9, "to": 0.95}

    await DeletionService(profile_id="p1").forget(
        session,
        ForgetRequest(target_type="fact", target_id=first.id, mode="hard")
    )
    hard_delete_diff = await MemoryVersionService().diff(
        session, "fact", first.id, 2, 3, profile_id="p1"
    )
    assert "sk-test-secret-value-1234" not in str(hard_delete_diff.changes)


@pytest.mark.asyncio
async def test_forget_message_records_versions_for_derived_fact_changes(db, session):
    event_id = uuid.uuid4()
    message_id = uuid.uuid4()
    session.add(
        MemoryEvent(
            id=event_id,
            event_type="message_ingested",
            source="test",
            actor="test",
            profile_id="p1",
            payload={},
        )
    )
    session.add(
        Message(
            id=message_id,
            event_id=event_id,
            profile_id="p1",
            session_id="s1",
            platform="telegram",
            role="user",
            content_raw="source message",
            content_redacted="source message",
            content_hash="hash-for-version-test",
        )
    )
    fact = await ExtractionService(profile_id="p1").remember(
        session,
        RememberRequest(
            kind="fact",
            subject="source",
            predicate="depends_on",
            object="message",
            statement="This fact depends on one source message.",
        ),
    )
    fact_row = (await session.execute(select(Fact).where(Fact.id == fact.id))).scalar_one()
    fact_row.source_message_ids = [message_id]
    await session.flush()

    result = await DeletionService(profile_id="p1").forget(
        session,
        ForgetRequest(target_type="message", target_id=message_id, mode="soft")
    )

    assert result.invalidated_facts == 1
    changed = (await session.execute(select(Fact).where(Fact.id == fact.id))).scalar_one()
    assert changed.status == STATUS_INVALIDATED
    versions = (
        await session.execute(
            select(MemoryVersion)
            .where(MemoryVersion.target_id == fact.id)
            .order_by(MemoryVersion.version.asc())
        )
    ).scalars().all()
    assert [version.operation for version in versions] == ["create", "cascade_forget"]
    assert versions[-1].before_snapshot["status"] == STATUS_ACTIVE
    assert versions[-1].after_snapshot["status"] == STATUS_INVALIDATED


@pytest.mark.asyncio
async def test_hard_delete_scrubs_prior_version_snapshots_and_blocks_rollback(db, session):
    sensitive_value = "ultra-sensitive-value-12345"
    fact = await ExtractionService(profile_id="p1").remember(
        session,
        RememberRequest(
            kind="fact",
            subject="service",
            predicate="credential",
            object=sensitive_value,
            statement=f"Service credential is {sensitive_value}.",
            metadata={"sensitivity": "secret", "secret_value": sensitive_value},
        ),
    )
    await ExtractionService(profile_id="p1").remember(
        session,
        RememberRequest(
            kind="fact",
            subject="service",
            predicate="credential",
            object=sensitive_value,
            statement=f"Updated service credential is {sensitive_value}.",
            confidence=0.95,
            metadata={"sensitivity": "secret", "secret_value": sensitive_value},
        ),
    )

    await DeletionService(profile_id="p1").forget(
        session,
        ForgetRequest(target_type="fact", target_id=fact.id, mode="hard", reason="erase me"),
    )

    versions = (
        await session.execute(
            select(MemoryVersion)
            .where(MemoryVersion.target_id == fact.id)
            .order_by(MemoryVersion.version.asc())
        )
    ).scalars().all()
    rendered = str([(version.before_snapshot, version.after_snapshot) for version in versions])
    assert sensitive_value not in rendered
    assert versions[-1].after_snapshot is None
    with pytest.raises(ValueError, match="hard-deleted"):
        await MemoryVersionService().rollback(
            session,
            target_type="fact",
            target_id=fact.id,
            to_version=1,
            profile_id="p1",
            actor="test",
        )


@pytest.mark.asyncio
async def test_rollback_restores_fact_source_message_mapping(db, session):
    event_id = uuid.uuid4()
    message_id = uuid.uuid4()
    session.add(
        MemoryEvent(
            id=event_id,
            event_type="message_ingested",
            source="test",
            actor="test",
            profile_id="p1",
            payload={},
        )
    )
    session.add(
        Message(
            id=message_id,
            event_id=event_id,
            profile_id="p1",
            session_id="s1",
            platform="telegram",
            role="user",
            content_raw="source message",
            content_redacted="source message",
            content_hash="hash-source-map-version-test",
        )
    )
    fact = await ExtractionService(profile_id="p1").remember(
        session,
        RememberRequest(
            kind="fact",
            subject="source-map",
            predicate="keeps",
            object="message-link",
            statement="This fact has a source mapping.",
        ),
    )
    fact_row = (await session.execute(select(Fact).where(Fact.id == fact.id))).scalar_one()
    fact_row.source_event_ids = [event_id]
    fact_row.source_message_ids = [message_id]
    session.add(FactSource(fact_id=fact.id, event_id=event_id, message_id=message_id))
    await session.flush()
    version_two = await MemoryVersionService().record(
        session,
        target_type="fact",
        target_id=fact.id,
        profile_id="p1",
        operation="source_update",
        before_snapshot=None,
        after_snapshot=snapshot_memory(fact_row),
    )

    current_source = (
        await session.execute(
            select(FactSource).where(
                FactSource.fact_id == fact.id,
                FactSource.event_id == event_id,
            )
        )
    ).scalar_one()
    current_source.message_id = None
    fact_row.source_message_ids = []
    await session.flush()

    await MemoryVersionService().rollback(
        session,
        target_type="fact",
        target_id=fact.id,
        to_version=version_two.version,
        profile_id="p1",
        actor="test",
    )

    restored_source = (
        await session.execute(
            select(FactSource).where(
                FactSource.fact_id == fact.id,
                FactSource.event_id == event_id,
            )
        )
    ).scalar_one()
    assert restored_source.event_id == event_id
    assert restored_source.message_id == message_id


@pytest.mark.asyncio
async def test_nested_sensitive_value_metadata_is_redacted_in_version_diff(db, session):
    sensitive_value = "nested-sensitive_value-value-12345"
    fact = await ExtractionService(profile_id="p1").remember(
        session,
        RememberRequest(
            kind="fact",
            subject="nested-service",
            predicate="credential",
            object="configured",
            statement="Nested service credential is configured.",
            metadata={"sensitivity": "secret", "nested": {"password": sensitive_value}},
        ),
    )
    await ExtractionService(profile_id="p1").remember(
        session,
        RememberRequest(
            kind="fact",
            subject="nested-service",
            predicate="credential",
            object="configured",
            statement="Nested service credential is configured and rotated.",
            confidence=0.95,
            metadata={
                "sensitivity": "secret",
                "nested": {"password": sensitive_value, "rotated": True},
            },
        ),
    )

    diff = await MemoryVersionService().diff(session, "fact", fact.id, 1, 2, profile_id="p1")
    assert sensitive_value not in str(diff.changes)


@pytest.mark.asyncio
async def test_version_reason_is_truncated_to_column_limit(db, session):
    fact = await ExtractionService(profile_id="p1").remember(
        session,
        RememberRequest(
            kind="fact",
            subject="reason",
            predicate="is",
            object="long",
            statement="Reason is long.",
        ),
    )
    long_reason = "r" * 2048

    row = await MemoryVersionService().record(
        session,
        target_type="fact",
        target_id=fact.id,
        profile_id="p1",
        operation="manual",
        before_snapshot=None,
        after_snapshot=None,
        reason=long_reason,
    )

    assert row.reason is not None
    assert len(row.reason) <= 1024
