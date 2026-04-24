"""Partition automation. Plan §28.5 / §31.11."""

from __future__ import annotations

from datetime import date

import pytest

from siqueira_memo.services.partition_service import (
    PARTITIONED_TABLES,
    PartitionService,
    planned_windows,
)


def test_planned_windows_covers_three_months():
    windows = planned_windows(anchor=date(2026, 1, 15))
    tables = {w.table for w in windows}
    assert tables == set(PARTITIONED_TABLES)
    months = sorted({(w.start.year, w.start.month) for w in windows})
    assert months == [(2026, 1), (2026, 2), (2026, 3)]


def test_planned_windows_handles_december_rollover():
    windows = planned_windows(anchor=date(2026, 12, 5))
    months = sorted({(w.start.year, w.start.month) for w in windows})
    assert months == [(2026, 12), (2027, 1), (2027, 2)]


def test_partition_name_format():
    windows = planned_windows(anchor=date(2026, 7, 1))
    memory = [w for w in windows if w.table == "memory_events"]
    assert {w.partition_name for w in memory} == {
        "memory_events_y2026m07",
        "memory_events_y2026m08",
        "memory_events_y2026m09",
    }


@pytest.mark.asyncio
async def test_ensure_partitions_noop_on_sqlite(session):
    svc = PartitionService()
    report = await svc.ensure_partitions_exist(session, anchor=date(2026, 4, 24))
    assert report.dialect == "sqlite"
    assert report.created == []
    assert report.existed == []
    assert len(report.windows) == 3 * len(PARTITIONED_TABLES)


@pytest.mark.asyncio
async def test_missing_current_partitions_empty_on_sqlite(session):
    svc = PartitionService()
    assert await svc.missing_current_partitions(session, anchor=date(2026, 4, 24)) == []
