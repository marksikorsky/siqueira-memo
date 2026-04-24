"""Partition automation. Plan §28.5 / §31.11.

Production uses ``pg_partman`` when available but the service also supports a
fallback worker job that creates monthly partitions for the current, next, and
month-after-next windows on the append-heavy tables:

- ``memory_events``
- ``messages``
- ``tool_events``
- ``retrieval_logs``

On SQLite the service performs no DDL and simply reports the windows it
would have created — enough to keep tests hermetic while exercising the
scheduling logic.

Plan §31.11 also requires startup readiness to fail if the current partition
is missing. :func:`missing_current_partitions` returns the list of tables that
need a partition for the current month — callers wire this into ``/readyz``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.logging import get_logger

log = get_logger(__name__)


PARTITIONED_TABLES: tuple[str, ...] = (
    "memory_events",
    "messages",
    "tool_events",
    "retrieval_logs",
)


@dataclass
class PartitionWindow:
    table: str
    partition_name: str
    start: datetime
    end: datetime

    def to_dict(self) -> dict[str, str]:
        return {
            "table": self.table,
            "partition_name": self.partition_name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


@dataclass
class PartitionReport:
    windows: list[PartitionWindow] = field(default_factory=list)
    created: list[str] = field(default_factory=list)
    existed: list[str] = field(default_factory=list)
    dialect: str = "unknown"


def _month_bounds(anchor: date) -> tuple[datetime, datetime]:
    start = datetime(anchor.year, anchor.month, 1, tzinfo=UTC)
    if anchor.month == 12:
        end = datetime(anchor.year + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(anchor.year, anchor.month + 1, 1, tzinfo=UTC)
    return start, end


def _next_month(anchor: date) -> date:
    start, end = _month_bounds(anchor)
    return end.date()


def planned_windows(
    *, anchor: date | None = None, tables: Iterable[str] = PARTITIONED_TABLES
) -> list[PartitionWindow]:
    anchor = anchor or datetime.now(UTC).date()
    month_a = anchor
    month_b = _next_month(anchor)
    month_c = _next_month(month_b)
    windows: list[PartitionWindow] = []
    for table in tables:
        for month in (month_a, month_b, month_c):
            start, end = _month_bounds(month)
            name = f"{table}_y{start.year:04d}m{start.month:02d}"
            windows.append(
                PartitionWindow(table=table, partition_name=name, start=start, end=end)
            )
    return windows


class PartitionService:
    """Ensure monthly partitions exist on Postgres.

    The service is idempotent: it only issues ``CREATE TABLE IF NOT EXISTS``
    statements. It does not convert non-partitioned tables into partitioned
    ones — operators run the initial conversion via the plan's §28.5 SQL
    scripts or via ``pg_partman``.
    """

    def __init__(self, *, tables: Iterable[str] = PARTITIONED_TABLES) -> None:
        self._tables = tuple(tables)

    async def ensure_partitions_exist(
        self,
        session: AsyncSession,
        *,
        anchor: date | None = None,
    ) -> PartitionReport:
        bind = session.bind
        dialect = bind.dialect.name if bind is not None else "unknown"
        report = PartitionReport(dialect=dialect)
        report.windows = planned_windows(anchor=anchor, tables=self._tables)

        if dialect != "postgresql":
            # Hermetic / SQLite: we keep the control-flow observable via the
            # report but issue no DDL.
            log.info(
                "partition.noop",
                extra={"dialect": dialect, "windows": len(report.windows)},
            )
            return report

        for window in report.windows:
            existed = await _partition_exists(session, window.partition_name)
            if existed:
                report.existed.append(window.partition_name)
                continue
            await session.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {window.partition_name} "
                    f"PARTITION OF {window.table} "
                    f"FOR VALUES FROM (:start) TO (:end)"
                ),
                {
                    "start": window.start.isoformat(),
                    "end": window.end.isoformat(),
                },
            )
            report.created.append(window.partition_name)

        log.info(
            "partition.ensured",
            extra={
                "dialect": dialect,
                "created": report.created,
                "existed": len(report.existed),
            },
        )
        return report

    async def missing_current_partitions(
        self, session: AsyncSession, *, anchor: date | None = None
    ) -> list[str]:
        """Return partitioned tables missing a partition for the current month."""
        bind = session.bind
        if bind is None or bind.dialect.name != "postgresql":
            return []
        anchor = anchor or datetime.now(UTC).date()
        start, _end = _month_bounds(anchor)
        missing: list[str] = []
        for table in self._tables:
            name = f"{table}_y{start.year:04d}m{start.month:02d}"
            if not await _partition_exists(session, name):
                missing.append(table)
        return missing


async def _partition_exists(session: AsyncSession, name: str) -> bool:
    row = await session.execute(
        text(
            "SELECT 1 FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE c.relname = :name"
        ),
        {"name": name},
    )
    return row.scalar_one_or_none() is not None
