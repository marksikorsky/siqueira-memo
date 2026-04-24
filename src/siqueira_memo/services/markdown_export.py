"""Human-readable Markdown export. Plan §9.2 Task 9.3.

Produces a single Markdown document for a project or topic containing:

- active decisions with rationale;
- active facts grouped by topic;
- latest session summaries;
- conflict list with resolution hints.

Secrets are already redacted in ``content_redacted`` / ``statement`` fields;
the exporter pulls those fields only and never touches ``content_raw``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.models import Decision, Fact, MemoryConflict, SessionSummary
from siqueira_memo.models.constants import (
    CONFLICT_STATUS_OPEN,
    STATUS_ACTIVE,
)


@dataclass
class ExportFilter:
    profile_id: str
    project: str | None = None
    topic: str | None = None


async def export_markdown(
    session: AsyncSession, filt: ExportFilter
) -> str:
    sections: list[str] = []
    heading = filt.project or filt.topic or "all memory"
    sections.append(f"# Siqueira Memo — {heading}")

    decisions = await _fetch_decisions(session, filt)
    if decisions:
        sections.append("## Active decisions")
        for d in decisions:
            sections.append(f"### {d.topic}")
            sections.append(d.decision)
            if d.rationale:
                sections.append(f"*Rationale:* {d.rationale}")
            sections.append(
                f"*Status:* {d.status} · *Decided at:* {d.decided_at.isoformat()}"
            )
            sections.append("")

    facts = await _fetch_facts(session, filt)
    if facts:
        sections.append("## Active facts")
        for f in facts:
            sections.append(f"- **{f.subject} · {f.predicate}** → {f.object}")
            if f.statement:
                sections.append(f"  - {f.statement}")

    summaries = await _fetch_summaries(session, filt)
    if summaries:
        sections.append("## Recent session summaries")
        for s in summaries:
            sections.append(f"### {s.session_id} ({s.created_at.isoformat()})")
            sections.append(s.summary_short)
            if s.summary_long and s.summary_long != s.summary_short:
                sections.append("")
                sections.append(s.summary_long)
            sections.append("")

    conflicts = await _fetch_conflicts(session, filt)
    if conflicts:
        sections.append("## Open conflicts")
        for c in conflicts:
            sections.append(
                f"- `{c.conflict_type}` severity={c.severity} "
                f"left={c.left_id} right={c.right_id}"
            )
            if c.resolution_hint:
                sections.append(f"  - hint: {c.resolution_hint}")

    sections.append("")
    sections.append("*Generated from Siqueira Memo. Source-backed; no raw secrets.*")
    return "\n".join(sections).rstrip() + "\n"


async def _fetch_decisions(
    session: AsyncSession, filt: ExportFilter
) -> list[Decision]:
    stmt = (
        select(Decision)
        .where(Decision.profile_id == filt.profile_id)
        .where(Decision.status == STATUS_ACTIVE)
    )
    if filt.project:
        stmt = stmt.where(Decision.project == filt.project)
    if filt.topic:
        stmt = stmt.where(Decision.topic == filt.topic)
    stmt = stmt.order_by(Decision.decided_at.desc())
    return list((await session.execute(stmt)).scalars().all())


async def _fetch_facts(session: AsyncSession, filt: ExportFilter) -> list[Fact]:
    stmt = (
        select(Fact)
        .where(Fact.profile_id == filt.profile_id)
        .where(Fact.status == STATUS_ACTIVE)
    )
    if filt.project:
        stmt = stmt.where(Fact.project == filt.project)
    if filt.topic:
        stmt = stmt.where(Fact.topic == filt.topic)
    stmt = stmt.order_by(Fact.created_at.desc())
    return list((await session.execute(stmt)).scalars().all())


async def _fetch_summaries(
    session: AsyncSession, filt: ExportFilter
) -> list[SessionSummary]:
    stmt = (
        select(SessionSummary)
        .where(SessionSummary.profile_id == filt.profile_id)
        .order_by(SessionSummary.created_at.desc())
        .limit(10)
    )
    return list((await session.execute(stmt)).scalars().all())


async def _fetch_conflicts(
    session: AsyncSession, filt: ExportFilter
) -> list[MemoryConflict]:
    stmt = (
        select(MemoryConflict)
        .where(MemoryConflict.profile_id == filt.profile_id)
        .where(MemoryConflict.status == CONFLICT_STATUS_OPEN)
        .order_by(MemoryConflict.created_at.desc())
    )
    return list((await session.execute(stmt)).scalars().all())
