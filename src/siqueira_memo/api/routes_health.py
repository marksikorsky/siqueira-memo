"""Health + readiness endpoints. Plan §4.1."""

from __future__ import annotations

from fastapi import APIRouter, Request, status
from sqlalchemy import text

from siqueira_memo.api.deps import SessionDep
from siqueira_memo.config import Settings
from siqueira_memo.schemas.admin import HealthStatus
from siqueira_memo.services.partition_service import PartitionService

router = APIRouter()


def _app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


@router.get("/healthz", response_model=HealthStatus)
async def healthz(request: Request) -> HealthStatus:
    settings = _app_settings(request)
    """Liveness — no external dependencies are touched."""
    return HealthStatus(
        ok=True,
        env=settings.env,
        database={"configured": settings.database_url.split("://", 1)[0]},
        queue={"backend": settings.queue_backend},
        providers=["siqueira-memo"],
    )


@router.get("/readyz", response_model=HealthStatus, status_code=status.HTTP_200_OK)
async def readyz(session: SessionDep, request: Request) -> HealthStatus:
    settings = _app_settings(request)
    """Readiness — touches DB, reports pgvector and Redis status best-effort."""
    db_ok = False
    pgvector_ok = False
    migration_version: str | None = None
    try:
        ping = await session.execute(text("SELECT 1"))
        db_ok = ping.scalar_one() == 1
        if settings.is_postgres:
            try:
                result = await session.execute(
                    text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                )
                pgvector_ok = result.scalar_one_or_none() is not None
            except Exception:
                pgvector_ok = False
            try:
                mv = await session.execute(text("SELECT version_num FROM alembic_version"))
                migration_version = mv.scalar_one_or_none()
            except Exception:
                migration_version = None
        else:
            pgvector_ok = False
            migration_version = "sqlite-adhoc"
    except Exception:
        db_ok = False

    queue_backend = settings.queue_backend
    queue_status = {"backend": queue_backend, "configured": True}

    # Plan §31.11: surface missing current-month partitions so operators can
    # see at a glance whether the daily worker is running. Postgres only; on
    # SQLite we return an empty list and ``required=False``.
    # The one-command Docker quickstart runs Postgres without converting append
    # tables to native partitions; that is fine for development. Production can
    # keep the stricter plan invariant by setting SIQUEIRA_ENV=production, where
    # missing current-month partitions make readiness fail.
    partitions_required = settings.is_postgres and settings.env == "production"
    missing_partitions: list[str] = []
    if partitions_required:
        try:
            missing_partitions = await PartitionService().missing_current_partitions(session)
        except Exception:
            missing_partitions = []

    partitions = {
        "required": partitions_required,
        "missing_current": missing_partitions,
    }

    # Plan §31.13: expose boot-time prompt hash parity. Populated in the
    # lifespan; default to empty when the helper hasn't run yet.
    prompt_parity = {
        "canonical_hash": getattr(request.app.state, "hermes_prompt_hash", None),
        "plugin_hash": getattr(request.app.state, "hermes_plugin_prompt_hash", None),
    }
    prompt_parity["ok"] = (
        prompt_parity["canonical_hash"] is not None
        and prompt_parity["canonical_hash"] == prompt_parity["plugin_hash"]
    )

    overall_ok = db_ok and not missing_partitions and (prompt_parity["ok"] or settings.env != "production")

    return HealthStatus(
        ok=overall_ok,
        env=settings.env,
        database={"url_scheme": settings.database_url.split("://", 1)[0], "reachable": db_ok},
        queue=queue_status,
        pgvector={"present": pgvector_ok, "required": settings.is_postgres},
        migration_version=migration_version,
        providers=["siqueira-memo"],
        partitions=partitions,
        prompt_parity=prompt_parity,
    )
