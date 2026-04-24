"""FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from siqueira_memo.api.routes_admin import router as admin_router
from siqueira_memo.api.routes_admin_ui import router as admin_ui_router
from siqueira_memo.api.routes_health import router as health_router
from siqueira_memo.api.routes_ingest import router as ingest_router
from siqueira_memo.api.routes_memory import router as memory_router
from siqueira_memo.api.routes_recall import router as recall_router
from siqueira_memo.config import Settings, get_settings
from siqueira_memo.db import dispose_engines, get_engine
from siqueira_memo.logging import configure_logging
from siqueira_memo.services.prompt_registry import assert_hermes_prompt_hash_parity


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)
    # Plan §31.13: verify plugin system_prompt.md matches canonical copy.
    # Raises in production; warns in dev/test.
    canonical_hash, plugin_hash = assert_hermes_prompt_hash_parity(env=settings.env)
    app.state.hermes_prompt_hash = canonical_hash
    app.state.hermes_plugin_prompt_hash = plugin_hash
    # Warm the engine so the first request does not pay connection cost.
    get_engine(settings)
    yield
    await dispose_engines()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="Siqueira Memo",
        version="0.1.0",
        description="Hermes-native long-term memory with provenance, redaction, and hybrid retrieval.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(recall_router)
    app.include_router(memory_router)
    app.include_router(admin_router)
    app.include_router(admin_ui_router)
    return app


app = create_app()


def main() -> None:  # pragma: no cover
    """CLI entrypoint used by ``siqueira-memo`` script."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "siqueira_memo.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
