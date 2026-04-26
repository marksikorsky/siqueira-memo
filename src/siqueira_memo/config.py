"""Runtime configuration for Siqueira Memo.

Settings are loaded from environment variables prefixed with ``SIQUEIRA_``. See
``.env.example`` for the full list. Tests typically override values via
:func:`get_settings` or by instantiating :class:`Settings` directly.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["openai", "local", "mock"]
RerankerProvider = Literal["local", "cohere", "mock"]
QueueBackend = Literal["redis", "memory"]


class Settings(BaseSettings):
    """Application settings.

    Values are immutable per-process. Tests that need different values should
    construct a fresh :class:`Settings` instance and inject it rather than
    mutating the global cache.
    """

    model_config = SettingsConfigDict(
        env_prefix="SIQUEIRA_",
        env_file=(".env", ".env.local"),
        extra="ignore",
        case_sensitive=False,
    )

    env: Literal["development", "test", "staging", "production"] = "development"

    host: str = "127.0.0.1"
    port: int = 8787
    api_token: SecretStr = Field(default=SecretStr("local-dev-token"))
    admin_password: SecretStr | None = None
    admin_session_secret: SecretStr | None = None
    admin_session_ttl_seconds: int = 60 * 60 * 12
    admin_cookie_secure: bool = False
    memory_capture_mode: Literal["off", "conservative", "balanced", "aggressive"] = "aggressive"
    memory_capture_target_ratio: float = 0.8
    memory_capture_save_raw_turns: bool = True
    memory_capture_extract_structured: bool = True
    memory_capture_llm_enabled: bool = False
    memory_capture_llm_base_url: str = ""
    memory_capture_llm_api_key: SecretStr = Field(default=SecretStr(""))
    memory_capture_llm_model: str = "gpt-5"
    memory_capture_llm_timeout_seconds: float = 30.0

    database_url: str = "sqlite+aiosqlite:///./siqueira_memo.db"
    database_echo: bool = False

    redis_url: str = "redis://127.0.0.1:6379/0"
    queue_backend: QueueBackend = "memory"

    embedding_provider: EmbeddingProvider = "mock"
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 3072
    openai_api_key: SecretStr = Field(default=SecretStr(""))

    reranker_provider: RerankerProvider = "mock"

    retention_retrieval_logs_days: int = 180
    retention_worker_logs_days: int = 30

    log_level: str = "INFO"
    otel_enabled: bool = False

    # Identity / profile defaults used when Hermes has not supplied them.
    default_profile_id: str = "default"
    hermes_home: str | None = None
    agent_identity: str | None = None

    # Prefetch budgets (tokens).
    prefetch_fast_budget_tokens: int = 1200
    prefetch_balanced_budget_tokens: int = 2000
    prefetch_max_source_snippets: int = 3

    # Deletion/regeneration thresholds. See plan §24.1 / §31.12.
    summary_stale_threshold: float = 0.10
    summary_invalid_threshold: float = 0.60

    @field_validator("database_url")
    @classmethod
    def _normalize_database_url(cls, value: str) -> str:
        # Accept postgres:// shorthand.
        if value.startswith("postgres://"):
            return "postgresql+asyncpg://" + value[len("postgres://") :]
        if value.startswith("postgresql://") and "+asyncpg" not in value:
            return "postgresql+asyncpg://" + value[len("postgresql://") :]
        return value

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")

    def derive_profile_id(self) -> str:
        """Derive the durable profile ID per plan §33.6."""
        if self.agent_identity:
            return self.agent_identity
        if self.hermes_home:
            normalized = str(Path(self.hermes_home).expanduser().resolve())
            return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
        return self.default_profile_id


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings. Tests use this after mutating the env."""
    get_settings.cache_clear()


def settings_for_tests(**overrides: object) -> Settings:
    """Produce a fresh :class:`Settings` suitable for unit tests.

    The default test configuration is hermetic: in-memory SQLite, mock providers,
    no external network access.
    """
    defaults: dict[str, object] = {
        "env": "test",
        "database_url": "sqlite+aiosqlite:///:memory:",
        "queue_backend": "memory",
        "embedding_provider": "mock",
        "reranker_provider": "mock",
        "api_token": SecretStr("test-token"),
        "log_level": "WARNING",
        "memory_capture_llm_enabled": False,
        "memory_capture_llm_base_url": "",
        "memory_capture_llm_api_key": SecretStr(""),
    }
    defaults.update(overrides)
    # Clear possible stale env so overrides win.
    env_snapshot = {k: v for k, v in os.environ.items() if k.startswith("SIQUEIRA_")}
    for k in env_snapshot:
        os.environ.pop(k, None)
    try:
        return Settings(**defaults)  # type: ignore[arg-type]
    finally:
        os.environ.update(env_snapshot)
