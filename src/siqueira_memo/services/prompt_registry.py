"""Prompt registry. Plan §22.1 / §22.2 / §31.13.

Loads versioned prompt artifacts under ``src/siqueira_memo/prompts`` and
registers them in the ``prompt_versions`` table. The registry exposes:

* ``scan()`` — enumerate prompt files on disk.
* ``sync(session)`` — upsert ``PromptVersion`` rows and verify content hashes.
* ``get(name, version)`` — load a prompt's content with deterministic caching.

The Hermes prompt artifact lives outside ``src/siqueira_memo/prompts`` (inside
``plugins/memory/siqueira-memo/system_prompt.md``). The registry has a
dedicated helper for that path so plan §31.13's boot-time hash check is a one
function call.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.logging import get_logger
from siqueira_memo.models import PromptVersion

log = get_logger(__name__)


_NAME_VERSION_RE = re.compile(r"^(?P<name>[\w_]+)\.(?P<version>v\d+[\w.]*)\.md$")


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts"


@dataclass(frozen=True)
class PromptEntry:
    name: str
    version: str
    path: Path
    content: str
    content_hash: str

    @classmethod
    def from_path(cls, path: Path) -> PromptEntry | None:
        match = _NAME_VERSION_RE.match(path.name)
        if match is None:
            return None
        content = path.read_text(encoding="utf-8")
        return cls(
            name=match.group("name"),
            version=match.group("version"),
            path=path,
            content=content,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )


class PromptRegistry:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or _prompt_dir()
        self._cache: dict[tuple[str, str], PromptEntry] = {}

    def scan(self) -> list[PromptEntry]:
        entries: list[PromptEntry] = []
        if not self.base_dir.exists():
            return entries
        for path in sorted(self.base_dir.glob("*.md")):
            entry = PromptEntry.from_path(path)
            if entry is None:
                continue
            self._cache[(entry.name, entry.version)] = entry
            entries.append(entry)
        return entries

    def get(self, name: str, version: str) -> PromptEntry:
        key = (name, version)
        if key in self._cache:
            return self._cache[key]
        self.scan()
        if key not in self._cache:
            raise KeyError(f"prompt {name}@{version} not registered")
        return self._cache[key]

    async def sync(self, session: AsyncSession) -> list[PromptVersion]:
        entries = self.scan()
        persisted: list[PromptVersion] = []
        for entry in entries:
            row = (
                await session.execute(
                    select(PromptVersion).where(
                        PromptVersion.name == entry.name,
                        PromptVersion.version == entry.version,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                row = PromptVersion(
                    name=entry.name,
                    version=entry.version,
                    content_hash=entry.content_hash,
                    path=str(entry.path),
                    body=entry.content,
                    extra_metadata={"bytes": len(entry.content)},
                )
                session.add(row)
            else:
                # Hash drift is allowed in dev but flagged; production callers
                # should refuse to boot on mismatch (plan §31.13).
                if row.content_hash != entry.content_hash:
                    log.warning(
                        "prompt.hash_drift",
                        extra={
                            "prompt_name": entry.name,
                            "prompt_version": entry.version,
                        },
                    )
                row.content_hash = entry.content_hash
                row.path = str(entry.path)
                row.body = entry.content
            persisted.append(row)
        await session.flush()
        log.info("prompt.registry.synced", extra={"count": len(persisted)})
        return persisted


def register_hermes_system_prompt(
    registry: PromptRegistry, plugin_prompt_path: Path
) -> tuple[str, str]:
    """Compute the hash of the Hermes-facing system prompt.

    Returns ``(canonical_hash, plugin_copy_hash)``. Plan §31.13 requires the
    plugin copy to match the canonical copy; callers compare these values and
    fail startup in production when they diverge.
    """
    canonical_path = _prompt_dir().parent / "hermes_provider" / "system_prompt.md"
    canonical_hash = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    plugin_hash = hashlib.sha256(plugin_prompt_path.read_bytes()).hexdigest()
    return canonical_hash, plugin_hash


class PromptDriftError(RuntimeError):
    """Raised in production when the plugin-copy system prompt has drifted."""


def default_plugin_prompt_path() -> Path:
    """Resolve the plugin-copy path under ``plugins/memory/siqueira-memo``.

    Editable/dev installs keep the plugin at repo root. Container/package installs
    may run from ``/app`` while the Python package lives in site-packages, so we
    probe both locations before falling back to the historical repo-relative
    path.
    """
    env_path = os.environ.get("SIQUEIRA_HERMES_PLUGIN_PROMPT_PATH")
    if env_path:
        return Path(env_path).expanduser()

    candidates = [
        Path.cwd() / "plugins" / "memory" / "siqueira-memo" / "system_prompt.md",
        Path(__file__).resolve().parents[3]
        / "plugins"
        / "memory"
        / "siqueira-memo"
        / "system_prompt.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def assert_hermes_prompt_hash_parity(
    *, env: str, plugin_path: Path | None = None
) -> tuple[str, str]:
    """Plan §31.13 boot-time check.

    In ``env == "production"`` a mismatch raises :class:`PromptDriftError`.
    In other envs the mismatch is logged as a warning and the boot continues.
    Returns ``(canonical_hash, plugin_hash)`` for observability.
    """
    plugin_path = plugin_path or default_plugin_prompt_path()
    registry = PromptRegistry()
    canonical_hash, plugin_hash = register_hermes_system_prompt(
        registry, plugin_path
    )
    if canonical_hash != plugin_hash:
        extra = {
            "canonical_hash": canonical_hash,
            "plugin_hash": plugin_hash,
            "plugin_path": str(plugin_path),
        }
        if env == "production":
            raise PromptDriftError(
                "Hermes system prompt copy under plugins/memory/siqueira-memo "
                "drifted from the canonical prompt under src/siqueira_memo/hermes_provider"
            )
        log.warning("prompt.hash_drift_warning", extra=extra)
    return canonical_hash, plugin_hash
