"""Hermes MemoryProvider plugin entrypoint for Siqueira Memo.

Hermes loads external memory providers via ``memory.provider`` and calls
``register(ctx)`` from the plugin package. We forward to the provider class
that lives inside the service package so the plugin stays a thin shim.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# The plugin lives outside src/ so we make sure the service package is
# importable even when Hermes loads this plugin from a clean ``PYTHONPATH``.
_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[3] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from siqueira_memo.hermes_provider.provider import SiqueiraMemoProvider  # noqa: E402


def register(ctx: Any) -> None:
    """Register the memory provider. Called by Hermes at plugin load time."""
    provider = SiqueiraMemoProvider()
    try:
        ctx.register_memory_provider(provider)
    except AttributeError:
        # Older Hermes builds may expose a different API — fall back to the
        # direct registration attribute if available. Plugin authors should
        # not silently swallow this, so log to stderr when it happens.
        registry = getattr(ctx, "memory_providers", None)
        if isinstance(registry, list):
            registry.append(provider)
        else:
            print(
                "siqueira-memo: ctx has no register_memory_provider(); skipping registration",
                file=sys.stderr,
            )


__all__ = ["SiqueiraMemoProvider", "register"]
