"""Boot-time Hermes prompt hash parity. Plan §31.13."""

from __future__ import annotations

from pathlib import Path

import pytest

from siqueira_memo.services.prompt_registry import (
    PromptDriftError,
    assert_hermes_prompt_hash_parity,
    default_plugin_prompt_path,
)


def test_parity_passes_in_dev_regardless(caplog, tmp_path: Path):
    drifted = tmp_path / "drifted.md"
    drifted.write_text("totally different content", encoding="utf-8")
    with caplog.at_level("WARNING"):
        canonical, plugin = assert_hermes_prompt_hash_parity(
            env="development", plugin_path=drifted
        )
    assert canonical != plugin
    assert any("prompt.hash_drift_warning" in rec.message for rec in caplog.records)


def test_parity_raises_in_production_on_mismatch(tmp_path: Path):
    drifted = tmp_path / "drifted.md"
    drifted.write_text("different", encoding="utf-8")
    with pytest.raises(PromptDriftError):
        assert_hermes_prompt_hash_parity(env="production", plugin_path=drifted)


def test_parity_ok_on_matching_plugin_copy(tmp_path: Path):
    plugin_path = default_plugin_prompt_path()
    # The repo's plugin-copy isn't identical to the canonical (it's a pointer).
    # Build a matching copy to confirm the "ok" branch.
    canonical_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "siqueira_memo"
        / "hermes_provider"
        / "system_prompt.md"
    )
    copy_target = tmp_path / "match.md"
    copy_target.write_bytes(canonical_path.read_bytes())
    canonical, plugin = assert_hermes_prompt_hash_parity(
        env="production", plugin_path=copy_target
    )
    assert canonical == plugin
    # Also confirm the real repo plugin-copy path resolves (may mismatch; that's
    # caught by the dev-warning test above, not this one).
    assert plugin_path.exists()
