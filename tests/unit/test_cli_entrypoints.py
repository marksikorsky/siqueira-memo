"""Console-script entrypoints. Plan §16."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import siqueira_memo.cli.export_markdown as export_cli
import siqueira_memo.cli.import_hermes_sessions as import_cli
import siqueira_memo.cli.rebuild_embeddings as rebuild_cli


def test_export_markdown_parser_accepts_documented_flags():
    args = export_cli._parser().parse_args(
        ["--project", "siqueira-memo", "--topic", "memory", "--output", "-"]
    )
    assert args.project == "siqueira-memo"
    assert args.output == "-"


def test_rebuild_embeddings_parser_restricts_source():
    # Valid choice.
    args = rebuild_cli._parser().parse_args(["--source", "message"])
    assert args.source == "message"
    # Invalid choice rejected by argparse.
    with pytest.raises(SystemExit):
        rebuild_cli._parser().parse_args(["--source", "bogus"])


def test_import_hermes_parser_requires_format(tmp_path: Path):
    with pytest.raises(SystemExit):
        import_cli._parser().parse_args([str(tmp_path / "x.jsonl")])
    args = import_cli._parser().parse_args([str(tmp_path / "x.jsonl"), "--format", "jsonl"])
    assert args.format == "jsonl"


def test_export_markdown_main_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Invoke the CLI's ``main`` against an ephemeral SQLite DB and capture
    the rendered markdown file. Proves the console-script entrypoint is wired
    correctly.
    """
    from siqueira_memo.config import settings_for_tests
    from siqueira_memo.db import create_all_for_tests

    settings = settings_for_tests(
        database_url="sqlite+aiosqlite:///" + str(tmp_path / "smoke.db"),
    )
    # Patch get_settings so the CLI picks up our tmp DB.
    monkeypatch.setattr("siqueira_memo.cli.export_markdown.get_settings", lambda: settings)

    async def _bootstrap():
        await create_all_for_tests(settings)

    asyncio.run(_bootstrap())

    out = tmp_path / "memo.md"
    rc = export_cli.main(["--output", str(out)])
    assert rc == 0
    body = out.read_text(encoding="utf-8")
    assert body.startswith("# Siqueira Memo")
    # No active memory yet — but the footer is still there.
    assert "no raw secrets" in body


def test_console_scripts_declared_in_pyproject():
    """Keep pyproject.toml entrypoints aligned with this CLI package."""
    text = Path(__file__).resolve().parents[2].joinpath("pyproject.toml").read_text("utf-8")
    for name in (
        "siqueira-memo =",
        "siqueira-memo-worker =",
        "siqueira-memo-evals =",
        "siqueira-memo-import-hermes =",
        "siqueira-memo-rebuild-embeddings =",
        "siqueira-memo-export-markdown =",
    ):
        assert name in text, f"missing console script: {name}"
