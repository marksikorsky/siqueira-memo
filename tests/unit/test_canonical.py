"""Tests for canonical normalization and key derivation."""

from __future__ import annotations

from siqueira_memo.utils.canonical import (
    advisory_lock_key,
    content_hash,
    decision_canonical_key,
    fact_canonical_key,
    normalize_text,
)


def test_normalize_text_basic():
    assert normalize_text("  Hello World!  ") == "hello world"


def test_normalize_text_unicode_casefold():
    assert normalize_text("STRAßE") == normalize_text("strasse")


def test_normalize_text_strips_markdown():
    assert normalize_text("**Key** `decision`") == "key decision"


def test_normalize_text_handles_unicode_dashes():
    # em-dash becomes ascii dash.
    assert normalize_text("foo—bar") == "foo-bar"


def test_fact_canonical_key_is_deterministic():
    a = fact_canonical_key("siqueira-memo", "primary_integration", "MemoryProvider plugin")
    b = fact_canonical_key("siqueira-memo", "primary_integration", "MemoryProvider plugin")
    assert a == b


def test_fact_canonical_key_normalization():
    a = fact_canonical_key("Siqueira-Memo ", "PRIMARY_integration", "memoryprovider  plugin")
    b = fact_canonical_key("siqueira-memo", "primary_integration", "MemoryProvider plugin")
    assert a == b


def test_fact_canonical_key_profile_isolated():
    a = fact_canonical_key("x", "y", "z", profile_id="p1")
    b = fact_canonical_key("x", "y", "z", profile_id="p2")
    assert a != b


def test_decision_canonical_key_ignores_wording_changes():
    a = decision_canonical_key("siqueira-memo", "integration", "use MemoryProvider plugin")
    b = decision_canonical_key("Siqueira-Memo", "Integration", "Use MemoryProvider Plugin")
    assert a == b


def test_content_hash_stable():
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")


def test_advisory_lock_key_in_bigint_range():
    k = advisory_lock_key("canonical")
    assert -(2**63) <= k < 2**63
