"""Tokenization helpers.

Uses ``tiktoken`` when OpenAI-compatible models are in play, falls back to a
simple whitespace splitter so non-Postgres/offline environments still boot.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class Tokenizer(Protocol):
    name: str
    version: str | None

    def count(self, text: str) -> int:
        ...

    def encode(self, text: str) -> list[int]:
        ...


class _WhitespaceTokenizer:
    name: str = "whitespace"
    version: str | None = "1"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    def encode(self, text: str) -> list[int]:
        return [hash(tok) & 0xFFFF for tok in (text or "").split()]


class _TiktokenTokenizer:
    def __init__(self, encoding_name: str) -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding_name)
        self.name: str = f"tiktoken:{encoding_name}"
        self.version: str | None = "1"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))

    def encode(self, text: str) -> list[int]:
        result: list[int] = self._enc.encode(text or "")
        return result


@lru_cache(maxsize=8)
def get_tokenizer(name: str = "cl100k_base") -> Tokenizer:
    """Return a tokenizer by name. Falls back to whitespace if tiktoken lacks the encoding."""
    if name == "whitespace":
        return _WhitespaceTokenizer()
    try:
        return _TiktokenTokenizer(name)
    except Exception:  # pragma: no cover
        return _WhitespaceTokenizer()
