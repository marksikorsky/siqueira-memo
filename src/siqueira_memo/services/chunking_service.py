"""Dialogue/tool-aware chunking. Plan §20."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from siqueira_memo.utils.tokens import Tokenizer, get_tokenizer


@dataclass
class ChunkInput:
    source_id: str
    text: str
    created_at: datetime | None
    role: str | None = None
    project: str | None = None
    topic: str | None = None
    entities: list[str] = field(default_factory=list)
    sensitivity: str = "normal"


@dataclass
class ChunkOutput:
    chunk_text: str
    chunk_index: int
    token_count: int
    source_ids: list[str]
    tokenizer_name: str
    tokenizer_version: str | None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueWindow:
    chunk_text: str
    token_count: int
    source_ids: list[str]
    chunk_index: int
    extra_metadata: dict[str, Any] = field(default_factory=dict)


def _re_split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


class ChunkingService:
    """Dialogue/tool/code/log-aware chunker.

    The tokenizer abstraction makes chunks portable across embedding models and
    records the tokenizer name/version on each chunk (plan §20.1).
    """

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        *,
        short_max_tokens: int = 350,
        medium_target_tokens: int = 500,
        medium_overlap_tokens: int = 80,
        long_min_tokens: int = 1200,
        long_target_tokens: int = 800,
        long_overlap_tokens: int = 120,
        dialogue_window_max_tokens: int = 700,
        dialogue_window_max_messages: int = 8,
    ) -> None:
        self._tok = tokenizer or get_tokenizer()
        self.short_max_tokens = short_max_tokens
        self.medium_target_tokens = medium_target_tokens
        self.medium_overlap_tokens = medium_overlap_tokens
        self.long_min_tokens = long_min_tokens
        self.long_target_tokens = long_target_tokens
        self.long_overlap_tokens = long_overlap_tokens
        self.dialogue_window_max_tokens = dialogue_window_max_tokens
        self.dialogue_window_max_messages = dialogue_window_max_messages

    # ------------------------------------------------------------------
    # Dialogue message chunking
    # ------------------------------------------------------------------
    def chunk_message(
        self,
        text: str,
        *,
        source_id: str,
        sensitivity: str = "normal",
        already_redacted: bool = True,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[ChunkOutput]:
        if sensitivity == "sensitive" and not already_redacted:
            raise ValueError("refusing to chunk unredacted sensitive text")
        if not text or not text.strip():
            return []
        count = self._tok.count(text)
        meta = dict(extra_metadata or {})
        if count <= self.short_max_tokens:
            return [
                ChunkOutput(
                    chunk_text=text,
                    chunk_index=0,
                    token_count=count,
                    source_ids=[source_id],
                    tokenizer_name=self._tok.name,
                    tokenizer_version=self._tok.version,
                    extra_metadata={**meta, "strategy": "short"},
                )
            ]
        if count <= self.long_min_tokens:
            return self._chunk_by_words(
                text,
                source_id=source_id,
                target=self.medium_target_tokens,
                overlap=self.medium_overlap_tokens,
                strategy="medium",
                extra_metadata=meta,
            )
        return self._chunk_by_words(
            text,
            source_id=source_id,
            target=self.long_target_tokens,
            overlap=self.long_overlap_tokens,
            strategy="long",
            extra_metadata=meta,
        )

    def _chunk_by_words(
        self,
        text: str,
        *,
        source_id: str,
        target: int,
        overlap: int,
        strategy: str,
        extra_metadata: dict[str, Any],
    ) -> list[ChunkOutput]:
        # We use whitespace-tokenisation as the chunking unit. Production
        # tokenizers may differ but the chunk boundaries remain stable.
        tokens = text.split()
        if not tokens:
            return []
        chunks: list[ChunkOutput] = []
        step = max(1, target - overlap)
        index = 0
        i = 0
        while i < len(tokens):
            slice_end = min(len(tokens), i + target)
            window = tokens[i:slice_end]
            window_text = " ".join(window)
            chunks.append(
                ChunkOutput(
                    chunk_text=window_text,
                    chunk_index=index,
                    token_count=self._tok.count(window_text),
                    source_ids=[source_id],
                    tokenizer_name=self._tok.name,
                    tokenizer_version=self._tok.version,
                    extra_metadata={**extra_metadata, "strategy": strategy},
                )
            )
            index += 1
            if slice_end >= len(tokens):
                break
            i += step
        return chunks

    # ------------------------------------------------------------------
    # Dialogue window chunking (adjacent short messages)
    # ------------------------------------------------------------------
    def chunk_dialogue_window(
        self, messages: Iterable[ChunkInput]
    ) -> list[DialogueWindow]:
        windows: list[DialogueWindow] = []
        buf: list[ChunkInput] = []
        buf_tokens = 0
        index = 0
        for msg in messages:
            msg_tokens = self._tok.count(msg.text)
            new_total = buf_tokens + msg_tokens + (1 if buf else 0)
            if (
                buf
                and (
                    new_total > self.dialogue_window_max_tokens
                    or len(buf) >= self.dialogue_window_max_messages
                )
            ):
                windows.append(self._flush_window(buf, index))
                index += 1
                buf = []
                buf_tokens = 0
            buf.append(msg)
            buf_tokens += msg_tokens
        if buf:
            windows.append(self._flush_window(buf, index))
        return windows

    def _flush_window(self, buf: list[ChunkInput], index: int) -> DialogueWindow:
        chunk_text = "\n".join(
            f"{msg.role or 'user'}: {msg.text}" for msg in buf
        )
        return DialogueWindow(
            chunk_text=chunk_text,
            token_count=self._tok.count(chunk_text),
            source_ids=[msg.source_id for msg in buf],
            chunk_index=index,
            extra_metadata={"roles": [msg.role for msg in buf]},
        )

    # ------------------------------------------------------------------
    # Tool output chunking
    # ------------------------------------------------------------------
    def chunk_json(
        self, payload: Any, *, source_id: str, path_prefix: str = "$"
    ) -> list[ChunkOutput]:
        chunks: list[ChunkOutput] = []
        if isinstance(payload, dict):
            for i, (key, value) in enumerate(payload.items()):
                jsonpath = f"{path_prefix}.{key}"
                text = json.dumps({key: value}, ensure_ascii=False, indent=2, default=str)
                chunks.append(
                    ChunkOutput(
                        chunk_text=text,
                        chunk_index=i,
                        token_count=self._tok.count(text),
                        source_ids=[source_id],
                        tokenizer_name=self._tok.name,
                        tokenizer_version=self._tok.version,
                        extra_metadata={"jsonpath": jsonpath, "strategy": "json_keys"},
                    )
                )
        elif isinstance(payload, list):
            for i, item in enumerate(payload):
                jsonpath = f"{path_prefix}[{i}]"
                text = json.dumps(item, ensure_ascii=False, indent=2, default=str)
                chunks.append(
                    ChunkOutput(
                        chunk_text=text,
                        chunk_index=i,
                        token_count=self._tok.count(text),
                        source_ids=[source_id],
                        tokenizer_name=self._tok.name,
                        tokenizer_version=self._tok.version,
                        extra_metadata={"jsonpath": jsonpath, "strategy": "json_items"},
                    )
                )
        else:
            text = json.dumps(payload, ensure_ascii=False, default=str)
            chunks.append(
                ChunkOutput(
                    chunk_text=text,
                    chunk_index=0,
                    token_count=self._tok.count(text),
                    source_ids=[source_id],
                    tokenizer_name=self._tok.name,
                    tokenizer_version=self._tok.version,
                    extra_metadata={"jsonpath": path_prefix, "strategy": "json_value"},
                )
            )
        return chunks

    def chunk_log(self, log_text: str, *, source_id: str) -> list[ChunkOutput]:
        """Chunk logs by grouping error blocks plus surrounding context lines.

        ``ERROR``/``FATAL``/``Traceback`` markers start new chunks; consecutive
        non-error lines are merged into preceding info chunks.
        """
        lines = log_text.splitlines()
        if not lines:
            return []
        chunks: list[ChunkOutput] = []
        buf: list[str] = []
        error_mode = False
        index = 0

        def flush(kind: str) -> None:
            nonlocal buf, index
            if not buf:
                return
            text = "\n".join(buf)
            chunks.append(
                ChunkOutput(
                    chunk_text=text,
                    chunk_index=index,
                    token_count=self._tok.count(text),
                    source_ids=[source_id],
                    tokenizer_name=self._tok.name,
                    tokenizer_version=self._tok.version,
                    extra_metadata={"strategy": "log", "kind": kind},
                )
            )
            index += 1
            buf = []

        for line in lines:
            is_error = bool(re.match(r"^\s*(ERROR|FATAL|Traceback)", line))
            if is_error and not error_mode:
                flush("info")
                error_mode = True
            elif not is_error and error_mode:
                flush("error")
                error_mode = False
            buf.append(line)
        flush("error" if error_mode else "info")
        return chunks


_default = ChunkingService()


def chunk_text(text: str, *, source_id: str = "default") -> list[ChunkOutput]:
    return _default.chunk_message(text, source_id=source_id)
