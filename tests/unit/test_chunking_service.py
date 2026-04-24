"""Chunking tests. Plan §20."""

from __future__ import annotations

from siqueira_memo.services.chunking_service import ChunkingService, ChunkInput, chunk_text


def test_short_message_single_chunk():
    svc = ChunkingService()
    chunks = svc.chunk_message("Hello world.", source_id="1")
    assert len(chunks) == 1
    assert chunks[0].chunk_text == "Hello world."
    assert chunks[0].chunk_index == 0


def test_medium_message_produces_overlap():
    svc = ChunkingService(
        short_max_tokens=5, medium_target_tokens=6, medium_overlap_tokens=2
    )
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    chunks = svc.chunk_message(text, source_id="2")
    assert len(chunks) >= 2
    # Chunks must carry ordered indices.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Overlap: token sets of consecutive chunks must share at least one word.
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert set(a.chunk_text.split()) & set(b.chunk_text.split())


def test_long_message_sliding_window():
    svc = ChunkingService(
        short_max_tokens=10,
        medium_target_tokens=25,
        long_min_tokens=30,
        long_target_tokens=25,
        long_overlap_tokens=5,
    )
    words = [f"tok{i}" for i in range(150)]
    text = " ".join(words)
    chunks = svc.chunk_message(text, source_id="3")
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count > 0


def test_combine_adjacent_short_messages():
    svc = ChunkingService(dialogue_window_max_tokens=30, dialogue_window_max_messages=3)
    inputs = [
        ChunkInput(source_id=str(i), text=f"msg {i} short", created_at=None, role="user")
        for i in range(6)
    ]
    windows = svc.chunk_dialogue_window(inputs)
    assert len(windows) >= 2
    # First window merges several source ids.
    assert len(windows[0].source_ids) > 1
    # Each window respects the max-messages budget.
    for w in windows:
        assert len(w.source_ids) <= 3


def test_json_tool_output_chunks_by_top_level_keys():
    svc = ChunkingService()
    payload = {
        "status": "ok",
        "results": [{"id": 1, "value": "alpha"}, {"id": 2, "value": "beta"}],
        "pagination": {"next": None},
    }
    chunks = svc.chunk_json(payload, source_id="json1")
    assert len(chunks) == 3
    assert any("results" in c.extra_metadata.get("jsonpath", "") for c in chunks)


def test_tool_log_output_chunks_by_error_blocks():
    svc = ChunkingService()
    log = (
        "INFO starting\n"
        "INFO step 1\n"
        "ERROR something broke\nERROR more\nERROR details\n"
        "INFO cleanup\n"
    )
    chunks = svc.chunk_log(log, source_id="log1")
    assert len(chunks) >= 2
    # First chunk should capture the error context.
    error_chunk = next(c for c in chunks if "ERROR" in c.chunk_text)
    assert "something broke" in error_chunk.chunk_text


def test_never_chunks_sensitive_unredacted_text():
    svc = ChunkingService()
    try:
        svc.chunk_message(
            "sk-proj-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            source_id="x",
            sensitivity="sensitive",
            already_redacted=False,
        )
    except ValueError as exc:
        assert "redacted" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for unredacted sensitive text")


def test_convenience_chunk_text_function():
    chunks = chunk_text("hello world")
    assert len(chunks) == 1
    assert chunks[0].chunk_text == "hello world"
