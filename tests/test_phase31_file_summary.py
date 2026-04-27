"""Phase 31: per-file summary chunk synthesis.

Pure-Python, no LLM. Verifies the summary chunk format, idempotency,
truncation, and edge cases (empty file, all-anonymous chunks, mixed kinds).
"""
from __future__ import annotations

from code_rag.indexing.summary import synthesize_file_summary
from code_rag.models import Chunk, ChunkKind


def _chunk(symbol: str | None, kind: ChunkKind, end_line: int = 10) -> Chunk:
    return Chunk(
        id=f"id-{symbol}",
        repo="repo",
        path="src/foo.py",
        language="python",
        symbol=symbol,
        kind=kind,
        start_line=1,
        end_line=end_line,
        text=f"def {symbol or 'x'}(): pass",
    )


def test_synthesize_emits_chunk_for_normal_file() -> None:
    chunks = [
        _chunk("connect_db", ChunkKind.FUNCTION),
        _chunk("query_users", ChunkKind.FUNCTION),
        _chunk("UserService", ChunkKind.CLASS),
    ]
    summary = synthesize_file_summary("repo", "src/foo.py", "python", chunks)
    assert summary is not None
    assert summary.kind == ChunkKind.FILE_SUMMARY
    assert summary.symbol is None
    # Path embedded in text so the embedder can hit it.
    assert "src/foo.py" in summary.text
    # Symbol names embedded.
    assert "connect_db" in summary.text
    assert "UserService" in summary.text
    # Kind histogram present.
    assert "2 functions" in summary.text
    assert "1 class" in summary.text


def test_synthesize_returns_none_for_empty_chunks() -> None:
    assert synthesize_file_summary("repo", "src/foo.py", "python", []) is None


def test_synthesize_returns_none_when_only_anonymous() -> None:
    """If a file's only chunks have no symbols and no kinds (shouldn't
    really happen but defensive), nothing meaningful to summarize."""
    # Note: ChunkKind.OTHER still counts as a kind, so even anonymous
    # chunks land in the histogram. The "returns None" path requires
    # an empty chunks list, which is the tested case above. This test
    # documents that ANY chunks → some summary.
    chunks = [_chunk(None, ChunkKind.OTHER)]
    summary = synthesize_file_summary("repo", "src/foo.py", "python", chunks)
    assert summary is not None
    assert "1 other" in summary.text


def test_synthesize_truncates_symbol_list() -> None:
    """Files with > 30 symbols inline only the first 30 + an `...(+N more)`
    marker so the chunk text doesn't blow past embedder context limits."""
    chunks = [
        _chunk(f"sym_{i}", ChunkKind.FUNCTION) for i in range(50)
    ]
    summary = synthesize_file_summary("repo", "src/foo.py", "python", chunks)
    assert summary is not None
    assert "+20 more" in summary.text
    # First 30 names should still be present.
    assert "sym_0" in summary.text
    assert "sym_29" in summary.text
    # 30th onwards should NOT be inlined.
    assert "sym_30" not in summary.text


def test_synthesize_id_is_content_addressed() -> None:
    """Same file with same symbol set → same id (idempotent reindex).
    Different symbols → different id (stale summary gets evicted)."""
    a1 = synthesize_file_summary("r", "p", "python",
                                 [_chunk("a", ChunkKind.FUNCTION)])
    a2 = synthesize_file_summary("r", "p", "python",
                                 [_chunk("a", ChunkKind.FUNCTION)])
    b = synthesize_file_summary("r", "p", "python",
                                [_chunk("b", ChunkKind.FUNCTION)])
    assert a1 is not None and a2 is not None and b is not None
    assert a1.id == a2.id
    assert a1.id != b.id


def test_synthesize_filters_existing_summaries() -> None:
    """Defensive: if the caller passes in a previous summary chunk by mistake,
    we don't recurse / double-count it in the kind histogram."""
    real = _chunk("connect", ChunkKind.FUNCTION)
    bogus = Chunk(
        id="bogus",
        repo="r", path="p", language="python",
        symbol=None,
        kind=ChunkKind.FILE_SUMMARY,
        start_line=1, end_line=2,
        text="[file] p\n[summary] should be ignored",
    )
    summary = synthesize_file_summary("r", "p", "python", [real, bogus])
    assert summary is not None
    assert "1 function" in summary.text
    # The fake summary chunk did NOT get folded into the kind histogram.
    assert "1 file_summary" not in summary.text


def test_synthesize_dedupes_repeated_symbols() -> None:
    """Same symbol declared twice (e.g. partial classes in C#) appears once
    in the inlined list but kind histogram still reflects both occurrences."""
    chunks = [
        _chunk("Helper", ChunkKind.CLASS),
        _chunk("Helper", ChunkKind.CLASS),
    ]
    summary = synthesize_file_summary("r", "p", "csharp", chunks)
    assert summary is not None
    # "Helper" appears in the defines line once.
    assert summary.text.count("Helper,") + summary.text.count("Helper\n") <= 1
    # But there are 2 classes in the histogram.
    assert "2 classes" in summary.text


def test_synthesize_end_line_is_max_input_end_line() -> None:
    chunks = [
        _chunk("a", ChunkKind.FUNCTION, end_line=10),
        _chunk("b", ChunkKind.FUNCTION, end_line=200),
        _chunk("c", ChunkKind.FUNCTION, end_line=50),
    ]
    summary = synthesize_file_summary("r", "p", "python", chunks)
    assert summary is not None
    assert summary.end_line == 200
