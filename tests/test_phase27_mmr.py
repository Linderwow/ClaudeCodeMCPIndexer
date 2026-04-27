"""Phase 27: MMR diversity at merge.

Pre-Phase-27, fused hits frequently came back with the same file occupying
multiple top slots (e.g. 5x MNQAlpha.cs for "OnBarUpdate"). MMR penalizes
near-duplicates so users see a more diverse slate.
"""
from __future__ import annotations

from code_rag.models import Chunk, ChunkKind, SearchHit
from code_rag.retrieval.fusion import mmr_diversify


def _hit(score: float, path: str, symbol: str | None = None, idx: int = 0) -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id=f"{path}:{symbol}:{idx}",
            repo="r",
            path=path,
            language="python",
            symbol=symbol,
            kind=ChunkKind.FUNCTION,
            start_line=idx + 1,
            end_line=idx + 5,
            text="...",
        ),
        score=score,
        source="hybrid",
    )


def test_mmr_breaks_single_file_dominance() -> None:
    """5 hits all from the same file should NOT all win the top 5 when a
    decent hit from a second file exists — that's the point of MMR.
    """
    hits = [
        _hit(0.99, "MNQAlpha.cs", "f1", 0),
        _hit(0.95, "MNQAlpha.cs", "f2", 1),
        _hit(0.90, "MNQAlpha.cs", "f3", 2),
        _hit(0.85, "MNQAlpha.cs", "f4", 3),
        _hit(0.80, "MNQAlpha.cs", "f5", 4),
        _hit(0.50, "Other.cs",    "g1", 5),
    ]
    out = mmr_diversify(hits, lam=0.7, top_k=5)
    paths = [h.chunk.path for h in out]
    # Top result should still be the highest-relevance hit.
    assert paths[0] == "MNQAlpha.cs"
    # But "Other.cs" should appear in the top 5, displacing one of the
    # repeated MNQAlpha hits.
    assert "Other.cs" in paths


def test_mmr_lambda_one_is_passthrough() -> None:
    """lam=1.0 disables diversity — output equals input order, truncated."""
    hits = [
        _hit(0.99, "a.py"), _hit(0.98, "a.py"),
        _hit(0.97, "a.py"), _hit(0.96, "a.py"),
    ]
    out = mmr_diversify(hits, lam=1.0, top_k=4)
    assert [h.chunk.path for h in out] == ["a.py"] * 4
    # Order preserved exactly.
    assert [h.score for h in out] == [0.99, 0.98, 0.97, 0.96]


def test_mmr_handles_empty_input() -> None:
    assert mmr_diversify([], lam=0.7, top_k=5) == []


def test_mmr_handles_top_k_zero() -> None:
    hits = [_hit(0.5, "a.py")]
    assert mmr_diversify(hits, lam=0.7, top_k=0) == []


def test_mmr_truncates_to_top_k() -> None:
    hits = [_hit(0.9 - i * 0.01, f"f{i}.py") for i in range(10)]
    out = mmr_diversify(hits, lam=0.7, top_k=3)
    assert len(out) == 3


def test_mmr_same_path_different_symbol_less_penalized_than_same_pair() -> None:
    """A second hit from the same file but a different symbol should rank
    higher than a second hit from the same (file, symbol) pair, because the
    pair_count penalty is double-weighted."""
    hits = [
        _hit(0.99, "a.py", "foo", 0),  # selected first
        # All three candidates have identical relevance; tiebreaker is the
        # MMR penalty.
        _hit(0.50, "a.py", "foo", 1),  # same pair — penalty 1 + 2 = 3
        _hit(0.50, "a.py", "bar", 2),  # same path — penalty 1
        _hit(0.50, "b.py", "baz", 3),  # fresh file — penalty 0
    ]
    out = mmr_diversify(hits, lam=0.5, top_k=4)
    # First pick is the high-relevance hit.
    assert out[0].chunk.path == "a.py" and out[0].chunk.symbol == "foo"
    # Second pick is the fresh file (zero penalty wins the tied relevance).
    assert out[1].chunk.path == "b.py"
    # Third pick is same-path-different-symbol (penalty 1).
    assert out[2].chunk.path == "a.py" and out[2].chunk.symbol == "bar"
    # Last is the (path, symbol) duplicate (penalty 3).
    assert out[3].chunk.symbol == "foo"


def test_mmr_stable_on_ties() -> None:
    """Two hits with identical relevance and zero penalty should resolve to
    the original input order (lower-rank-first), not arbitrary order.
    """
    hits = [
        _hit(0.5, "a.py", "x", 0),
        _hit(0.5, "b.py", "y", 1),
    ]
    out = mmr_diversify(hits, lam=0.7, top_k=2)
    assert [h.chunk.path for h in out] == ["a.py", "b.py"]
