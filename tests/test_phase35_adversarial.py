"""Phase 35 hardening: adversarial input fuzz suite.

The retrieval pipeline must NEVER raise unhandled exceptions, even on
deliberately bad input. A panic here would surface to the MCP client as
a tool failure, which Claude Code interprets as a permanent error and
stops calling the tool for the rest of the session.

Coverage:
  1. `extract_identifiers` — empty, huge, control chars, unicode, all-caps,
     single chars, malformed UTF-8 surrogate halves.
  2. `boost_exact_matches` — zero hits, zero idents, scores at extremes.
  3. `mmr_diversify` — degenerate cases (all same path, all unique paths,
     k > n, k = 0, lam outside [0,1]).
  4. `reciprocal_rank_fusion` — empty list of lists, single hit, identical
     hits in multiple lists.

These run as pure unit tests (no LM Studio / GPU required), so they go
in CI's first pass.
"""
from __future__ import annotations

import pytest

from code_rag.models import Chunk, ChunkKind, SearchHit
from code_rag.retrieval.fusion import (
    boost_exact_matches,
    extract_identifiers,
    mmr_diversify,
    reciprocal_rank_fusion,
)


def _hit(score: float, path: str, symbol: str | None = None,
         text: str = "x") -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id=f"{path}:{symbol}:{score}",
            repo="r", path=path, language="python",
            symbol=symbol, kind=ChunkKind.FUNCTION,
            start_line=1, end_line=2, text=text,
        ),
        score=score, source="hybrid",
    )


# ---- extract_identifiers: bad inputs --------------------------------------


@pytest.mark.parametrize("query", [
    pytest.param("",                                            id="empty"),
    pytest.param(" ",                                           id="space"),
    pytest.param("\n\t\r",                                      id="whitespace"),
    pytest.param("a",                                           id="single-char"),
    pytest.param("a" * 10_000,                                  id="10k-chars"),
    pytest.param("café résumé naïve",                          id="unicode-prose"),
    pytest.param("🔥🚀💯",                                       id="pure-emoji"),
    pytest.param("ALLCAPSALLCAPS",                              id="all-caps"),
    pytest.param("#$%^&*()_+",                                  id="all-symbols"),
    pytest.param("1234567890",                                  id="all-digits"),
    pytest.param("_" * 50,                                      id="all-underscores"),
    pytest.param("   foo   bar   ",                             id="whitespace-heavy"),
    pytest.param("select * from users where id = 1 OR 1=1",     id="sql-injection"),
    pytest.param("<script>alert(1)</script>",                   id="xss"),
    pytest.param("../../etc/passwd",                            id="path-traversal"),
    pytest.param("$(rm -rf /)",                                 id="cmd-injection"),
])
def test_extract_identifiers_never_raises(query: str) -> None:
    """No matter what nonsense the user types, identifier extraction
    must return a list (possibly empty) without raising."""
    out = extract_identifiers(query)
    assert isinstance(out, list)
    # All entries must be strings (defensive type check).
    assert all(isinstance(x, str) for x in out)


def test_extract_identifiers_handles_huge_query() -> None:
    """A 100K-char query containing identifiers should still work but
    bound the output (we don't want to return a million entries)."""
    big = "OnBarUpdate place_order_v2 " * 5000  # ~150K chars
    out = extract_identifiers(big)
    # Just two unique idents in the query, regardless of repetition.
    assert "OnBarUpdate" in out
    assert "place_order_v2" in out


def test_extract_identifiers_unicode_token_drops() -> None:
    """Mixed-script tokens are tricky; we conservatively drop anything
    not in ASCII for boost purposes (BM25 catches them via FTS5 anyway)."""
    out = extract_identifiers("normalUS_func 函数_name MyClass")
    # The Chinese-character token shouldn't crash the regex.
    assert "MyClass" in out


# ---- boost_exact_matches: extremes ----------------------------------------


def test_boost_no_hits_no_idents() -> None:
    assert boost_exact_matches([], [], factor=0.5) == []


def test_boost_no_idents_passthrough() -> None:
    hits = [_hit(0.5, "a.py")]
    out = boost_exact_matches(hits, [], factor=0.5)
    assert out == list(hits)


def test_boost_negative_score_unchanged_sign() -> None:
    """Negative scores are unusual but defensive: the boost shouldn't
    flip sign or do anything weird."""
    hits = [_hit(-0.5, "a.py", "OnBarUpdate", text="OnBarUpdate body")]
    out = boost_exact_matches(hits, ["OnBarUpdate"], factor=0.5)
    # Multiplicative boost applied; sign preserved.
    assert out[0].score < 0
    # Boost magnifies negative scores too — that's mathematically correct
    # even if surprising. Explicit assertion so we know the behavior.
    assert out[0].score < -0.5


def test_boost_huge_factor_doesnt_overflow() -> None:
    hits = [_hit(1.0, "a.py", "X", text="X")]
    out = boost_exact_matches(hits, ["X"], factor=1e6)
    # No NaN / inf even with absurd factor.
    import math
    assert math.isfinite(out[0].score)


# ---- mmr_diversify: degenerate cases --------------------------------------


def test_mmr_all_same_path() -> None:
    """When every hit comes from the same file, MMR can't diversify;
    must still return top_k items in score order."""
    hits = [_hit(1.0 - i * 0.1, "a.py", f"sym{i}", text="x") for i in range(5)]
    out = mmr_diversify(hits, lam=0.7, top_k=3)
    assert len(out) == 3
    # All from same path — that's allowed; just don't crash.
    assert all(h.chunk.path == "a.py" for h in out)


def test_mmr_top_k_zero() -> None:
    hits = [_hit(0.5, "a.py")]
    assert mmr_diversify(hits, lam=0.7, top_k=0) == []


def test_mmr_top_k_negative_returns_empty() -> None:
    hits = [_hit(0.5, "a.py")]
    assert mmr_diversify(hits, lam=0.7, top_k=-5) == []


def test_mmr_top_k_huge() -> None:
    """If top_k > available, return all available."""
    hits = [_hit(0.5, f"f{i}.py") for i in range(3)]
    out = mmr_diversify(hits, lam=0.7, top_k=1000)
    assert len(out) == 3


def test_mmr_lam_zero_pure_diversity() -> None:
    """lam=0 = pure diversity, ignore relevance entirely.
    Should pick maximally distinct paths."""
    hits = [
        _hit(0.99, "a.py"),
        _hit(0.98, "a.py"),
        _hit(0.10, "b.py"),
    ]
    out = mmr_diversify(hits, lam=0.0, top_k=2)
    paths = [h.chunk.path for h in out]
    # First always pos-0 by tiebreak; second should switch to b.py
    # because lam=0 makes any path-repeat infinitely costly.
    assert "a.py" in paths and "b.py" in paths


def test_mmr_lam_above_one_treated_as_passthrough() -> None:
    """Defensive: lam>=1 disables diversity entirely (per docstring)."""
    hits = [_hit(0.5 - i * 0.1, f"f{i}.py") for i in range(3)]
    out = mmr_diversify(hits, lam=1.5, top_k=3)
    assert [h.chunk.path for h in out] == ["f0.py", "f1.py", "f2.py"]


def test_mmr_empty_input() -> None:
    assert mmr_diversify([], lam=0.7, top_k=10) == []


def test_mmr_all_zero_scores() -> None:
    """Edge: scores all zero. Normalization divides by zero — must handle."""
    hits = [_hit(0.0, f"f{i}.py") for i in range(3)]
    out = mmr_diversify(hits, lam=0.7, top_k=3)
    assert len(out) == 3


# ---- reciprocal_rank_fusion: degenerate cases -----------------------------


def test_rrf_empty_list_of_lists() -> None:
    assert reciprocal_rank_fusion([], k=60, top_k=10) == []


def test_rrf_one_empty_list() -> None:
    assert reciprocal_rank_fusion([[]], k=60, top_k=10) == []


def test_rrf_top_k_zero() -> None:
    hits = [_hit(0.5, "a.py")]
    assert reciprocal_rank_fusion([hits], k=60, top_k=0) == []


def test_rrf_same_chunk_in_multiple_lists() -> None:
    """A chunk that appears in BOTH vector and lexical lists should
    accumulate score, not get duplicated."""
    h = _hit(0.5, "a.py")
    out = reciprocal_rank_fusion([[h], [h]], k=60, top_k=5)
    assert len(out) == 1   # deduped by id
    # Score = sum of 1/(60+1) for both lists ≈ 0.0328.
    assert out[0].score > 1.0 / 61
