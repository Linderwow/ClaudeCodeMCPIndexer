"""Phase 28: identifier extraction + exact-match boost.

The exact-match boost re-asserts a floor that BM25 alone can't keep when
RRF blends it with a strong dense embedder: when the user types an
identifier verbatim, chunks containing that identifier outrank chunks
that are merely semantically similar.
"""
from __future__ import annotations

from code_rag.models import Chunk, ChunkKind, SearchHit
from code_rag.retrieval.fusion import (
    boost_exact_matches,
    extract_identifiers,
)


def _hit(score: float, path: str, symbol: str | None, text: str) -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id=f"{path}:{symbol}",
            repo="r",
            path=path,
            language="python",
            symbol=symbol,
            kind=ChunkKind.FUNCTION,
            start_line=1, end_line=2,
            text=text,
        ),
        score=score,
        source="hybrid",
    )


# ---- extract_identifiers ---------------------------------------------------


def test_extracts_camelcase() -> None:
    assert extract_identifiers("how does OnBarUpdate work") == ["OnBarUpdate"]


def test_extracts_snake_case() -> None:
    """`first` is pure-lowercase english — drops out. Only the snake_case
    identifier survives."""
    assert extract_identifiers("call ensure_lm_studio_ready first") == [
        "ensure_lm_studio_ready",
    ]


def test_filters_short_tokens() -> None:
    """Tokens shorter than 4 chars don't qualify — too noisy for boost."""
    assert extract_identifiers("the api on bar") == []


def test_filters_stopwords() -> None:
    """Common code-shaped english words like 'class' / 'self' don't anchor."""
    out = extract_identifiers("which class with self args")
    # All stopwords — nothing eligible.
    assert out == []


def test_dedupes_preserving_order() -> None:
    """`calls` and `again` are pure-lowercase english — drop out. Only
    OnBarUpdate qualifies, and it dedupes to a single entry."""
    out = extract_identifiers("OnBarUpdate calls OnBarUpdate again")
    assert out == ["OnBarUpdate"]


def test_mixed_query() -> None:
    out = extract_identifiers("Where is OnBarUpdate? It calls place_order_v2")
    assert "OnBarUpdate" in out
    assert "place_order_v2" in out


# ---- boost_exact_matches ----------------------------------------------------


def test_boost_lifts_matching_chunk_above_nonmatching() -> None:
    """Chunk B has a lower base score but contains the identifier — should
    end up ranked above A after the boost."""
    hits = [
        _hit(1.0, "a.py", "foo",        text="totally unrelated content"),
        _hit(0.5, "b.py", "OnBarUpdate", text="def OnBarUpdate(self): ..."),
    ]
    out = boost_exact_matches(hits, ["OnBarUpdate"], factor=0.5)
    assert out[0].chunk.path == "b.py"
    # Reason was stamped.
    assert "exact-match" in (out[0].match_reason or "")


def test_boost_no_idents_is_passthrough() -> None:
    hits = [_hit(0.5, "a.py", "x", "body")]
    out = boost_exact_matches(hits, [], factor=0.5)
    assert out == list(hits)


def test_boost_zero_factor_is_passthrough() -> None:
    """factor=0 -> no boost applied even if identifiers match."""
    hits = [_hit(0.5, "a.py", "OnBarUpdate", "OnBarUpdate body")]
    out = boost_exact_matches(hits, ["OnBarUpdate"], factor=0.0)
    # Score unchanged within float epsilon.
    assert abs(out[0].score - 0.5) < 1e-9


def test_struct_match_beats_body_match() -> None:
    """Chunk whose SYMBOL contains the ident outranks one that just has it
    in body text (when their base scores are equal). Verifies the
    structural-match bonus is heavier than the body-match bonus.
    """
    hits = [
        _hit(1.0, "a.py", "wrapper",     text="calls OnBarUpdate inside"),
        _hit(1.0, "b.py", "OnBarUpdate", text="real implementation"),
    ]
    out = boost_exact_matches(hits, ["OnBarUpdate"], factor=0.5)
    assert out[0].chunk.symbol == "OnBarUpdate"


def test_match_count_capped() -> None:
    """A chunk can't game the boost by mentioning the identifier 50 times.
    Capped at 6 (so roughly 4x boost max with default factor).
    """
    spammy = "OnBarUpdate " * 50
    hits = [
        _hit(0.1, "spam.py", "SpamFn",     text=spammy),
        _hit(0.5, "real.py", "OnBarUpdate", text="def OnBarUpdate(): ..."),
    ]
    out = boost_exact_matches(hits, ["OnBarUpdate"], factor=0.5)
    # The chunk where OnBarUpdate is the SYMBOL should still win (struct
    # match weighted 2x), even though body has many occurrences.
    assert out[0].chunk.path == "real.py"


def test_case_insensitive_match() -> None:
    """`OnBarUpdate` in query matches `onbarupdate` in body. BM25 folds case
    so the boost should too."""
    hits = [_hit(0.5, "a.py", None, text="def onbarupdate(): pass")]
    out = boost_exact_matches(hits, ["OnBarUpdate"], factor=0.5)
    assert out[0].score > 0.5
