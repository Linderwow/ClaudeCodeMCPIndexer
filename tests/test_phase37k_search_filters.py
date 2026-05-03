"""Phase 37-K: better MCP filter UX.

Pure-function tests for the two new SearchParams knobs:
  - recent_files_only_days  (mtime cutoff filter in _apply_filters)
  - prefer_root             (multiplicative boost in _apply_root_boost)

The MCP wiring + tool schema is exercised indirectly via the existing
MCP test suite; these tests prove the retrieval primitives are correct.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from code_rag.models import Chunk, ChunkKind, SearchHit
from code_rag.retrieval.search import HybridSearcher, SearchParams


def _hit(path: str, score: float = 0.5, language: str = "python") -> SearchHit:
    chunk = Chunk(
        id=f"id-{path}-{score}",
        repo="repo",
        path=path,
        language=language,
        symbol=None,
        kind=ChunkKind.FUNCTION,
        start_line=1,
        end_line=10,
        text="def f():\n    pass\n",
    )
    return SearchHit(chunk=chunk, score=score, source="hybrid",
                    match_reason="fusion")


# ---- recent_files_only_days ----------------------------------------------

def test_recent_files_only_drops_old(tmp_path: Path) -> None:
    """A file with mtime older than the cutoff is filtered out."""
    old = tmp_path / "old.py"
    old.write_text("x = 1\n")
    # Backdate to 60 days ago.
    sixty_days_ago = time.time() - 60 * 86400
    os.utime(old, (sixty_days_ago, sixty_days_ago))

    new = tmp_path / "new.py"
    new.write_text("y = 2\n")     # mtime is now

    hits = [
        _hit(str(old)),
        _hit(str(new)),
    ]
    params = SearchParams(recent_files_only_days=30)
    out = HybridSearcher._apply_filters(hits, params)
    assert len(out) == 1
    assert out[0].chunk.path == str(new)


def test_recent_files_only_zero_disables_filter(tmp_path: Path) -> None:
    """`recent_files_only_days=0` is a no-op (semantically: zero-day window
    is meaningless, so we treat it as 'don't filter'). This avoids an
    edge case where `days=0` translates to cutoff=now and all already-
    indexed files look 'too old' due to clock skew."""
    yesterday = tmp_path / "yesterday.py"
    yesterday.write_text("x = 1\n")
    one_day_one_hour_ago = time.time() - 25 * 3600
    os.utime(yesterday, (one_day_one_hour_ago, one_day_one_hour_ago))

    today = tmp_path / "today.py"
    today.write_text("y = 2\n")

    hits = [_hit(str(yesterday)), _hit(str(today))]
    params = SearchParams(recent_files_only_days=0)
    out = HybridSearcher._apply_filters(hits, params)
    assert len(out) == 2     # both pass — filter is disabled


def test_recent_files_only_one_day(tmp_path: Path) -> None:
    """`days=1` keeps files modified within the last 24h."""
    yesterday = tmp_path / "yesterday.py"
    yesterday.write_text("x = 1\n")
    twenty_five_hours_ago = time.time() - 25 * 3600
    os.utime(yesterday, (twenty_five_hours_ago, twenty_five_hours_ago))

    today = tmp_path / "today.py"
    today.write_text("y = 2\n")

    hits = [_hit(str(yesterday)), _hit(str(today))]
    params = SearchParams(recent_files_only_days=1)
    out = HybridSearcher._apply_filters(hits, params)
    assert len(out) == 1
    assert out[0].chunk.path == str(today)


def test_recent_files_filter_dropped_when_none() -> None:
    """When `recent_files_only_days` is None, no filtering happens — even
    nonexistent paths pass through."""
    hits = [_hit("/no/such/path/a.py"), _hit("/no/such/path/b.py")]
    params = SearchParams(recent_files_only_days=None)
    out = HybridSearcher._apply_filters(hits, params)
    assert len(out) == 2


def test_recent_files_drops_missing_files(tmp_path: Path) -> None:
    """If recent filter is on AND the file no longer exists, drop it
    (deleted between index and query). Don't crash on the stat()."""
    real = tmp_path / "real.py"
    real.write_text("x = 1\n")
    hits = [
        _hit(str(real)),
        _hit(str(tmp_path / "ghost.py")),  # doesn't exist
    ]
    params = SearchParams(recent_files_only_days=30)
    out = HybridSearcher._apply_filters(hits, params)
    assert len(out) == 1
    assert out[0].chunk.path == str(real)


# ---- prefer_root boost ---------------------------------------------------

def test_prefer_root_boost_lifts_matching_hits() -> None:
    """A hit under prefer_root gets multiplied by (1 + boost)."""
    hits = [
        _hit("repoA/src/foo.py", score=0.50),
        _hit("repoB/src/bar.py", score=0.55),
    ]
    params = SearchParams(prefer_root="repoA", prefer_root_boost=0.25)
    out = HybridSearcher._apply_root_boost(hits, params)

    # repoA hit was boosted by 25% -> 0.625; repoB hit unchanged -> 0.55.
    boosted = next(h for h in out if h.chunk.path == "repoA/src/foo.py")
    untouched = next(h for h in out if h.chunk.path == "repoB/src/bar.py")
    assert boosted.score == pytest.approx(0.625)
    assert untouched.score == pytest.approx(0.55)
    # Sort order: boosted first now.
    assert out[0].chunk.path == "repoA/src/foo.py"


def test_prefer_root_case_insensitive_and_slash_agnostic() -> None:
    """Path comparison normalizes case + slashes (Windows-friendly)."""
    hits = [_hit("RepoA\\src\\foo.py", score=0.5)]
    params = SearchParams(prefer_root="repoa", prefer_root_boost=0.5)
    out = HybridSearcher._apply_root_boost(hits, params)
    assert out[0].score == pytest.approx(0.75)


def test_prefer_root_disabled_when_unset() -> None:
    """No prefer_root → no-op."""
    hits = [_hit("any/path.py", score=0.5)]
    params = SearchParams()  # defaults
    out = HybridSearcher._apply_root_boost(hits, params)
    assert len(out) == 1
    assert out[0].score == 0.5


def test_prefer_root_zero_boost_is_noop() -> None:
    """Boost factor of 0 short-circuits the function."""
    hits = [_hit("repoA/foo.py", score=0.5)]
    params = SearchParams(prefer_root="repoA", prefer_root_boost=0.0)
    out = HybridSearcher._apply_root_boost(hits, params)
    assert out[0].score == 0.5
    assert "+prefer_root" not in (out[0].match_reason or "")


def test_prefer_root_marks_match_reason() -> None:
    """The boosted hit's match_reason is annotated with the boost factor
    so agents can see why it rose."""
    hits = [_hit("repoA/x.py", score=0.5)]
    params = SearchParams(prefer_root="repoA", prefer_root_boost=0.25)
    out = HybridSearcher._apply_root_boost(hits, params)
    reason = out[0].match_reason or ""
    assert "prefer_root" in reason
    assert "+0.25" in reason


def test_prefer_root_exact_path_matches() -> None:
    """When path == prefer_root exactly (no trailing slash), still boosted."""
    hits = [_hit("repoA", score=0.5)]
    params = SearchParams(prefer_root="repoA", prefer_root_boost=0.25)
    out = HybridSearcher._apply_root_boost(hits, params)
    assert out[0].score == pytest.approx(0.625)


def test_prefer_root_partial_prefix_doesnt_match() -> None:
    """`repoAB` should NOT be boosted just because we asked for `repoA`."""
    hits = [
        _hit("repoA/x.py", score=0.5),
        _hit("repoAB/y.py", score=0.5),
    ]
    params = SearchParams(prefer_root="repoA", prefer_root_boost=0.25)
    out = HybridSearcher._apply_root_boost(hits, params)
    # Only repoA/x.py was boosted.
    boosted_count = sum(1 for h in out if h.score > 0.5 + 1e-9)
    assert boosted_count == 1
    assert next(h for h in out if h.score > 0.5 + 1e-9).chunk.path == "repoA/x.py"


def test_prefer_root_resorts_after_boost() -> None:
    """A hit that was 4th by score but in prefer_root should rise."""
    hits = [
        _hit("repoB/a.py", score=0.90),
        _hit("repoB/b.py", score=0.85),
        _hit("repoB/c.py", score=0.80),
        _hit("repoA/d.py", score=0.75),    # would be 4th
    ]
    params = SearchParams(prefer_root="repoA", prefer_root_boost=0.30)
    out = HybridSearcher._apply_root_boost(hits, params)
    # 0.75 * 1.30 = 0.975 → now ranks first.
    assert out[0].chunk.path == "repoA/d.py"
    assert out[0].score == pytest.approx(0.975)
