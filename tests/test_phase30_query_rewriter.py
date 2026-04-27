"""Phase 30: query rewriting / identifier expansion + cache.

Three surfaces under test:
  1. `local_rewrite` — pure-Python casing variant generation, no LLM.
  2. `RewriteCache` — SQLite TTL cache; verifies hit/miss/expiry/eviction.
  3. `LMStudioQueryRewriter` — rewriter wired with a stubbed HTTP client to
     verify the LLM merge/fallback paths without hitting LM Studio.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from code_rag.retrieval.query_rewriter import (
    LMStudioQueryRewriter,
    Rewrite,
    RewriteCache,
    local_rewrite,
)


# ---- local rewrite ---------------------------------------------------------


def test_local_rewrite_no_idents_is_literal() -> None:
    rw = local_rewrite("how does it work")
    assert rw.via == "literal"
    assert rw.expansions == []


def test_local_rewrite_camelcase_to_snake() -> None:
    rw = local_rewrite("OnBarUpdate")
    assert "on_bar_update" in rw.expansions
    assert "on bar update" in rw.expansions


def test_local_rewrite_snake_to_camel() -> None:
    rw = local_rewrite("ensure_lm_studio_ready")
    assert "ensureLmStudioReady" in rw.expansions
    assert "ensure lm studio ready" in rw.expansions


def test_local_rewrite_mixed() -> None:
    rw = local_rewrite("Where is OnBarUpdate called by load_model")
    # Both identifiers expanded.
    assert any("on_bar_update" in e for e in rw.expansions)
    assert any("loadModel" in e for e in rw.expansions)


def test_rewrite_arms_includes_original_at_full_weight() -> None:
    rw = Rewrite(original="OnBarUpdate", expansions=["on_bar_update", "on bar update"])
    arms = rw.arms
    # Original first at weight 1.0
    assert arms[0] == ("OnBarUpdate", 1.0)
    # First expansion at 0.5, second dropped by the _MAX_ARMS=2 cap.
    assert all(w == 0.5 for _, w in arms[1:])
    assert len(arms) == 2


def test_rewrite_arms_drops_dup_expansions() -> None:
    """Expansion equal to original should be filtered."""
    rw = Rewrite(original="OnBarUpdate", expansions=["OnBarUpdate", "on_bar_update"])
    arms = rw.arms
    paths = [a[0] for a in arms]
    assert paths.count("OnBarUpdate") == 1


# ---- cache -----------------------------------------------------------------


def _open_cache(tmp_path: Path) -> RewriteCache:
    c = RewriteCache(tmp_path / "rewrite_cache.db")
    c.open()
    return c


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = _open_cache(tmp_path)
    try:
        got = asyncio.run(cache.get("nope"))
        assert got is None
    finally:
        cache.close()


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = _open_cache(tmp_path)
    try:
        rw = Rewrite(original="OnBarUpdate", expansions=["on_bar_update"])
        asyncio.run(cache.put(rw))
        got = asyncio.run(cache.get("OnBarUpdate"))
        assert got is not None
        assert got.expansions == ["on_bar_update"]
        # Cache hits are stamped via="cache" so callers can log them.
        assert got.via == "cache"
    finally:
        cache.close()


def test_cache_ttl_expiry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force time forward past the TTL — the entry should be invalidated."""
    cache = _open_cache(tmp_path)
    try:
        rw = Rewrite(original="OnBarUpdate", expansions=["on_bar_update"])
        asyncio.run(cache.put(rw))
        # Capture the real time.time before patching, otherwise the lambda
        # recurses into itself.
        real_now = time.time()
        monkeypatch.setattr(time, "time", lambda: real_now + 25 * 3600)
        got = asyncio.run(cache.get("OnBarUpdate"))
        assert got is None
    finally:
        cache.close()


def test_cache_persists_across_open(tmp_path: Path) -> None:
    """Caches are SQLite-backed; closing + reopening should preserve data."""
    cache = _open_cache(tmp_path)
    asyncio.run(cache.put(Rewrite(original="q", expansions=["q1"])))
    cache.close()

    cache2 = _open_cache(tmp_path)
    try:
        got = asyncio.run(cache2.get("q"))
        assert got is not None
        assert got.expansions == ["q1"]
    finally:
        cache2.close()


# ---- rewriter end-to-end (stubbed HTTP) ------------------------------------


def _build_rewriter_with_stub(
    *, content: str | None, http_fail: bool = False, cache: RewriteCache | None = None,
) -> LMStudioQueryRewriter:
    """Construct a rewriter whose internal httpx client is replaced with a
    mock returning `content` (or raising if http_fail)."""
    rw = LMStudioQueryRewriter(
        base_url="http://localhost:9999/v1", model="dummy", cache=cache,
    )
    fake_client = MagicMock()
    if http_fail:
        fake_client.post = AsyncMock(side_effect=RuntimeError("simulated outage"))
    else:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={
            "choices": [{"message": {"content": content}}],
        })
        fake_client.post = AsyncMock(return_value=resp)
    fake_client.aclose = AsyncMock()
    rw._client = fake_client  # type: ignore[assignment]
    return rw


def test_rewriter_skips_llm_when_no_idents() -> None:
    """For prose-only queries with no identifiers, LLM is NOT called."""
    rw = _build_rewriter_with_stub(content='{"expansions": ["nope"]}')
    out = asyncio.run(rw.rewrite("how does it work"))
    assert out.via == "literal"
    rw._client.post.assert_not_called()  # type: ignore[attr-defined]


def test_rewriter_merges_local_and_llm_expansions() -> None:
    rw = _build_rewriter_with_stub(
        content='{"expansions": ["bar update handler", "tick callback"]}',
    )
    out = asyncio.run(rw.rewrite("OnBarUpdate"))
    # Local expansions come from extract_identifiers + variants.
    assert any("on_bar_update" in e for e in out.expansions)
    # LLM expansions also present.
    assert any("bar update handler" in e for e in out.expansions)
    assert out.via == "llm"


def test_rewriter_falls_back_to_local_on_http_failure() -> None:
    rw = _build_rewriter_with_stub(content=None, http_fail=True)
    out = asyncio.run(rw.rewrite("OnBarUpdate"))
    # Local expansions still present.
    assert any("on_bar_update" in e for e in out.expansions)
    # via reflects the fallback.
    assert out.via == "local"


def test_rewriter_falls_back_on_bad_json() -> None:
    rw = _build_rewriter_with_stub(content="this is not json")
    out = asyncio.run(rw.rewrite("OnBarUpdate"))
    assert out.via == "local"
    assert any("on_bar_update" in e for e in out.expansions)


def test_rewriter_uses_cache(tmp_path: Path) -> None:
    """Second call with the same query should not invoke the LLM."""
    cache = _open_cache(tmp_path)
    try:
        rw = _build_rewriter_with_stub(
            content='{"expansions": ["one", "two"]}', cache=cache,
        )
        first = asyncio.run(rw.rewrite("OnBarUpdate"))
        assert first.via == "llm"
        # Reset the mock to verify second call is NOT made.
        post_mock = rw._client.post  # type: ignore[attr-defined]
        post_mock.reset_mock()
        second = asyncio.run(rw.rewrite("OnBarUpdate"))
        assert second.via == "cache"
        post_mock.assert_not_called()
    finally:
        cache.close()
