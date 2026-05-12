"""Phase 60-T: LMStudioReranker truncates docs before /v1/rerank POST.

THE BUG IT FIXES

vLLM-served bge-reranker-v2-m3 is pinned at `max_model_len=1024` tokens.
The Phase 60-O hotfix increased the chunker output size (hard-splitting
oversize paragraphs produced more chunks, some of them near max_chars
=2400). When the reranker received any chunk over ~2000 chars of code-
shaped text (lots of punctuation = ~2 chars/token), vLLM returned 400
on the WHOLE batch. The searcher silently fell back to noop rerank.

Result on the locked eval before this fix:
  recall@1   0.575 → 0.400   (-17.5 pp)
  recall@3   0.825 → 0.700   (-12.5 pp)
  mrr        0.684 → 0.557   (-12.7 pp)

After this fix (truncate each doc to 1800 chars before POST):
  recall@1   0.575 → 0.625   (+5.0 pp)
  mrr        0.684 → 0.700   (+1.58 pp)

The truncation only applies on the REQUEST side -- the underlying
chunks stay full-content for embedding and lexical retrieval. The
cross-encoder only needs signature + a few body lines to judge
relevance; full bodies past ~800 tokens don't add signal.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock

import pytest

from code_rag.models import Chunk, SearchHit
from code_rag.rerankers.lm_studio import LMStudioReranker


def _hit(text: str, path: str = "x.py", symbol: str = "foo") -> SearchHit:
    chunk = Chunk(
        id=f"{path}:{symbol}",
        repo="r", path=path, kind="function", language="python",
        symbol=symbol, start_line=1, end_line=10,
        content_hash="h", text=text,
    )
    return SearchHit(chunk=chunk, score=0.5, source="hybrid", match_reason="")


@pytest.mark.asyncio
async def test_rerank_truncates_long_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each doc longer than max_doc_chars must be truncated to that
    length before being placed in the POST payload. The actual SearchHit
    chunks stay intact -- this is request-side truncation only."""
    rr = LMStudioReranker("http://x", "test-model", max_doc_chars=100)

    # Stub the HTTP client. We capture the payload and return a fake
    # successful response.
    captured: dict[str, Any] = {}

    class _FakeResp:
        status_code = 200
        def raise_for_status(self) -> None: ...
        def json(self) -> dict:
            return {"results": [{"index": 0, "relevance_score": 0.9}]}

    async def _fake_post(url: str, *, json: dict) -> _FakeResp:
        captured["url"] = url
        captured["payload"] = json
        return _FakeResp()

    async def _fake_get(url: str, **_kwargs):
        class _M:
            status_code = 200
            def raise_for_status(self) -> None: ...
            def json(self) -> dict:
                return {"data": [{"id": "test-model"}]}
        return _M()

    rr._endpoint_ok = True  # skip /models probe
    # Inject a fake client. _ensure_client returns this without re-creating.
    fake_client = AsyncMock()
    fake_client.post = _fake_post
    fake_client.get = _fake_get
    rr._client = fake_client  # type: ignore[assignment]

    long_doc = "a" * 1000  # well over the 100-char limit
    short_doc = "short"
    hits = [_hit(long_doc), _hit(short_doc)]
    await rr.rerank("query", hits, top_k=2)

    assert "documents" in captured["payload"]
    sent = captured["payload"]["documents"]
    assert len(sent[0]) == 100, \
        f"long doc not truncated: sent {len(sent[0])} chars, expected 100"
    assert sent[0] == "a" * 100
    assert sent[1] == "short", "short doc must pass through unchanged"


@pytest.mark.asyncio
async def test_rerank_underlying_chunk_text_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Truncation is request-side ONLY. The Chunk objects we received
    must NOT be mutated -- downstream consumers (telemetry, get_chunk_text,
    LSP-style "jump to definition") rely on the full text."""
    rr = LMStudioReranker("http://x", "test-model", max_doc_chars=50)

    async def _fake_post(url: str, *, json: dict):
        class _R:
            status_code = 200
            def raise_for_status(self) -> None: ...
            def json(self) -> dict:
                return {"results": [{"index": 0, "relevance_score": 0.9}]}
        return _R()

    rr._endpoint_ok = True
    fake_client = AsyncMock()
    fake_client.post = _fake_post
    rr._client = fake_client  # type: ignore[assignment]

    long_text = "z" * 500
    h = _hit(long_text)
    await rr.rerank("q", [h], top_k=1)

    # The Chunk is immutable per its pydantic config; assert defensively
    # anyway -- if we ever switch to mutable models, this catches it.
    assert h.chunk.text == long_text, \
        "rerank must not mutate underlying chunk text"
    assert len(h.chunk.text) == 500


@pytest.mark.asyncio
async def test_rerank_default_max_is_safe_for_bge_reranker_v2_m3() -> None:
    """The default max_doc_chars must keep us under
    bge-reranker-v2-m3's 1024-token context window for realistic
    code-shaped content (~2 chars per token). 1800 chars × (1 token
    / 2 chars) = ~900 tokens, plus ~100 tokens of query + special
    tokens fits comfortably."""
    rr = LMStudioReranker("http://x", "test-model")
    assert rr._max_doc_chars <= 2000, \
        f"default max_doc_chars={rr._max_doc_chars} exceeds the empirically " \
        "measured threshold for bge-reranker-v2-m3 on code content"
    assert rr._max_doc_chars >= 1000, \
        f"default max_doc_chars={rr._max_doc_chars} is too aggressive; would " \
        "discard useful cross-encoder signal"
