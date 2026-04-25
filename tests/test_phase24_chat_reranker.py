"""Phase 24: chat-completion-based reranker tests.

We exercise the parser exhaustively (it's the bug-prone part), the prompt
assembly, and the end-to-end rerank() with a mock httpx transport so no
LM Studio process is needed.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from code_rag.models import Chunk, ChunkKind, SearchHit
from code_rag.rerankers.lm_chat import LMStudioChatReranker

# ---- helpers ---------------------------------------------------------------


def _hit(idx: int, *, path: str | None = None, symbol: str | None = None,
         text: str | None = None, source: str = "hybrid",
         match_reason: str | None = None) -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id=f"chunk-{idx}",
            repo="r",
            path=path or f"f{idx}.py",
            language="python",
            symbol=symbol or f"sym_{idx}",
            kind=ChunkKind.FUNCTION,
            start_line=1, end_line=10,
            text=text or f"def sym_{idx}(): return {idx}",
        ),
        score=1.0 - idx * 0.01,
        source=source,  # type: ignore[arg-type]
        match_reason=match_reason,
    )


def _hits(n: int) -> list[SearchHit]:
    return [_hit(i) for i in range(n)]


def _make_reranker(transport: httpx.MockTransport) -> LMStudioChatReranker:
    rr = LMStudioChatReranker(
        base_url="http://test/v1",
        model="qwen/qwen3-1.7b",
        timeout_s=5.0,
        max_candidates=20,
        max_chars_per_doc=300,
    )
    # Inject a mock-backed client so we never touch the network.
    rr._client = httpx.AsyncClient(base_url="http://test/v1", transport=transport)
    return rr


def _ok_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}}],
    })


# ---- parser ----------------------------------------------------------------


@pytest.mark.parametrize("content,expected", [
    # Strict JSON object with "ranking" key.
    ('{"ranking": [3, 1, 0, 2]}', [3, 1, 0, 2]),
    # Bare JSON array.
    ("[2, 0, 1, 3]", [2, 0, 1, 3]),
    # JSON wrapped in ```json fence (small models leak).
    ('```json\n{"ranking": [1, 2, 0]}\n```', [1, 2, 0]),
    # JSON wrapped in plain ``` fence.
    ('```\n{"ranking": [0, 3]}\n```', [0, 3]),
    # JSON object using "ids" alias.
    ('{"ids": [3, 0]}', [3, 0]),
    # JSON object using "order" alias.
    ('{"order": [2, 1]}', [2, 1]),
    # Object with prose around it.
    ('Here is the ranking: {"ranking": [1, 0, 2]} thanks!', [1, 0, 2]),
    # Float ids tolerated (some models emit "1.0" for integer-like).
    ('[1.0, 0.0, 2.0]', [1, 0, 2]),
])
def test_parser_happy_paths(content: str, expected: list[int]) -> None:
    parsed = LMStudioChatReranker._parse_ranking(content, n_candidates=4)
    assert parsed == expected


def test_parser_drops_out_of_range_ids() -> None:
    # 99 is out of range for 4 candidates; must be filtered, others kept in order.
    parsed = LMStudioChatReranker._parse_ranking('[2, 99, 0, 4, 1]', n_candidates=4)
    assert parsed == [2, 0, 1]


def test_parser_falls_back_to_int_scrape_when_json_unparseable() -> None:
    # No JSON at all — last-ditch parser scrapes ints in order.
    parsed = LMStudioChatReranker._parse_ranking(
        "I think the order is 3, then 1, then 0.", n_candidates=4,
    )
    assert parsed == [3, 1, 0]


def test_parser_returns_none_on_total_failure() -> None:
    # No ints, no JSON — caller must take the fallback path.
    parsed = LMStudioChatReranker._parse_ranking("nope", n_candidates=4)
    assert parsed is None


def test_parser_returns_none_for_non_string_input() -> None:
    assert LMStudioChatReranker._parse_ranking(None, n_candidates=4) is None  # type: ignore[arg-type]
    assert LMStudioChatReranker._parse_ranking(42, n_candidates=4) is None  # type: ignore[arg-type]


# ---- prompt assembly -------------------------------------------------------


def test_prompt_truncates_long_doc_text() -> None:
    rr = LMStudioChatReranker(
        base_url="http://x", model="m", max_chars_per_doc=20,
    )
    long_text = "x" * 500
    hit = _hit(0, text=long_text)
    prompt = rr._build_prompt("query", [hit])
    # The candidate line must contain at most max_chars_per_doc of the body.
    candidate_line = next(ln for ln in prompt.splitlines() if ln.startswith("[0]"))
    # Strip the prefix; whatever remains as the snippet must be <= 20 chars.
    snippet_part = candidate_line.split("::", 2)[-1].strip()
    assert len(snippet_part) <= 20


def test_prompt_lists_every_candidate_with_index() -> None:
    rr = LMStudioChatReranker(base_url="http://x", model="m")
    hits = _hits(5)
    prompt = rr._build_prompt("q", hits)
    for i in range(5):
        assert f"[{i}]" in prompt
    assert prompt.lower().count("query") >= 1


# ---- end-to-end rerank() ---------------------------------------------------


def test_rerank_reorders_per_llm_response() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        # LLM ranks: 2 most relevant, then 0, then 1.
        return _ok_response('{"ranking": [2, 0, 1]}')
    rr = _make_reranker(httpx.MockTransport(handler))
    hits = _hits(3)
    out = asyncio.run(rr.rerank("q", hits, top_k=3))
    assert [h.chunk.id for h in out] == ["chunk-2", "chunk-0", "chunk-1"]
    # source should be 'rerank' and match_reason should mention the new rank.
    assert all(h.source == "rerank" for h in out)
    assert all((h.match_reason or "").startswith("rerank chat") for h in out)


def test_rerank_truncates_to_top_k() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _ok_response('{"ranking": [4, 3, 2, 1, 0]}')
    rr = _make_reranker(httpx.MockTransport(handler))
    hits = _hits(5)
    out = asyncio.run(rr.rerank("q", hits, top_k=2))
    assert len(out) == 2
    assert [h.chunk.id for h in out] == ["chunk-4", "chunk-3"]


def test_rerank_pads_with_unranked_when_llm_omits_some() -> None:
    """If the LLM only ranks a subset, the rest fall in input order so we
    still cover top_k."""
    def handler(req: httpx.Request) -> httpx.Response:
        # LLM only picks 2; the other 3 should fill in the original order.
        return _ok_response('{"ranking": [4, 1]}')
    rr = _make_reranker(httpx.MockTransport(handler))
    hits = _hits(5)
    out = asyncio.run(rr.rerank("q", hits, top_k=5))
    ids = [h.chunk.id for h in out]
    assert ids[0] == "chunk-4"
    assert ids[1] == "chunk-1"
    # Remaining: 0, 2, 3 in input order.
    assert ids[2:] == ["chunk-0", "chunk-2", "chunk-3"]


def test_rerank_ignores_duplicate_ids_from_llm() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _ok_response('{"ranking": [1, 1, 1, 0]}')
    rr = _make_reranker(httpx.MockTransport(handler))
    hits = _hits(3)
    out = asyncio.run(rr.rerank("q", hits, top_k=3))
    # Dupes deduped; padding fills with input-order remainders.
    assert [h.chunk.id for h in out] == ["chunk-1", "chunk-0", "chunk-2"]


def test_rerank_falls_back_to_input_order_on_http_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")
    rr = _make_reranker(httpx.MockTransport(handler))
    hits = _hits(4)
    out = asyncio.run(rr.rerank("q", hits, top_k=3))
    assert [h.chunk.id for h in out] == ["chunk-0", "chunk-1", "chunk-2"]
    # Reason should mark these as fallback so debugging is obvious.
    assert all("fallback" in (h.match_reason or "") for h in out)


def test_rerank_falls_back_when_response_unparseable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _ok_response("I'm sorry, I can't help with that.")
    rr = _make_reranker(httpx.MockTransport(handler))
    hits = _hits(3)
    out = asyncio.run(rr.rerank("q", hits, top_k=3))
    # Fallback preserves input order.
    assert [h.chunk.id for h in out] == ["chunk-0", "chunk-1", "chunk-2"]
    assert all("fallback" in (h.match_reason or "") for h in out)


def test_rerank_handles_empty_hits() -> None:
    rr = _make_reranker(httpx.MockTransport(lambda r: _ok_response("{}")))
    out = asyncio.run(rr.rerank("q", [], top_k=5))
    assert out == []


def test_rerank_handles_single_hit_without_calling_llm() -> None:
    """Single-hit shortcut: skip the LLM call entirely."""
    calls: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return _ok_response('{"ranking": [0]}')
    rr = _make_reranker(httpx.MockTransport(handler))
    out = asyncio.run(rr.rerank("q", [_hit(0)], top_k=8))
    assert len(out) == 1
    assert out[0].chunk.id == "chunk-0"
    assert calls == [], "single-hit case must not call the LLM"


def test_rerank_caps_to_max_candidates_window() -> None:
    """If hits > max_candidates, only the top window is shown to the LLM.
    Everything else is dropped — caller already chose top-N upstream."""
    captured_payloads: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _j
        captured_payloads.append(_j.loads(req.content))
        # Reverse the visible window.
        return _ok_response('{"ranking": [4, 3, 2, 1, 0]}')

    rr = LMStudioChatReranker(
        base_url="http://test/v1", model="m", max_candidates=5,
    )
    rr._client = httpx.AsyncClient(
        base_url="http://test/v1", transport=httpx.MockTransport(handler),
    )
    hits = _hits(20)  # 20 candidates, but max_candidates=5
    out = asyncio.run(rr.rerank("q", hits, top_k=3))
    assert len(out) == 3

    # The prompt should only mention candidates 0..4, not 5..19.
    prompt = captured_payloads[0]["messages"][0]["content"]
    for i in range(5):
        assert f"[{i}]" in prompt
    for i in range(5, 20):
        assert f"[{i}]" not in prompt


def test_rerank_preserves_prior_match_reason() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _ok_response('{"ranking": [0, 1]}')
    rr = _make_reranker(httpx.MockTransport(handler))
    hits = [
        _hit(0, match_reason="hybrid rrf from [vector r1, lexical r2]"),
        _hit(1, match_reason="hybrid rrf from [lexical r1]"),
    ]
    out = asyncio.run(rr.rerank("q", hits, top_k=2))
    assert "hybrid rrf" in (out[0].match_reason or "")
    assert "rerank chat" in (out[0].match_reason or "")


# ---- prompt safety: don't crash on weird content --------------------------


def test_rerank_handles_chunk_with_no_symbol_or_text() -> None:
    """Real-world chunks sometimes have None for symbol or empty text. The
    prompt builder must produce a valid line for them anyway."""
    def handler(req: httpx.Request) -> httpx.Response:
        return _ok_response('{"ranking": [0]}')
    rr = _make_reranker(httpx.MockTransport(handler))
    hit = SearchHit(
        chunk=Chunk(
            id="empty", repo="r", path="x.md", language="markdown",
            symbol=None, kind=ChunkKind.DOC, start_line=1, end_line=1, text="",
        ),
        score=0.5, source="hybrid",
    )
    out = asyncio.run(rr.rerank("q", [hit], top_k=1))
    assert len(out) == 1
    assert out[0].chunk.id == "empty"
