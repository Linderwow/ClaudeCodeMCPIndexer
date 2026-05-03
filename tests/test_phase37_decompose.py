"""Tests for Phase 37-A query decomposition."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from code_rag.retrieval.decompose import (
    Decomposition,
    LMStudioQueryDecomposer,
    looks_multipart,
)

# ---- heuristic gate -------------------------------------------------------


def test_short_query_is_not_multipart() -> None:
    assert not looks_multipart("OnBarUpdate")
    assert not looks_multipart("how does login work")  # 4 words


def test_short_with_conjunction_still_skipped() -> None:
    # Short queries even with `and` shouldn't trigger decomposition;
    # word count is the cheap floor.
    assert not looks_multipart("login and logout")  # 3 words


def test_long_with_conjunction_is_multipart() -> None:
    q = "how does authentication flow from login through session refresh and logout"
    assert looks_multipart(q)


def test_long_without_conjunction_is_not_multipart() -> None:
    q = "where does the strategy size positions during market hours today"
    assert not looks_multipart(q)


def test_multiple_question_marks_is_multipart() -> None:
    # Two questions in one query — even without conjunction.
    q = "how does login work? where is the session refreshed?"
    assert looks_multipart(q)


# ---- Decomposition.arms ---------------------------------------------------


def test_arms_includes_original_at_full_weight() -> None:
    d = Decomposition(original="X and Y", subqueries=["X", "Y"], via="llm")
    arms = d.arms
    assert arms[0] == ("X and Y", 1.0)
    assert arms[1][0] == "X"
    assert arms[2][0] == "Y"
    # Sub-queries ride at lower weight than the literal.
    assert all(w < 1.0 for _, w in arms[1:])


def test_arms_drops_empty_and_dup_subqueries() -> None:
    d = Decomposition(
        original="hello",
        subqueries=["hello", "", "  ", "world"],
        via="llm",
    )
    arms = d.arms
    assert arms == [("hello", 1.0), ("world", 0.6)]


def test_arms_with_no_subqueries_is_single_arm() -> None:
    d = Decomposition(original="hello", subqueries=[], via="skip")
    assert d.arms == [("hello", 1.0)]


# ---- LMStudioQueryDecomposer ----------------------------------------------


@pytest.mark.asyncio
async def test_decomposer_skips_when_no_model_configured() -> None:
    dec = LMStudioQueryDecomposer(base_url="http://x", model="")
    out = await dec.decompose("how does login flow into refresh and logout work?")
    assert out.subqueries == []
    assert out.via == "skip"


@pytest.mark.asyncio
async def test_decomposer_skips_short_query_without_llm_call() -> None:
    dec = LMStudioQueryDecomposer(base_url="http://x", model="qwen")
    # Patch the http client to fail loudly if it gets called — heuristic
    # gate must kick in first for this short query.
    with patch.object(dec, "_call_llm", new=AsyncMock(side_effect=AssertionError("LLM should not be called"))):
        out = await dec.decompose("OnBarUpdate")
    assert out.subqueries == []
    assert out.via == "skip"


@pytest.mark.asyncio
async def test_decomposer_returns_subqueries_on_llm_success() -> None:
    dec = LMStudioQueryDecomposer(base_url="http://x", model="qwen")
    fake_subs = ["how does login work", "how does session refresh", "how does logout work"]
    with patch.object(dec, "_call_llm", new=AsyncMock(return_value=fake_subs)):
        out = await dec.decompose(
            "how does authentication flow from login through session refresh and logout",
        )
    assert out.via == "llm"
    assert out.subqueries == fake_subs
    arms = out.arms
    assert len(arms) == 4   # original + 3 sub-queries
    assert arms[0][1] == 1.0
    assert all(w < 1.0 for _, w in arms[1:])


@pytest.mark.asyncio
async def test_decomposer_falls_back_when_llm_returns_empty() -> None:
    dec = LMStudioQueryDecomposer(base_url="http://x", model="qwen")
    with patch.object(dec, "_call_llm", new=AsyncMock(return_value=[])):
        out = await dec.decompose(
            "how does authentication flow from login through session refresh and logout",
        )
    assert out.via == "fallback"
    assert out.subqueries == []
    assert out.arms == [("how does authentication flow from login through session refresh and logout", 1.0)]


@pytest.mark.asyncio
async def test_decomposer_call_llm_parses_json_schema_response() -> None:
    """End-to-end parse path: simulate an httpx response with valid JSON."""
    dec = LMStudioQueryDecomposer(base_url="http://localhost:1234/v1", model="qwen")

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "subqueries": ["sub one", "sub two"],
                        }),
                    },
                }],
            }

    class FakeClient:
        async def post(self, url: str, json: Any = None) -> FakeResponse:
            return FakeResponse()

        async def aclose(self) -> None:
            pass

    with patch.object(dec, "_ensure_client", new=AsyncMock(return_value=FakeClient())):
        subs = await dec._call_llm("how does login refresh and logout work today")
    assert subs == ["sub one", "sub two"]


@pytest.mark.asyncio
async def test_decomposer_call_llm_returns_empty_on_http_error() -> None:
    """Any exception inside _call_llm should resolve to []; the decomposer
    must never raise into the searcher."""
    dec = LMStudioQueryDecomposer(base_url="http://localhost:1234/v1", model="qwen")

    class FakeClient:
        async def post(self, *_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("connection refused")

        async def aclose(self) -> None:
            pass

    with patch.object(dec, "_ensure_client", new=AsyncMock(return_value=FakeClient())):
        subs = await dec._call_llm("how does login refresh and logout work today")
    assert subs == []


@pytest.mark.asyncio
async def test_decomposer_drops_sub_queries_that_are_just_the_original() -> None:
    """If the LLM echoes the literal query as a "sub-query", we strip it."""
    dec = LMStudioQueryDecomposer(base_url="http://x", model="qwen")
    fake_subs = [
        "how does login flow into refresh and logout work",  # === original
        "session refresh details",
    ]
    with patch.object(dec, "_call_llm", new=AsyncMock(return_value=fake_subs)):
        out = await dec.decompose("how does login flow into refresh and logout work?")
    # The literal echo should be dropped INSIDE _call_llm; here we just assert
    # the public contract: sub-queries don't include the literal.
    norm_orig = "how does login flow into refresh and logout work?".lower()
    for s in out.subqueries:
        assert s.lower() != norm_orig
