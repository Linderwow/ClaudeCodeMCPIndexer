"""Tests for Phase 37-B post-rerank reflection."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from code_rag.models import Chunk, ChunkKind, SearchHit
from code_rag.retrieval.reflection import LMStudioReflector


def _hit(cid: str, score: float, text: str = "body") -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id=cid, repo="repo", path="a.py", language="python",
            symbol="f", kind=ChunkKind.FUNCTION,
            start_line=1, end_line=2, text=text,
        ),
        score=score,
    )


@pytest.mark.asyncio
async def test_reflector_skips_when_no_model() -> None:
    refl = LMStudioReflector(base_url="http://x", model="")
    hits = [_hit("a", 0.3), _hit("b", 0.2)]
    out = await refl.reflect("q", hits)
    assert out is hits  # passthrough


@pytest.mark.asyncio
async def test_reflector_skips_when_top1_already_confident() -> None:
    refl = LMStudioReflector(base_url="http://x", model="qwen", skip_if_top1_above=0.7)
    hits = [_hit("a", 0.85), _hit("b", 0.4)]
    # If the LLM is reached, _score raises — confirms gate skipped it.
    with patch.object(
        refl, "_score",
        new=AsyncMock(side_effect=AssertionError("LLM should not be called")),
    ):
        out = await refl.reflect("q", hits)
    assert out == hits


@pytest.mark.asyncio
async def test_reflector_blends_scores_and_reorders() -> None:
    refl = LMStudioReflector(
        base_url="http://x", model="qwen",
        skip_if_top1_above=0.95,   # force the gate to NOT skip
        blend_alpha=0.5,            # equal weight blend
    )
    hits = [_hit("a", 0.5), _hit("b", 0.3)]
    # Reflection thinks "b" answers the question better than "a".
    with patch.object(refl, "_score",
                      new=AsyncMock(return_value={"a": 0.0, "b": 1.0})):
        out = await refl.reflect("q", hits)
    # Blended scores: a -> 0.5*0.5 + 0.5*0.0 = 0.25
    #                 b -> 0.5*0.3 + 0.5*1.0 = 0.65
    # b should now be first.
    assert out[0].chunk.id == "b"
    assert out[1].chunk.id == "a"


@pytest.mark.asyncio
async def test_reflector_passes_through_when_llm_fails() -> None:
    refl = LMStudioReflector(
        base_url="http://x", model="qwen", skip_if_top1_above=0.95,
    )
    hits = [_hit("a", 0.5), _hit("b", 0.3)]
    with patch.object(refl, "_score", new=AsyncMock(return_value={})):
        out = await refl.reflect("q", hits)
    assert out == hits


@pytest.mark.asyncio
async def test_reflector_only_rescues_top_k_keeps_tail_intact() -> None:
    refl = LMStudioReflector(
        base_url="http://x", model="qwen",
        skip_if_top1_above=0.95, top_k=2,
    )
    hits = [_hit("a", 0.5), _hit("b", 0.4), _hit("c", 0.3)]
    with patch.object(refl, "_score",
                      new=AsyncMock(return_value={"a": 0.0, "b": 1.0})):
        out = await refl.reflect("q", hits)
    # Tail (c) preserved at the end; head reordered.
    assert out[-1].chunk.id == "c"
    assert {h.chunk.id for h in out[:2]} == {"a", "b"}
    assert out[0].chunk.id == "b"  # b promoted via reflection


@pytest.mark.asyncio
async def test_reflector_score_clamps_out_of_range_values() -> None:
    """A misbehaving model returning score=2.5 must be clamped to [0, 1]
    so the blend stays bounded."""
    refl = LMStudioReflector(base_url="http://x", model="qwen")

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": json.dumps({
                "scores": [
                    {"id": "a", "score": 2.5},   # clamp to 1.0
                    {"id": "b", "score": -0.7},  # clamp to 0.0
                ],
            })}}]}

    class FakeClient:
        async def post(self, *_a: Any, **_kw: Any) -> FakeResponse:
            return FakeResponse()

        async def aclose(self) -> None:
            pass

    with patch.object(refl, "_ensure_client",
                      new=AsyncMock(return_value=FakeClient())):
        scores = await refl._score("q", [_hit("a", 0.5), _hit("b", 0.5)])
    assert scores == {"a": 1.0, "b": 0.0}


@pytest.mark.asyncio
async def test_reflector_handles_empty_input() -> None:
    refl = LMStudioReflector(base_url="http://x", model="qwen")
    out = await refl.reflect("q", [])
    assert out == []
