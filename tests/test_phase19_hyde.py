"""Phase 19: HyDE intent classifier + retriever plan."""
from __future__ import annotations

import pytest

from code_rag.retrieval.hyde import HydeRetrieverPlan, classify_intent

# ---- intent classifier ----------------------------------------------------


@pytest.mark.parametrize("q,expected", [
    ("OnBarUpdate",                                    "identifier"),
    ("RunGhostEvaluation",                             "identifier"),
    ("read_trade_gate",                                "identifier"),
    ("regime_predictor",                               "identifier"),
    ("foo",                                            "natural_language"),  # short single word, no signals
    ("find PositionSizer",                             "definition_lookup"),
    ("OnBarUpdate calls",                              "definition_lookup"),
    ("where does the strategy size positions",         "natural_language"),
    ("how is the noise floor stop calculated in MNQAlpha", "mixed"),
    ("when was the kelly sizing logic added",          "natural_language"),
])
def test_classify_intent(q: str, expected: str) -> None:
    assert classify_intent(q) == expected


# ---- HydeRetrieverPlan ----------------------------------------------------


class _RecordingGenerator:
    def __init__(self, response: str | None = "def hypothetical(): ...\n") -> None:
        self._response = response
        self.calls = 0

    async def generate(self, query: str, *, max_tokens: int = 256) -> str:
        self.calls += 1
        if self._response is None:
            return query  # simulate fallback to literal
        return self._response


@pytest.mark.asyncio
async def test_plan_skips_hyde_for_identifier_queries() -> None:
    gen = _RecordingGenerator()
    plan = HydeRetrieverPlan(generator=gen)
    result = await plan.plan("OnBarUpdate")
    assert result == [("OnBarUpdate", 1.0)]
    assert gen.calls == 0, "HyDE should NOT fire on identifier queries"


@pytest.mark.asyncio
async def test_plan_fires_hyde_for_natural_language() -> None:
    gen = _RecordingGenerator(response="def size_positions(account, edge):\n    return account * edge.kelly\n")
    plan = HydeRetrieverPlan(generator=gen)
    result = await plan.plan("where does the strategy size positions")
    assert len(result) == 2
    assert result[0][1] == 1.0
    assert result[1][1] == pytest.approx(0.7)
    assert "size_positions" in result[1][0]
    assert gen.calls == 1


@pytest.mark.asyncio
async def test_plan_fires_for_mixed_intent() -> None:
    gen = _RecordingGenerator(response="// hypothetical hot path\n")
    plan = HydeRetrieverPlan(generator=gen)
    result = await plan.plan("how is the noise floor stop calculated in MNQAlpha")
    assert len(result) == 2
    assert gen.calls == 1


@pytest.mark.asyncio
async def test_plan_falls_back_when_generator_returns_query_unchanged() -> None:
    """Generator gracefully returning the query (probe failed, model not
    loaded) must NOT add a redundant search arm — that would just double
    cost for zero new signal."""
    gen = _RecordingGenerator(response=None)  # will return the query as-is
    plan = HydeRetrieverPlan(generator=gen)
    result = await plan.plan("where is the helper")
    assert result == [("where is the helper", 1.0)]


@pytest.mark.asyncio
async def test_plan_with_no_generator_returns_literal_only() -> None:
    """Constructing HydeRetrieverPlan(generator=None) is the no-op fallback —
    same behavior as having HyDE turned off entirely. Useful for testing
    the rest of the pipeline without an LLM."""
    plan = HydeRetrieverPlan(generator=None)
    result = await plan.plan("where does the strategy size positions")
    assert result == [("where does the strategy size positions", 1.0)]


@pytest.mark.asyncio
async def test_plan_swallows_generator_exceptions() -> None:
    """If the generator raises (LM Studio crashed, network died), the plan
    must still produce SOMETHING — we degrade to literal-only retrieval."""

    class _ExplodingGenerator:
        async def generate(self, query: str, *, max_tokens: int = 256) -> str:
            raise RuntimeError("boom")

    plan = HydeRetrieverPlan(generator=_ExplodingGenerator())
    result = await plan.plan("how does session phase work")
    assert result == [("how does session phase work", 1.0)]
