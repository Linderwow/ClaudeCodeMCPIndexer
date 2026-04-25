"""Phase 21: graph-augmented retrieval boosts hits within the radius of an
anchor symbol."""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from code_rag.interfaces.graph_store import SymbolRef
from code_rag.models import Chunk, ChunkKind, SearchHit
from code_rag.retrieval.graph_augmented import GraphAugmentedSearcher
from code_rag.retrieval.search import SearchParams, SearchResponse

# ---- minimal fakes --------------------------------------------------------


def _hit(path: str, symbol: str | None, score: float) -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id=f"{path}:{symbol or ''}", repo="r", path=path, language="python",
            symbol=symbol, kind=ChunkKind.FUNCTION,
            start_line=1, end_line=1, text="",
        ),
        score=score, source="hybrid",
    )


@dataclass
class _FakeHybrid:
    response_hits: Sequence[SearchHit]

    async def search_full(self, query: str, params: SearchParams) -> SearchResponse:
        _ = (query, params)
        return SearchResponse(
            hits=list(self.response_hits),
            no_confident_match=False, elapsed_ms=1.0,
        )


class _FakeGraph:
    def __init__(self, edges: dict[str, list[SymbolRef]],
                 symbols: dict[str, list[SymbolRef]]) -> None:
        self._edges = edges
        self._symbols = symbols

    def find_symbol(self, symbol: str, path: str | None = None) -> list[SymbolRef]:
        out = self._symbols.get(symbol, [])
        if path is not None:
            out = [s for s in out if s.path == path]
        return out

    def callers_of(self, symbol: str, path: str | None = None) -> list[SymbolRef]:
        return list(self._edges.get(f"callers:{symbol}", []))

    def callees_of(self, symbol: str, path: str | None = None) -> list[SymbolRef]:
        return list(self._edges.get(f"callees:{symbol}", []))


# ---- behavior tests -------------------------------------------------------


@pytest.mark.asyncio
async def test_no_anchor_returns_plain_search() -> None:
    """No anchor_symbol → plain hybrid search, no graph involvement."""
    hybrid = _FakeHybrid([_hit("a.py", "f", 0.6), _hit("b.py", "g", 0.4)])
    aug = GraphAugmentedSearcher(hybrid, graph=None)  # type: ignore[arg-type]
    res = await aug.search("anything", SearchParams(), anchor_symbol=None)
    assert [h.chunk.path for h in res.response.hits] == ["a.py", "b.py"]
    assert res.related_set_size == 0


@pytest.mark.asyncio
async def test_unresolvable_anchor_returns_plain_search() -> None:
    hybrid = _FakeHybrid([_hit("a.py", "f", 0.6)])
    graph = _FakeGraph(edges={}, symbols={})  # `bar` won't resolve
    aug = GraphAugmentedSearcher(hybrid, graph=graph)  # type: ignore[arg-type]
    res = await aug.search("q", SearchParams(), anchor_symbol="bar")
    assert res.related_set_size == 0
    assert [h.chunk.path for h in res.response.hits] == ["a.py"]


@pytest.mark.asyncio
async def test_anchor_boosts_in_radius_hits_above_unrelated() -> None:
    """Anchor `OnBarUpdate` calls `helper`. Plain search ranks an unrelated
    `unrelated` chunk above `helper`; with anchor + boost, `helper` should
    win."""
    hybrid = _FakeHybrid([
        _hit("file_a.py", "unrelated", 0.50),  # unrelated to anchor
        _hit("file_b.py", "helper",    0.40),  # in anchor's radius
    ])
    anchor = SymbolRef(path="file_main.py", symbol="OnBarUpdate", kind="function",
                       start_line=1, end_line=1)
    callee = SymbolRef(path="file_b.py", symbol="helper", kind="function",
                       start_line=1, end_line=1)
    graph = _FakeGraph(
        symbols={"OnBarUpdate": [anchor]},
        edges={
            "callees:OnBarUpdate": [callee],
            "callers:OnBarUpdate": [],
            "callees:helper": [],
            "callers:helper": [anchor],
        },
    )
    aug = GraphAugmentedSearcher(hybrid, graph=graph)  # type: ignore[arg-type]
    res = await aug.search("q", SearchParams(), anchor_symbol="OnBarUpdate", hops=1)
    paths = [h.chunk.path for h in res.response.hits]
    assert paths[0] == "file_b.py", \
        f"helper should be boosted above unrelated; got order {paths}"
    assert res.related_set_size >= 2  # anchor itself + at least the callee
    # Reason string should mention the graph distance / boost factor.
    assert "graph" in (res.response.hits[0].match_reason or "").lower()


@pytest.mark.asyncio
async def test_anchor_unrelated_to_all_hits_doesnt_break() -> None:
    """If hits and anchor share no graph proximity, plain ordering is preserved."""
    hybrid = _FakeHybrid([_hit("x.py", "a", 0.7), _hit("y.py", "b", 0.5)])
    anchor = SymbolRef(path="z.py", symbol="anchor", kind="function",
                       start_line=1, end_line=1)
    graph = _FakeGraph(
        symbols={"anchor": [anchor]},
        edges={"callees:anchor": [], "callers:anchor": []},
    )
    aug = GraphAugmentedSearcher(hybrid, graph=graph)  # type: ignore[arg-type]
    res = await aug.search("q", SearchParams(), anchor_symbol="anchor", hops=2)
    assert [h.chunk.path for h in res.response.hits] == ["x.py", "y.py"]


@pytest.mark.asyncio
async def test_path_level_match_is_weaker_than_symbol_level() -> None:
    """A hit in the SAME FILE as a related symbol still gets a boost (file-
    level proximity), but smaller than an exact (path, symbol) match."""
    hybrid = _FakeHybrid([
        _hit("a.py", "exact",      0.40),  # exact (path, symbol) match in radius
        _hit("a.py", "near",       0.40),  # same path as a related symbol
        _hit("c.py", "unrelated",  0.40),
    ])
    anchor = SymbolRef(path="a.py", symbol="exact", kind="function",
                       start_line=1, end_line=1)
    graph = _FakeGraph(
        symbols={"exact": [anchor]},
        edges={"callees:exact": [], "callers:exact": []},
    )
    aug = GraphAugmentedSearcher(hybrid, graph=graph)  # type: ignore[arg-type]
    res = await aug.search("q", SearchParams(), anchor_symbol="exact", hops=1)
    paths = [(h.chunk.path, h.chunk.symbol) for h in res.response.hits]
    # exact > near > unrelated
    assert paths[0] == ("a.py", "exact")
    assert paths[1] == ("a.py", "near")
    assert paths[2] == ("c.py", "unrelated")


def test_async_dispatch_smoke() -> None:
    """Just confirm the asyncio.run wrapper executes cleanly for all of the
    above; pytest-asyncio handles individual cases but a synchronous smoke
    keeps stdlib-only callers happy."""
    hybrid = _FakeHybrid([_hit("a.py", "f", 0.5)])
    aug = GraphAugmentedSearcher(hybrid, graph=None)  # type: ignore[arg-type]
    res = asyncio.run(aug.search("q", SearchParams()))
    assert res.response.hits
