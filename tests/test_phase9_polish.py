"""Phase 9 polish: match_reason, confidence threshold, smart truncation,
neighborhood attach, get_chunk_text MCP tool."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from code_rag.config import load_settings
from code_rag.embedders.fake import FakeEmbedder
from code_rag.graph.ingest import GraphIngester
from code_rag.indexing.indexer import Indexer
from code_rag.models import Chunk, ChunkKind, SearchHit
from code_rag.rerankers.noop import NoopReranker
from code_rag.retrieval.fusion import reciprocal_rank_fusion
from code_rag.retrieval.search import HybridSearcher, SearchParams
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.stores.kuzu_graph import KuzuGraphStore
from code_rag.stores.sqlite_lexical import SqliteLexicalStore

# ---- fixtures ---------------------------------------------------------------


@pytest.fixture()
def searcher_with_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Open all four stores + indexer + searcher against a small corpus."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text(
        """\
def helper_fn_UNIQ():
    return 1

class Bar:
    def baz(self):
        helper_fn_UNIQ()
        return 2
""",
        encoding="utf-8",
    )
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""
[paths]
roots    = ["{root.as_posix()}"]
data_dir = "{(tmp_path / 'data').as_posix()}"
log_dir  = "{(tmp_path / 'logs').as_posix()}"

[embedder]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "qwen3-embedding-4b"

[reranker]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "qwen3-reranker-4b"

[chunker]
min_chars = 5
max_chars = 2400
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))
    settings = load_settings()

    embedder = FakeEmbedder(dim=32)
    vec = ChromaVectorStore(settings.chroma_dir, settings.vector_store.collection,
                            settings.index_meta_path)
    lex = SqliteLexicalStore(settings.fts_path)
    graph = KuzuGraphStore(settings.kuzu_dir)
    meta = ChromaVectorStore.build_meta("fake", embedder.model, embedder.dim)
    vec.open(meta)
    lex.open()
    graph.open()
    indexer = Indexer(settings, embedder, vec,
                      lexical_store=lex, graph_store=GraphIngester(graph))
    asyncio.run(indexer.reindex_all())
    searcher = HybridSearcher(embedder, vec, lex, NoopReranker(), graph_store=graph)

    yield searcher, lex, graph

    vec.close()
    lex.close()
    graph.close()


# ---- match_reason ----------------------------------------------------------


def test_match_reason_lexical_has_tokens(tmp_path: Path) -> None:
    lex = SqliteLexicalStore(tmp_path / "fts.db")
    lex.open()
    try:
        lex.upsert([
            Chunk(id="a", repo="r", path="a.py", language="python",
                  symbol="MNQAlpha_V91_calc", kind=ChunkKind.FUNCTION,
                  start_line=1, end_line=1, text="def MNQAlpha_V91_calc(): pass"),
        ])
        hits = lex.query("MNQAlpha_V91", k=5)
        assert hits
        r = hits[0].match_reason or ""
        assert r.startswith("lexical bm25")
        assert "MNQAlpha_V91" in r
    finally:
        lex.close()


def test_match_reason_fusion_breadcrumb_has_sources() -> None:
    """RRF must stamp match_reason with the source+rank breadcrumb."""
    vec = SearchHit(
        chunk=Chunk(id="x", repo="r", path="x.py", language="python",
                    symbol="x", kind=ChunkKind.FUNCTION,
                    start_line=1, end_line=1, text=""),
        score=0.9, source="vector", match_reason="vector cosine 0.9",
    )
    lex = SearchHit(
        chunk=Chunk(id="x", repo="r", path="x.py", language="python",
                    symbol="x", kind=ChunkKind.FUNCTION,
                    start_line=1, end_line=1, text=""),
        score=0.7, source="lexical", match_reason="lexical bm25 matched tokens ['x']",
    )
    fused = reciprocal_rank_fusion([[vec], [lex]], k=60, top_k=5)
    assert fused
    r = fused[0].match_reason or ""
    assert r.startswith("hybrid rrf from")
    assert "vector" in r and "lexical" in r


# ---- confidence threshold --------------------------------------------------


def test_min_score_drops_noise_and_flags_no_match(searcher_with_graph) -> None:
    searcher, _lex, _graph = searcher_with_graph
    # Impossibly high threshold => everything drops => no_confident_match=True.
    params = SearchParams(k_final=5, min_score=9.99)
    resp = asyncio.run(searcher.search_full("helper_fn_UNIQ", params))
    assert resp.hits == []
    assert resp.no_confident_match is True


def test_min_score_zero_keeps_everything(searcher_with_graph) -> None:
    searcher, _lex, _graph = searcher_with_graph
    params = SearchParams(k_final=5, min_score=0.0)
    resp = asyncio.run(searcher.search_full("helper_fn_UNIQ", params))
    assert resp.hits
    assert resp.no_confident_match is False


# ---- smart truncation ------------------------------------------------------


def test_max_chars_per_hit_truncates_big_chunks(searcher_with_graph) -> None:
    searcher, _lex, _graph = searcher_with_graph
    # The tiny test corpus has short chunks; seed a big synthetic one via lexical.
    big_text = "class Big:\n    " + ("x = 1\n    " * 600)
    assert len(big_text) > 3000
    _lex.upsert([
        Chunk(id="big1", repo="r", path="big.py", language="python",
              symbol="Big", kind=ChunkKind.CLASS,
              start_line=1, end_line=600, text=big_text),
    ])
    params = SearchParams(k_final=5, max_chars_per_hit=500)
    resp = asyncio.run(searcher.search_full("Big", params))
    assert resp.hits
    matched = [h for h in resp.hits if h.chunk.path == "big.py"]
    assert matched, "seeded chunk must come back"
    assert len(matched[0].chunk.text) <= 550  # marker adds a few chars
    assert "truncated" in matched[0].chunk.text


def test_truncation_disabled_by_default(searcher_with_graph) -> None:
    searcher, _lex, _graph = searcher_with_graph
    big_text = "def huge():\n" + ("    pass\n" * 800)
    _lex.upsert([
        Chunk(id="b2", repo="r", path="huge.py", language="python",
              symbol="huge", kind=ChunkKind.FUNCTION,
              start_line=1, end_line=800, text=big_text),
    ])
    params = SearchParams(k_final=5)  # max_chars_per_hit=0 default
    resp = asyncio.run(searcher.search_full("huge", params))
    matched = [h for h in resp.hits if h.chunk.path == "huge.py"]
    assert matched
    assert len(matched[0].chunk.text) == len(big_text), "no truncation by default"


# ---- neighborhood attach ---------------------------------------------------


def test_attach_neighbors_populates_callers_and_callees(searcher_with_graph) -> None:
    searcher, _lex, _graph = searcher_with_graph
    params = SearchParams(k_final=5, attach_neighbors=True)
    resp = asyncio.run(searcher.search_full("helper_fn_UNIQ", params))
    assert resp.hits
    # Every hit should have a neighborhood entry, even if empty.
    for h in resp.hits:
        assert h.chunk.id in resp.neighborhood
    # The `helper_fn_UNIQ` hit must have Bar.baz as a caller.
    hit_for_helper = next(
        (h for h in resp.hits if h.chunk.symbol and "helper_fn_UNIQ" in h.chunk.symbol),
        None,
    )
    if hit_for_helper:  # may or may not be top-1 under FakeEmbedder
        nb = resp.neighborhood[hit_for_helper.chunk.id]
        assert any("baz" in c.symbol for c in nb["callers"]), nb


# ---- MCP get_chunk_text ----------------------------------------------------


def test_mcp_get_chunk_text_fetches_full_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from code_rag.mcp_server.server import ServerResources, _tool_get_chunk_text

    root = tmp_path / "repo"
    (root / "a.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""
[paths]
roots    = ["{root.as_posix()}"]
data_dir = "{(tmp_path / 'data').as_posix()}"
log_dir  = "{(tmp_path / 'logs').as_posix()}"

[embedder]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "qwen3-embedding-4b"

[reranker]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "qwen3-reranker-4b"

[chunker]
min_chars = 5
max_chars = 2400
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))
    settings = load_settings()
    res = ServerResources(settings)
    res.embedder = FakeEmbedder(dim=32)
    res.vec = ChromaVectorStore(settings.chroma_dir, settings.vector_store.collection,
                                settings.index_meta_path)
    res.lex = SqliteLexicalStore(settings.fts_path)
    res.graph = KuzuGraphStore(settings.kuzu_dir)
    res.reranker = NoopReranker()
    meta = ChromaVectorStore.build_meta("fake", res.embedder.model, res.embedder.dim)
    res.vec.open(meta)
    res.lex.open()
    res.graph.open()
    res.indexer = Indexer(settings, res.embedder, res.vec,
                          lexical_store=res.lex, graph_store=GraphIngester(res.graph))
    res.searcher = HybridSearcher(res.embedder, res.vec, res.lex, res.reranker,
                                  graph_store=res.graph)
    asyncio.run(res.indexer.reindex_all())

    try:
        # Known-text query so the FakeEmbedder returns the same vector.
        q_vecs = asyncio.run(res.embedder.embed(["seed"]))
        some_hit = res.vec.query(q_vecs[0], k=1)[0]
        full = asyncio.run(_tool_get_chunk_text(res, {"chunk_id": some_hit.chunk.id}))
        assert full["chunk_id"] == some_hit.chunk.id
        assert full["text"] == some_hit.chunk.text
        assert full["path"] == "a.py"

        # Missing id -> error.
        miss = asyncio.run(_tool_get_chunk_text(res, {"chunk_id": "does-not-exist"}))
        assert "error" in miss
    finally:
        asyncio.run(res.close())
