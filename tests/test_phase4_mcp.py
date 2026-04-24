"""Phase 4: MCP tool handlers, tested in-process without stdio.

We exercise the handler functions directly against a ServerResources with a
FakeEmbedder and NoopReranker — no LM Studio needed.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from code_rag.config import load_settings
from code_rag.embedders.fake import FakeEmbedder
from code_rag.graph.ingest import GraphIngester
from code_rag.indexing.indexer import Indexer
from code_rag.mcp_server.server import (
    ServerResources,
    _tool_get_callees,
    _tool_get_callers,
    _tool_get_file_context,
    _tool_get_symbol,
    _tool_index_stats,
    _tool_search_code,
)
from code_rag.rerankers.noop import NoopReranker
from code_rag.retrieval.search import HybridSearcher
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.stores.kuzu_graph import KuzuGraphStore
from code_rag.stores.sqlite_lexical import SqliteLexicalStore


@pytest.fixture()
def opened_res(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A ServerResources bound to FakeEmbedder + NoopReranker, fully opened,
    with a small sample corpus indexed. Yields the opened resources and cleans up."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text(
        """\
def helper():
    return 1

class Bar:
    def baz(self):
        helper()
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
min_chars = 10
max_chars = 2400
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))
    settings = load_settings()

    res = ServerResources(settings)
    # Swap in the offline implementations — we don't want to hit LM Studio.
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

    res.indexer = Indexer(
        settings, res.embedder, res.vec,
        lexical_store=res.lex,
        graph_store=GraphIngester(res.graph),
    )
    res.searcher = HybridSearcher(res.embedder, res.vec, res.lex, res.reranker)

    asyncio.run(res.indexer.reindex_all())

    yield res

    asyncio.run(res.close())


def test_search_code_returns_hits(opened_res: ServerResources) -> None:
    out = asyncio.run(_tool_search_code(opened_res, {"query": "helper", "k": 5}))
    assert out.get("hits"), out
    # Each hit carries enough to jump to the location.
    for h in out["hits"]:
        assert "path" in h and "start_line" in h and "end_line" in h
        assert "text" in h


def test_get_symbol_finds_known(opened_res: ServerResources) -> None:
    out = asyncio.run(_tool_get_symbol(opened_res, {"name": "helper"}))
    assert out["symbols"], out
    assert any(s["symbol"].endswith("helper") for s in out["symbols"])


def test_get_callers_matches_graph(opened_res: ServerResources) -> None:
    out = asyncio.run(_tool_get_callers(opened_res, {"symbol": "helper"}))
    assert out["callers"], out
    assert any("baz" in c["symbol"] for c in out["callers"])


def test_get_callees_matches_graph(opened_res: ServerResources) -> None:
    out = asyncio.run(_tool_get_callees(opened_res, {"symbol": "Bar.baz", "path": "pkg/a.py"}))
    assert any(c["symbol"] == "helper" for c in out["callees"])


def test_get_file_context_returns_symbols_with_neighbors(opened_res: ServerResources) -> None:
    out = asyncio.run(_tool_get_file_context(opened_res, {"path": "pkg/a.py"}))
    assert out["path"] == "pkg/a.py"
    syms = out["symbols"]
    assert syms, out
    for s in syms:
        assert "callers" in s and "callees" in s


def test_index_stats_has_counts(opened_res: ServerResources) -> None:
    out = asyncio.run(_tool_index_stats(opened_res, {}))
    assert out["vector_count"] > 0
    assert out["lexical_count"] > 0
    assert out["meta"] is not None
