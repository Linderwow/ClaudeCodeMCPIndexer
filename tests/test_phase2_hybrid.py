"""Phase 2: lexical store, RRF fusion, hybrid searcher end-to-end (offline)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from code_rag.config import load_settings
from code_rag.embedders.fake import FakeEmbedder
from code_rag.indexing.indexer import Indexer
from code_rag.models import Chunk, ChunkKind, SearchHit
from code_rag.rerankers.noop import NoopReranker
from code_rag.retrieval.fusion import reciprocal_rank_fusion
from code_rag.retrieval.search import HybridSearcher, SearchParams
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.stores.sqlite_lexical import SqliteLexicalStore


def _make_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "alpha.py").write_text(
        '''\
def MNQAlpha_V91_compute():
    """Rare identifier worth finding."""
    return 42

class OrderBook:
    def add_bid(self, px: float): ...
    def add_ask(self, px: float): ...
''',
        encoding="utf-8",
    )
    (root / "pkg" / "beta.cs").write_text(
        """\
namespace N {
    public class Strategy {
        public void OnBarUpdate() { /* rare sentinel OnBarUpdate_V91 */ }
        public void SomeOtherMethod() { }
    }
}
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


# ---- RRF unit test ----------------------------------------------------------


def _mk_hit(cid: str, path: str = "x", source: str = "vector") -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id=cid, repo="r", path=path, language="python", symbol=cid,
            kind=ChunkKind.FUNCTION, start_line=1, end_line=1, text="x",
        ),
        score=0.0,
        source=source,  # type: ignore[arg-type]
    )


def test_rrf_fuses_two_lists() -> None:
    # list_a: [1, 2, 3]; list_b: [2, 3, 4]. Doc 2 appears high in both -> should win.
    list_a = [_mk_hit("1"), _mk_hit("2"), _mk_hit("3")]
    list_b = [_mk_hit("2"), _mk_hit("3"), _mk_hit("4")]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60, top_k=10)
    ids = [h.chunk.id for h in fused]
    assert ids[0] == "2", f"doc 2 should dominate, got order {ids}"
    assert set(ids) == {"1", "2", "3", "4"}
    assert all(h.source == "hybrid" for h in fused)


def test_rrf_top_k_truncates() -> None:
    big = [_mk_hit(str(i)) for i in range(20)]
    fused = reciprocal_rank_fusion([big], k=60, top_k=5)
    assert len(fused) == 5


# ---- SQLite FTS5 --------------------------------------------------------


def test_fts5_rare_identifier_recall(tmp_path: Path) -> None:
    lex = SqliteLexicalStore(db_path=tmp_path / "fts.db")
    lex.open()
    try:
        chunks = [
            Chunk(id="a", repo="r", path="a.py", language="python",
                  symbol="MNQAlpha_V91_compute", kind=ChunkKind.FUNCTION,
                  start_line=1, end_line=3, text="def MNQAlpha_V91_compute(): return 42"),
            Chunk(id="b", repo="r", path="b.py", language="python",
                  symbol="other", kind=ChunkKind.FUNCTION,
                  start_line=1, end_line=3, text="def other(): return 0"),
        ]
        lex.upsert(chunks)
        hits = lex.query("MNQAlpha_V91", k=5)
        assert hits, "rare identifier must be found by FTS5"
        assert hits[0].chunk.id == "a"
    finally:
        lex.close()


def test_fts5_delete_by_path_removes_from_fts_too(tmp_path: Path) -> None:
    lex = SqliteLexicalStore(db_path=tmp_path / "fts.db")
    lex.open()
    try:
        lex.upsert([
            Chunk(id="a", repo="r", path="a.py", language="python", symbol="foo",
                  kind=ChunkKind.FUNCTION, start_line=1, end_line=1, text="xyzzy"),
        ])
        assert lex.query("xyzzy", k=5), "pre-delete"
        lex.delete_by_path("a.py")
        assert not lex.query("xyzzy", k=5), "post-delete"
        assert lex.count() == 0
    finally:
        lex.close()


# ---- end-to-end hybrid --------------------------------------------------


def test_hybrid_search_rare_identifier_wins_via_lexical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defining test for hybrid: a rare identifier (`MNQAlpha_V91_compute`)
    is NOT found by a hash-based fake embedder, but lexical BM25 nails it."""
    _make_settings(tmp_path, monkeypatch)
    settings = load_settings()

    embedder = FakeEmbedder(dim=32)
    vec = ChromaVectorStore(settings.chroma_dir, settings.vector_store.collection,
                            settings.index_meta_path)
    lex = SqliteLexicalStore(settings.fts_path)
    meta = ChromaVectorStore.build_meta("fake", embedder.model, embedder.dim)
    vec.open(meta)
    lex.open()
    try:
        indexer = Indexer(settings, embedder, vec, lexical_store=lex)
        asyncio.run(indexer.reindex_all())

        searcher = HybridSearcher(embedder, vec, lex, NoopReranker())
        hits = asyncio.run(searcher.search(
            "MNQAlpha_V91_compute",
            SearchParams(k_final=5, k_vector=20, k_lexical=20, k_rerank_in=20),
        ))
        assert hits, "expected at least one hit"
        # The rare-identifier chunk must be top-1 — lexical's rank-1 wins RRF
        # when the vector list is essentially random.
        assert hits[0].chunk.symbol and "MNQAlpha_V91_compute" in hits[0].chunk.symbol
    finally:
        vec.close()
        lex.close()


def test_hybrid_language_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_settings(tmp_path, monkeypatch)
    settings = load_settings()
    embedder = FakeEmbedder(dim=32)
    vec = ChromaVectorStore(settings.chroma_dir, settings.vector_store.collection,
                            settings.index_meta_path)
    lex = SqliteLexicalStore(settings.fts_path)
    meta = ChromaVectorStore.build_meta("fake", embedder.model, embedder.dim)
    vec.open(meta)
    lex.open()
    try:
        indexer = Indexer(settings, embedder, vec, lexical_store=lex)
        asyncio.run(indexer.reindex_all())
        searcher = HybridSearcher(embedder, vec, lex, NoopReranker())
        hits = asyncio.run(searcher.search(
            "method",
            SearchParams(k_final=10, language="python"),
        ))
        assert all(h.chunk.language == "python" for h in hits)
    finally:
        vec.close()
        lex.close()
