"""End-to-end pipeline tests with FakeEmbedder + persistent Chroma in tmp_path."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from code_rag.config import load_settings
from code_rag.embedders.fake import FakeEmbedder
from code_rag.indexing.indexer import Indexer
from code_rag.stores.chroma_vector import ChromaVectorStore


def _make_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text(
        "def foo():\n    return 1\n\nclass Bar:\n    def baz(self):\n        return 2\n",
        encoding="utf-8",
    )
    (root / "pkg" / "b.cs").write_text(
        "namespace N { public class K { public void M() {} } }\n",
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
min_chars     = 10
max_chars     = 2400
overlap_chars = 0
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))


def test_reindex_all_writes_meta_and_upserts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_settings(tmp_path, monkeypatch)
    settings = load_settings()
    embedder = FakeEmbedder(dim=32)
    store = ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection=settings.vector_store.collection,
        meta_path=settings.index_meta_path,
    )
    meta = ChromaVectorStore.build_meta("fake", embedder.model, embedder.dim)
    store.open(meta)
    try:
        indexer = Indexer(settings, embedder, store)
        stats = asyncio.run(indexer.reindex_all())
        assert stats.chunks_upserted > 0
        assert stats.files_indexed >= 2
        assert store.count() == stats.chunks_upserted
    finally:
        store.close()
    assert settings.index_meta_path.exists()


def test_reindex_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_settings(tmp_path, monkeypatch)
    settings = load_settings()
    embedder = FakeEmbedder(dim=32)
    store = ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection=settings.vector_store.collection,
        meta_path=settings.index_meta_path,
    )
    meta = ChromaVectorStore.build_meta("fake", embedder.model, embedder.dim)
    store.open(meta)
    try:
        indexer = Indexer(settings, embedder, store)
        a = asyncio.run(indexer.reindex_all())
        b = asyncio.run(indexer.reindex_all())
        # Same file set, same content → same count after the second pass.
        assert store.count() == a.chunks_upserted == b.chunks_upserted
    finally:
        store.close()


def test_meta_mismatch_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_settings(tmp_path, monkeypatch)
    settings = load_settings()
    embedder = FakeEmbedder(dim=32)
    store = ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection=settings.vector_store.collection,
        meta_path=settings.index_meta_path,
    )
    meta = ChromaVectorStore.build_meta("fake", embedder.model, embedder.dim)
    store.open(meta)
    store.close()

    # Reopen with a DIFFERENT embedder model — must raise.
    bad = ChromaVectorStore.build_meta("fake", "different-model", embedder.dim)
    store2 = ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection=settings.vector_store.collection,
        meta_path=settings.index_meta_path,
    )
    with pytest.raises(RuntimeError, match="Index metadata mismatch"):
        store2.open(bad)


def test_reindex_all_gcs_paths_for_deleted_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a source file is deleted while the watcher is offline, the next
    `reindex_all` must purge its chunks via the stale-path GC pass.
    Without GC, search would return ghost hits forever."""
    from code_rag.factory import build_lexical_store
    _make_settings(tmp_path, monkeypatch)
    settings = load_settings()
    embedder = FakeEmbedder(dim=32)
    vec = ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection=settings.vector_store.collection,
        meta_path=settings.index_meta_path,
    )
    lex = build_lexical_store(settings)
    meta = ChromaVectorStore.build_meta("fake", embedder.model, embedder.dim)
    vec.open(meta)
    lex.open()
    try:
        indexer = Indexer(settings, embedder, vec, lexical_store=lex)
        asyncio.run(indexer.reindex_all())
        before_paths = lex.list_paths()
        assert len(before_paths) >= 2  # a.py and b.cs

        # Now simulate watcher-offline file deletion: physically remove a file
        # from disk WITHOUT going through indexer.remove_path.
        (tmp_path / "repo" / "pkg" / "a.py").unlink()

        # Naive reindex would leave a.py's chunks in stores forever. Our GC
        # pass must catch it.
        stats = asyncio.run(indexer.reindex_all())
        assert stats.paths_gc >= 1, f"expected GC to remove orphan, stats={stats.as_dict()}"
        after_paths = lex.list_paths()
        assert "pkg/a.py" not in after_paths
        # b.cs (still on disk) survived.
        assert any(p.endswith("b.cs") for p in after_paths)
    finally:
        vec.close()
        lex.close()


def test_query_returns_relevant_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With FakeEmbedder (hash-based, deterministic) the exact-text query must
    return the identical chunk first (distance = 0)."""
    _make_settings(tmp_path, monkeypatch)
    settings = load_settings()
    embedder = FakeEmbedder(dim=32)
    store = ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection=settings.vector_store.collection,
        meta_path=settings.index_meta_path,
    )
    meta = ChromaVectorStore.build_meta("fake", embedder.model, embedder.dim)
    store.open(meta)
    try:
        indexer = Indexer(settings, embedder, store)
        asyncio.run(indexer.reindex_all())

        # Grab one stored chunk's text and query by it — must be top hit.
        all_hits = store.query(asyncio.run(embedder.embed(["seed"]))[0], k=50)
        assert all_hits
        target = next(h for h in all_hits if h.chunk.path.endswith("a.py"))

        async def q() -> list:
            vecs = await embedder.embed([target.chunk.text])
            return store.query(vecs[0], k=3)

        hits = asyncio.run(q())
        assert hits[0].chunk.id == target.chunk.id
        assert hits[0].score == pytest.approx(1.0, abs=1e-3)
    finally:
        store.close()
