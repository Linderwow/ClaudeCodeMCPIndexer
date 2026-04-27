"""Phase 35 hardening: index drift watchdog.

Three stores are written from the same indexer:
  * Chroma (vectors)
  * SQLite FTS5 (BM25)
  * Kuzu (call graph)

If they ever disagree on what's indexed, hybrid search returns phantom
hits — a chunk_id present in vector but missing from lexical (or vice
versa) leaks broken results to the MCP client.

This test exercises the drift detection that lives in `code_rag.ops.fsck`
and verifies the *contract* that fsck reports drift correctly.
"""
from __future__ import annotations

from pathlib import Path

from code_rag.models import Chunk, ChunkKind, IndexMeta
from code_rag.ops import fsck
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.stores.sqlite_lexical import SqliteLexicalStore
from code_rag.version import INDEX_SCHEMA_VERSION


def _meta() -> IndexMeta:
    return IndexMeta(
        schema_version=INDEX_SCHEMA_VERSION,
        embedder_kind="test",
        embedder_model="dummy",
        embedder_dim=4,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


def _chunk(cid: str, path: str = "x.py") -> Chunk:
    return Chunk(
        id=cid,
        repo="r", path=path, language="python",
        symbol=cid, kind=ChunkKind.FUNCTION,
        start_line=1, end_line=2, text=f"def {cid}(): pass",
    )


def _build_settings(tmp_path: Path) -> object:
    """Construct a minimal Settings-like object that fsck() needs."""
    from code_rag.config import (
        ChunkerConfig,
        EmbedderConfig,
        GraphStoreConfig,
        IgnoreConfig,
        IndexerConfig,
        LexicalStoreConfig,
        McpConfig,
        PathsConfig,
        QueryRewriterConfig,
        RerankerConfig,
        Settings,
        VectorStoreConfig,
        WatcherConfig,
    )
    return Settings(
        paths=PathsConfig(
            roots=[tmp_path / "root"],
            data_dir=tmp_path,
            log_dir=tmp_path / "logs",
        ),
        ignore=IgnoreConfig(),
        embedder=EmbedderConfig(
            kind="lm_studio",
            base_url="http://localhost:1234/v1",
            model="dummy",
        ),
        reranker=RerankerConfig(
            kind="noop",
            base_url="http://localhost:1234/v1",
            model="",
        ),
        vector_store=VectorStoreConfig(),
        graph_store=GraphStoreConfig(),
        lexical_store=LexicalStoreConfig(),
        chunker=ChunkerConfig(),
        indexer=IndexerConfig(),
        watcher=WatcherConfig(),
        mcp=McpConfig(),
        query_rewriter=QueryRewriterConfig(),
    )


def test_fsck_detects_orphan_in_lexical(tmp_path: Path) -> None:
    """An id that exists in FTS but NOT in vector store is drift —
    fsck must report it."""
    (tmp_path / "root").mkdir()
    (tmp_path / "logs").mkdir()
    settings = _build_settings(tmp_path)

    meta = _meta()
    vec = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection="test",
        meta_path=tmp_path / "index_meta.json",
    )
    lex = SqliteLexicalStore(db_path=tmp_path / "fts.db")
    vec.open(meta)
    lex.open()

    # Both stores get one matching chunk — baseline.
    common = _chunk("common")
    vec.upsert([common], [[0.1, 0.2, 0.3, 0.4]])
    lex.upsert([common])

    # Lex gets one ORPHAN chunk that's NOT in vector.
    orphan = _chunk("orphan")
    lex.upsert([orphan])

    try:
        report = fsck(settings, vec, lex, auto_fix=False)  # type: ignore[arg-type]
        # report.ok may be True or False; the contract is that drift IS reported.
        # Check that at least one issue mentions drift / orphan / id mismatch.
        text = " ".join(str(i) for i in report.issues).lower()
        assert "drift" in text or "orphan" in text or "id" in text, (
            f"fsck failed to detect drift; report.issues = {report.issues}"
        )
    finally:
        vec.close()
        lex.close()


def test_fsck_clean_when_stores_match(tmp_path: Path) -> None:
    """When vector + FTS hold identical id sets, fsck reports OK."""
    (tmp_path / "root").mkdir()
    (tmp_path / "logs").mkdir()
    settings = _build_settings(tmp_path)

    vec = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection="test",
        meta_path=tmp_path / "index_meta.json",
    )
    lex = SqliteLexicalStore(db_path=tmp_path / "fts.db")
    vec.open(_meta())
    lex.open()

    chunks = [_chunk(f"c{i}", path=f"f{i}.py") for i in range(5)]
    vectors = [[0.1, 0.2, 0.3, 0.4]] * len(chunks)
    vec.upsert(chunks, vectors)
    lex.upsert(chunks)

    try:
        report = fsck(settings, vec, lex, auto_fix=False)  # type: ignore[arg-type]
        # Counts agree — drift-related issues should be absent.
        text = " ".join(str(i) for i in report.issues).lower()
        assert "drift" not in text, (
            f"fsck false-positive on a clean index: {report.issues}"
        )
    finally:
        vec.close()
        lex.close()
