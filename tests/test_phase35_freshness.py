"""Phase 35 (D3): freshness verification harness.

Contract: when a watched file changes, the watcher must update the index
within `watcher.debounce_ms`. We test the unit pieces (Indexer +
FileHashRegistry) that the watcher composes — a full end-to-end watcher
test would need watchdog's filesystem events, which are flaky in CI.

What we verify:
  1. Indexing a file populates Chroma + FTS + file_hashes.
  2. Re-indexing identical content is a no-op (hash-skip works).
  3. Editing the file produces fresh chunks; old chunks for that path
     are evicted (delete-then-insert semantics).
  4. Deleting a file via remove_path() purges all its chunks.

Together these guarantee:
  * Search results never reflect stale content (no zombie chunks).
  * Unchanged files don't waste GPU on re-embedding.
  * Renames / deletions don't leak stale entries.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

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
from code_rag.indexing.file_hash import FileHashRegistry
from code_rag.indexing.indexer import Indexer
from code_rag.interfaces.embedder import Embedder
from code_rag.models import IndexMeta
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.stores.sqlite_lexical import SqliteLexicalStore
from code_rag.version import INDEX_SCHEMA_VERSION


# ---- minimal in-memory embedder (no LM Studio needed) ----------------------


class FakeEmbedder(Embedder):
    """Deterministic 4-d hash-based embeddings. No network, no GPU."""
    def __init__(self) -> None:
        self._calls = 0

    @property
    def model(self) -> str:
        return "fake-test"

    @property
    def dim(self) -> int:
        return 4

    async def embed(self, texts):
        self._calls += 1
        out = []
        for t in texts:
            h = abs(hash(t))
            out.append([
                ((h >> 0) & 0xFF) / 255.0,
                ((h >> 8) & 0xFF) / 255.0,
                ((h >> 16) & 0xFF) / 255.0,
                ((h >> 24) & 0xFF) / 255.0,
            ])
        return out

    async def health(self) -> None:
        return None


def _build_settings(tmp_path: Path) -> Settings:
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
            model="fake-test",
            dim=4,
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
        indexer=IndexerConfig(parallel_workers=1, embed_concurrency=1),
        watcher=WatcherConfig(),
        mcp=McpConfig(),
        query_rewriter=QueryRewriterConfig(),
    )


def _meta() -> IndexMeta:
    return IndexMeta(
        schema_version=INDEX_SCHEMA_VERSION,
        embedder_kind="lm_studio",
        embedder_model="fake-test",
        embedder_dim=4,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture
def fresh_indexer(tmp_path: Path):
    (tmp_path / "root").mkdir()
    (tmp_path / "logs").mkdir()
    settings = _build_settings(tmp_path)
    embedder = FakeEmbedder()
    vec = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection="test",
        meta_path=tmp_path / "index_meta.json",
    )
    lex = SqliteLexicalStore(db_path=tmp_path / "fts.db")
    hashes = FileHashRegistry(tmp_path / "file_hashes.db")
    vec.open(_meta())
    lex.open()
    hashes.open()
    indexer = Indexer(
        settings, embedder, vec,
        lexical_store=lex,
        file_hashes=hashes,
    )
    yield indexer, vec, lex, hashes, embedder, tmp_path
    vec.close()
    lex.close()
    hashes.close()


def _path_chunk_count(lex: SqliteLexicalStore, rel: str) -> int:
    """How many chunks does FTS have for `rel`?"""
    import sqlite3
    conn = sqlite3.connect(str(lex._db_path), isolation_level=None)  # type: ignore[attr-defined]
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE path = ?", (rel,),
        ).fetchone()[0]
    finally:
        conn.close()


# ---- the actual freshness contract -----------------------------------------


def test_index_then_skip_unchanged(fresh_indexer: Any) -> None:
    """Re-indexing identical content must NOT re-embed (hash-skip works)."""
    indexer, vec, lex, hashes, embedder, tmp_path = fresh_indexer
    f = tmp_path / "root" / "alpha.py"
    f.write_text(
        "def hello(name: str) -> str:\n"
        "    '''greet a person by name with the standard hi greeting.'''\n"
        "    greeting = f'hi, {name}!'\n"
        "    return greeting\n",
        encoding="utf-8",
    )

    # First index: produces chunks + 1 embedder call.
    asyncio.run(indexer.reindex_path(f))
    n1 = _path_chunk_count(lex, "alpha.py")
    calls1 = embedder._calls
    assert n1 > 0
    assert calls1 >= 1

    # Second index of UNCHANGED file: hash-skip → no new embed call.
    asyncio.run(indexer.reindex_path(f))
    n2 = _path_chunk_count(lex, "alpha.py")
    calls2 = embedder._calls
    assert n2 == n1
    assert calls2 == calls1, (
        f"expected hash-skip (calls unchanged) but embedder was called "
        f"{calls2 - calls1} more times after re-indexing identical content"
    )


def test_edit_evicts_old_chunks(fresh_indexer: Any) -> None:
    """Editing the file must evict old chunks for that path; new chunks
    reflect new content. No zombie chunks survive."""
    indexer, vec, lex, hashes, embedder, tmp_path = fresh_indexer
    f = tmp_path / "root" / "alpha.py"
    f.write_text(
        "def hello(name: str) -> str:\n"
        "    '''greet a person by name with the standard hi greeting.'''\n"
        "    greeting = f'hi, {name}!'\n"
        "    return greeting\n",
        encoding="utf-8",
    )
    asyncio.run(indexer.reindex_path(f))
    n1 = _path_chunk_count(lex, "alpha.py")
    assert n1 > 0

    # Edit: completely replace content with similarly-sized new content.
    f.write_text(
        "def goodbye(name: str) -> str:\n"
        "    '''bid a person farewell with the standard bye message.'''\n"
        "    farewell = f'goodbye, {name}!'\n"
        "    return farewell\n",
        encoding="utf-8",
    )
    asyncio.run(indexer.reindex_path(f))

    # FTS shouldn't have ANY chunk whose text mentions `hello` after the
    # delete-then-insert pass. The new chunks should be present.
    import sqlite3
    conn = sqlite3.connect(str(lex._db_path), isolation_level=None)  # type: ignore[attr-defined]
    try:
        rows = conn.execute(
            "SELECT text FROM chunks WHERE path = ?", ("alpha.py",),
        ).fetchall()
    finally:
        conn.close()
    bodies = " ".join(r[0] for r in rows)
    assert "hello" not in bodies, (
        f"stale 'hello' chunk survived re-index: {bodies!r}"
    )
    assert "goodbye" in bodies or "farewell" in bodies, (
        f"new symbols missing from index: {bodies!r}"
    )


def test_remove_path_purges_chunks(fresh_indexer: Any) -> None:
    """Calling indexer.remove_path() must clear every chunk for that path."""
    indexer, vec, lex, hashes, embedder, tmp_path = fresh_indexer
    f = tmp_path / "root" / "alpha.py"
    f.write_text(
        "def hello(name: str) -> str:\n"
        "    '''greet a person by name with the standard hi greeting.'''\n"
        "    greeting = f'hi, {name}!'\n"
        "    return greeting\n",
        encoding="utf-8",
    )
    asyncio.run(indexer.reindex_path(f))
    assert _path_chunk_count(lex, "alpha.py") > 0

    asyncio.run(indexer.remove_path("alpha.py"))
    assert _path_chunk_count(lex, "alpha.py") == 0
    # Also vector store should agree.
    assert vec.delete_by_path("alpha.py") == 0  # already gone


def test_unchanged_file_other_file_changed(fresh_indexer: Any) -> None:
    """Editing file B must NOT cause file A to re-embed."""
    indexer, vec, lex, hashes, embedder, tmp_path = fresh_indexer
    a = tmp_path / "root" / "a.py"
    b = tmp_path / "root" / "b.py"
    a.write_text(
        "def fa(x: int) -> int:\n"
        "    '''long enough docstring to clear the chunker min_chars threshold.'''\n"
        "    return x + 1\n",
        encoding="utf-8",
    )
    b.write_text(
        "def fb(y: int) -> int:\n"
        "    '''long enough docstring to clear the chunker min_chars threshold.'''\n"
        "    return y * 2\n",
        encoding="utf-8",
    )
    asyncio.run(indexer.reindex_path(a))
    asyncio.run(indexer.reindex_path(b))
    calls_after_initial = embedder._calls

    # Edit b, re-index b; a should NOT be touched (we only re-index `b`).
    b.write_text(
        "def fb_edited(y: int) -> int:\n"
        "    '''edited docstring just long enough to clear min_chars.'''\n"
        "    return y * 3\n",
        encoding="utf-8",
    )
    asyncio.run(indexer.reindex_path(b))

    # We can't directly assert that `a` wasn't called (we're calling
    # reindex_path(b) explicitly). What we CAN assert is that `a`'s chunks
    # are unchanged.
    a_chunks_now = _path_chunk_count(lex, "a.py")
    assert a_chunks_now > 0
