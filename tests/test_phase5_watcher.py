"""Phase 5: live watcher end-to-end.

We spawn a real watchdog Observer against a temp dir, edit files, and assert
the index reflects changes within a short deadline.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from code_rag.config import load_settings
from code_rag.embedders.fake import FakeEmbedder
from code_rag.graph.ingest import GraphIngester
from code_rag.indexing.indexer import Indexer
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.stores.kuzu_graph import KuzuGraphStore
from code_rag.stores.sqlite_lexical import SqliteLexicalStore
from code_rag.watcher.live import LiveWatcher


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
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

[watcher]
debounce_ms = 50
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))
    return root


@pytest.mark.asyncio
async def test_watcher_reindexes_on_create_and_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_config(tmp_path, monkeypatch)
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

    try:
        indexer = Indexer(
            settings, embedder, vec,
            lexical_store=lex,
            graph_store=GraphIngester(graph),
        )
        watcher = LiveWatcher(settings, indexer)
        task = asyncio.create_task(watcher.run())
        # Give the Observer a moment to subscribe to the directory.
        await asyncio.sleep(0.2)

        # --- create ---
        (root / "x.py").write_text(
            "def uniq_marker_42():\n    return 1\n", encoding="utf-8",
        )
        assert await _wait_until(lambda: lex.query("uniq_marker_42", k=5), timeout=3.0)

        # --- modify ---
        (root / "x.py").write_text(
            "def different_marker_99():\n    return 2\n", encoding="utf-8",
        )
        assert await _wait_until(lambda: lex.query("different_marker_99", k=5), timeout=3.0)
        # Old symbol must be purged.
        assert not lex.query("uniq_marker_42", k=5), "old symbol must disappear on modify"

        # --- delete ---
        (root / "x.py").unlink()
        assert await _wait_until(lambda: not lex.query("different_marker_99", k=5), timeout=3.0)

        watcher.request_stop()
        await asyncio.wait_for(task, timeout=5.0)
    finally:
        vec.close()
        lex.close()
        graph.close()


async def _wait_until(fn, *, timeout: float) -> bool:
    """Poll fn() until truthy or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if fn():
            return True
        await asyncio.sleep(0.05)
    return bool(fn())
