"""Regression: reindex_path on a NON-EXISTENT target must NOT walk the
entire indexed tree. Editor save events sometimes fire on transiently
missing files (rename-in-flight); the watcher used to walk all 12k+
config-root files on each phantom event, eating 11+ minutes of CPU per
event and stacking up under the watcher's serial drain loop.

Fix in indexer.py:
    1. target doesn't exist  -> remove_path + return (O(1))
    2. target.is_file()       -> process just that file (O(1))
    3. target is directory    -> rglob INSIDE target only (O(files-under-target))
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from code_rag.config import load_settings
from code_rag.embedders.fake import FakeEmbedder
from code_rag.indexing.indexer import Indexer
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.stores.sqlite_lexical import SqliteLexicalStore


def _make_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_decoy_files: int,
) -> None:
    """Set up a config with N pre-indexed code files in a 'noise' subtree
    that should NEVER be walked when reindex_path is given a different
    target. Tests that the fix scopes the walk correctly."""
    root = tmp_path / "repo"
    noise = root / "noise"
    noise.mkdir(parents=True)
    for i in range(n_decoy_files):
        (noise / f"decoy_{i}.py").write_text(
            f"def decoy_{i}():\n    return {i}\n", encoding="utf-8",
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


def test_reindex_path_on_missing_target_is_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug: reindex_path on a non-existent file used to walk the entire
    config-root tree filtering by `is_relative_to(target)`. With 200 noise
    files this should still complete in well under a second; before the
    fix it scaled with the size of every configured root."""
    _make_settings(tmp_path, monkeypatch, n_decoy_files=200)
    settings = load_settings()
    emb = FakeEmbedder(dim=32)
    vec = ChromaVectorStore(settings.chroma_dir, settings.vector_store.collection,
                            settings.index_meta_path)
    lex = SqliteLexicalStore(settings.fts_path)
    meta = ChromaVectorStore.build_meta("fake", emb.model, emb.dim)
    vec.open(meta)
    lex.open()
    try:
        indexer = Indexer(settings, emb, vec, lexical_store=lex)
        # Index everything once so list_paths is non-empty.
        asyncio.run(indexer.reindex_all())

        # Now hit reindex_path with a target that DOESN'T EXIST. Pre-fix this
        # would walk all 200 decoy files; post-fix it short-circuits.
        ghost = tmp_path / "repo" / "ghost_dir" / "vanished.py"
        t0 = time.monotonic()
        stats = asyncio.run(indexer.reindex_path(ghost))
        dur = time.monotonic() - t0

        # Hard upper bound: must finish in well under a second on a phantom.
        # Pre-fix this scaled with N_decoy_files; post-fix it's O(1) modulo
        # one rglob walk that doesn't happen because target doesn't exist.
        assert dur < 0.5, f"phantom reindex_path took {dur:.3f}s; expected <0.5s"
        # Stats must show NO files visited (the file doesn't exist).
        assert stats.files_seen == 0
        assert stats.files_indexed == 0
    finally:
        vec.close()
        lex.close()


def test_reindex_path_on_subtree_only_walks_subtree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reindex on a directory should walk only files INSIDE that directory,
    not every file in every configured root. We set up 100 decoy files
    outside the target subtree; reindex_path on the target must touch
    exactly the 2 files inside it."""
    _make_settings(tmp_path, monkeypatch, n_decoy_files=100)
    settings = load_settings()
    target = tmp_path / "repo" / "target_subtree"
    target.mkdir()
    (target / "real_a.py").write_text("def real_a(): return 1\n", encoding="utf-8")
    (target / "real_b.py").write_text("def real_b(): return 2\n", encoding="utf-8")
    emb = FakeEmbedder(dim=32)
    vec = ChromaVectorStore(settings.chroma_dir, settings.vector_store.collection,
                            settings.index_meta_path)
    lex = SqliteLexicalStore(settings.fts_path)
    meta = ChromaVectorStore.build_meta("fake", emb.model, emb.dim)
    vec.open(meta)
    lex.open()
    try:
        indexer = Indexer(settings, emb, vec, lexical_store=lex)
        stats = asyncio.run(indexer.reindex_path(target))
        # Exactly the 2 files in the target subtree, not the 100 decoys.
        assert stats.files_seen == 2, f"expected 2 visited, got {stats.files_seen}"
        assert stats.files_indexed == 2
    finally:
        vec.close()
        lex.close()


def test_reindex_path_missing_target_purges_prior_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a watched file is deleted (target.exists() False) AND it had
    been indexed previously, reindex_path must purge its chunks — not just
    no-op. Otherwise stale chunks linger forever."""
    _make_settings(tmp_path, monkeypatch, n_decoy_files=0)
    settings = load_settings()
    target = tmp_path / "repo" / "doomed.py"
    target.write_text("def doomed():\n    return 1\n", encoding="utf-8")
    emb = FakeEmbedder(dim=32)
    vec = ChromaVectorStore(settings.chroma_dir, settings.vector_store.collection,
                            settings.index_meta_path)
    lex = SqliteLexicalStore(settings.fts_path)
    meta = ChromaVectorStore.build_meta("fake", emb.model, emb.dim)
    vec.open(meta)
    lex.open()
    try:
        indexer = Indexer(settings, emb, vec, lexical_store=lex)
        # Index doomed.py.
        asyncio.run(indexer.reindex_all())
        before = vec.count()
        assert before > 0

        # Delete the file off disk + fire reindex_path on the now-missing path.
        target.unlink()
        stats = asyncio.run(indexer.reindex_path(target))

        # Chunks for doomed.py must be gone.
        after = vec.count()
        assert after == 0, f"expected 0 chunks after delete, got {after}"
        assert stats.paths_purged >= 1
    finally:
        vec.close()
        lex.close()
