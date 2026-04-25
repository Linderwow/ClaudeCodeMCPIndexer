"""Phase 23: metrics + fsck."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from code_rag.config import load_settings
from code_rag.dynamic_roots import DynamicRoots
from code_rag.embedders.fake import FakeEmbedder
from code_rag.indexing.file_hash import FileHashRegistry
from code_rag.indexing.indexer import Indexer
from code_rag.ops import fsck, metrics_text
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.stores.sqlite_lexical import SqliteLexicalStore


def _make(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def f(): return 1\n", encoding="utf-8")
    (root / "b.py").write_text("def g(): return 2\n", encoding="utf-8")
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
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))


def _open_indexed(tmp_path: Path) -> tuple[ChromaVectorStore, SqliteLexicalStore]:
    settings = load_settings()
    emb = FakeEmbedder(dim=32)
    vec = ChromaVectorStore(settings.chroma_dir, settings.vector_store.collection,
                            settings.index_meta_path)
    lex = SqliteLexicalStore(settings.fts_path)
    meta = ChromaVectorStore.build_meta("fake", emb.model, emb.dim)
    vec.open(meta)
    lex.open()
    indexer = Indexer(settings, emb, vec, lexical_store=lex)
    asyncio.run(indexer.reindex_all())
    return vec, lex


# ---- metrics ---------------------------------------------------------------


def test_metrics_text_includes_chunks_and_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make(tmp_path, monkeypatch)
    vec, lex = _open_indexed(tmp_path)
    try:
        text = metrics_text(load_settings(), vec, lex)
        # Must include the headline gauges, both store-tagged variants, and a
        # zero drift since nothing got out of sync.
        assert "code_rag_chunks_total" in text
        assert 'store="vector"' in text
        assert 'store="lexical"' in text
        assert "code_rag_chunks_drift 0" in text
        # Embedder metadata gets surfaced.
        assert "code_rag_embedder_dim" in text
    finally:
        vec.close()
        lex.close()


def test_metrics_text_is_openmetrics_parseable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make(tmp_path, monkeypatch)
    vec, lex = _open_indexed(tmp_path)
    try:
        text = metrics_text(load_settings(), vec, lex)
        # Naive OpenMetrics check: every non-comment line is `name value` or
        # `name{labels} value` with no syntax errors.
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            tokens = line.rsplit(" ", 1)
            assert len(tokens) == 2, f"bad metric line: {line!r}"
            float(tokens[1])  # must parse as a number
    finally:
        vec.close()
        lex.close()


# ---- fsck ------------------------------------------------------------------


def test_fsck_clean_index_reports_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make(tmp_path, monkeypatch)
    vec, lex = _open_indexed(tmp_path)
    try:
        report = fsck(load_settings(), vec, lex)
        # A fresh index from `_open_indexed` has no errors. Warnings about
        # orphan file-hash rows are possible if the registry was opened in
        # the indexer; check just for ok=True.
        assert report.ok
    finally:
        vec.close()
        lex.close()


def test_fsck_detects_missing_dynamic_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make(tmp_path, monkeypatch)
    settings = load_settings()
    # Add a dynamic root pointing at a non-existent dir.
    dyn = DynamicRoots.load(settings.dynamic_roots_path)
    fake = tmp_path / "vanished_repo"
    fake.mkdir()
    dyn.add(fake)
    fake.rmdir()  # now it's gone — fsck should flag it.

    vec, lex = _open_indexed(tmp_path)
    try:
        report = fsck(settings, vec, lex)
        codes = [i.code for i in report.issues]
        assert "dynamic_root_missing" in codes
    finally:
        vec.close()
        lex.close()


def test_fsck_auto_fix_prunes_missing_dynamic_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make(tmp_path, monkeypatch)
    settings = load_settings()
    dyn = DynamicRoots.load(settings.dynamic_roots_path)
    fake = tmp_path / "ghost"
    fake.mkdir()
    dyn.add(fake)
    fake.rmdir()

    vec, lex = _open_indexed(tmp_path)
    try:
        report = fsck(settings, vec, lex, auto_fix=True)
        # auto-fix must have pruned the entry. Re-loading should show empty.
        assert any(f.code == "dynamic_root_missing" for f in report.fixed)
        dyn2 = DynamicRoots.load(settings.dynamic_roots_path)
        assert len(dyn2.entries) == 0
    finally:
        vec.close()
        lex.close()


def test_fsck_detects_orphan_file_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make(tmp_path, monkeypatch)
    settings = load_settings()
    vec, lex = _open_indexed(tmp_path)
    # Manually inject a file-hash row that doesn't correspond to anything
    # in the lexical store.
    reg = FileHashRegistry(settings.file_hashes_path)
    reg.open()
    reg.upsert("never_indexed.py", b"content")
    reg.close()
    try:
        report = fsck(settings, vec, lex, auto_fix=True)
        assert any(i.code == "orphan_file_hashes" for i in report.issues)
        # auto_fix prunes it.
        reg2 = FileHashRegistry(settings.file_hashes_path)
        reg2.open()
        try:
            assert "never_indexed.py" not in reg2.list_paths()
        finally:
            reg2.close()
    finally:
        vec.close()
        lex.close()
